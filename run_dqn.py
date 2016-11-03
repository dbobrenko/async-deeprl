# Python 2 and 3 compatibility
from __future__ import print_function
from __future__ import division
from six.moves import range

import tensorflow as tf
import numpy as np
import threading as th
import random
from datetime import datetime
from asyncrl.environment import GymWrapperFactory
from asyncrl.agent import QlearningAgent, AgentSummary
import time
import os

#random.seed(201)  # To reproduce minimum epsilon sampling
# Distribution of epsilon exploration chances (0.1 = 0.4; 0.01 = 0.3; 05 = 0.3)
EPS_MIN_SAMPLES = 4 * [0.1] + 3 * [0.01] + 3 * [0.5]

# Configurations
tf.app.flags.DEFINE_integer("threads", 8, "Number of threads to use")
tf.app.flags.DEFINE_boolean("gpu", False, "Use CPU or GPU for training (default is CPU)")
# Training settings
tf.app.flags.DEFINE_integer("total_frames", 40000000, "Total frames (across all threads)")
tf.app.flags.DEFINE_integer("update_interval", 40000, "Update target network after X frames")
tf.app.flags.DEFINE_float("eps_steps", 4000000.0, "Decrease epsilon over X frames")
tf.app.flags.DEFINE_float("eps_start", 1.0, "Starting epsilon (initial exploration chance)")
tf.app.flags.DEFINE_float("gamma", 0.99, "Gamma discount factor")
tf.app.flags.DEFINE_integer("tmax", 5, "Maximum batch size")
tf.app.flags.DEFINE_integer("action_repeat", 4, "Applies last action to X next frames")
tf.app.flags.DEFINE_integer("memory_len", 4, "Memory length - number of stacked input images")
# Environment settings
tf.app.flags.DEFINE_string("env", 'Breakout-v0', "Environment name (available all OpenAI Gym environments)")
tf.app.flags.DEFINE_boolean("render", False, "Render frames? Significantly slows down training process")
tf.app.flags.DEFINE_integer("width", 84, "Screen image width")
tf.app.flags.DEFINE_integer("height", 84, "Screen image height")
# Logging
tf.app.flags.DEFINE_integer("test_iter", 3, "Number of test iterations. Used for logging.")
tf.app.flags.DEFINE_string("logdir", 'logs/', "Path to the directory used for checkpoints and loggings")
tf.app.flags.DEFINE_integer("log_interval", 40000, "Log and checkpoint every X frame")
# Evaluation
tf.app.flags.DEFINE_boolean("eval", False, "Disables training, evaluates agent's performance")
tf.app.flags.DEFINE_string("evaldir", 'eval/', "Path to the evaluation logging")
tf.app.flags.DEFINE_integer("eval_iter", 20, "Number of evaluation episodes")
# Optimizer
tf.app.flags.DEFINE_float("lr", 1e-4, "Starting learning rate")
FLAGS = tf.app.flags.FLAGS
# Hide all GPUs for current process if CPU was chosen
if not FLAGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
global_epsilons = [0.] * FLAGS.threads
training_finished = False


def update_epsilon(frames, eps_steps, eps_min):
    """Anneals epsilon based on current frame"""
    eps = FLAGS.eps_start - (frames / eps_steps) * (FLAGS.eps_start - eps_min)
    return eps if eps > eps_min else eps_min


def evaluate():
    envwrap = GymWrapperFactory.make(FLAGS.env,
                                     actrep=FLAGS.action_repeat,
                                     memlen=FLAGS.memory_len,
                                     w=FLAGS.width,
                                     h=FLAGS.height)
    with tf.Session() as sess:
        agent = QlearningAgent(session=sess,
                               lr=FLAGS.lr,
                               action_size=envwrap.action_size,
                               h=FLAGS.height,
                               w=FLAGS.width,
                               channels=FLAGS.memory_len)
        sess.run(tf.initialize_all_variables())
        if not os.path.exists(FLAGS.logdir):
            print('ERROR! No', FLAGS.logdir, 'folder found!')
            return
        ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
        if ckpt is not None:
            tf.train.Saver().restore(sess, ckpt)
            print('Restoring session from %s' % ckpt)
        else:
            print('ERROR! No checkpoint found at', FLAGS.logdir)
            return
        envwrap.env.monitor.start(os.path.join(FLAGS.evaldir, FLAGS.env))
        total_reward = 0
        for _ in range(FLAGS.eval_iter):
            s = envwrap.reset()
            terminal = False
            while not terminal:
                reward_per_action = agent.predict_rewards(s)
                s, r, terminal, info = envwrap.step(np.argmax(reward_per_action), test=True)
                total_reward += r
                envwrap.render()
        envwrap.env.monitor.close()
        print('Evaluation finished.')
        print('Average reward per episode: %0.4f' % (total_reward / FLAGS.eval_iter))


def test(agent, env, episodes):
    ep_rewards = []
    ep_q = []
    for _ in range(episodes):
        ep_reward = 0
        s = env.reset()
        terminal = False
        while not terminal:
            reward_per_action = agent.predict_rewards(s)
            s, r, terminal, info = env.step(np.argmax(reward_per_action), test=True)
            ep_q.append(np.max(reward_per_action))
            ep_reward += r
        ep_rewards.append(ep_reward)
    return ep_rewards, ep_q


def train_async_dqn(agent, env, sess, agent_summary, saver, thread_idx=0):
    """Starts Asynchronous one/n-step Q-Learning.
    Can be used as a worker for threading.Thread
    :agent: agent.GymWrapper object or any other derived object
    :env: environment.GymEnvironment object or any derived wrapper
    :sess: tensorflow.Session
    :agent_summary: agent.AgentSummary object
    :saver: tensorflow.train.Saver object
    :thread_idx: thread index (thread with index=0 used for target network update and logging)"""
    global global_epsilons
    eps_min = random.choice(EPS_MIN_SAMPLES)
    epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
    print('Thread: %d. Sampled min epsilon: %f' % (thread_idx, eps_min))
    last_logging = agent.frame
    last_target_update = agent.frame
    terminal = True
    # Training loop:
    while agent.frame < FLAGS.total_frames:
        batch_states, batch_rewards, batch_actions = [], [], []
        if terminal:
            terminal = False
            screen = env.reset_random()
        # Batch update loop:
        while not terminal and len(batch_states) < FLAGS.tmax:
            # Increment shared frame counter
            agent.frame_increment()
            batch_states.append(screen)
            # Exploration vs Exploitation, E-greedy action choose
            if random.random() < epsilon:
                action_index = random.randrange(agent.action_size)
            else:
                reward_per_action = agent.predict_rewards(screen)
                # Choose an action index with maximum expected reward
                action_index = np.argmax(reward_per_action)
            # Execute an action and receive new state, reward for action
            screen, reward, terminal, _ = env.step(action_index)
            reward = np.clip(reward, -1, 1)
            # 1-step Q-Learning: add discounted expected future reward
            if not terminal:
                reward += FLAGS.gamma * agent.predict_target(screen)
            batch_rewards.append(reward)
            batch_actions.append(action_index)
        # Apply asynchronous gradient update to shared agent
        agent.train(np.vstack(batch_states), batch_actions, batch_rewards)
        # Anneal epsilon
        epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
        global_epsilons[thread_idx] = epsilon  # Logging
        # Logging and target network update
        if thread_idx == 0:
            if agent.frame - last_target_update >= FLAGS.update_interval:
                last_target_update = agent.frame
                agent.update_target()
            if agent.frame - last_logging >= FLAGS.log_interval and terminal:
                last_logging = agent.frame
                saver.save(sess, os.path.join(FLAGS.logdir, "sess.ckpt"), global_step=agent.frame)
                print('Session saved to %s' % FLAGS.logdir)
                episode_rewards, episode_q = test(agent, env, episodes=FLAGS.test_iter)
                avg_r = np.mean(episode_rewards)
                avg_q = np.mean(episode_q)
                avg_eps = np.mean(global_epsilons)
                print("%s. Avg.Ep.R: %.4f. Avg.Ep.Q: %.2f. Avg.Eps: %.2f. T: %d" %
                      (str(datetime.now())[11:19], avg_r, avg_q, avg_eps, agent.frame))
                agent_summary.write_summary({
                    'total_frame_step': agent.frame,
                    'episode_avg_reward': avg_r,
                    'avg_q_value': avg_q,
                    'epsilon': avg_eps
                })
    global training_finished
    training_finished = True
    print('Thread %d. Training finished. Total frames: %s' % (thread_idx, agent.frame))


def run(worker):
    """Launches worker asynchronously in 'FLAGS.threads' threads"""
    print('Starting. Threads:', FLAGS.threads)
    processes = []
    envs = []
    for _ in range(FLAGS.threads):
        env = GymWrapperFactory.make(FLAGS.env,
                                     actrep=FLAGS.action_repeat,
                                     memlen=FLAGS.memory_len,
                                     w=FLAGS.width,
                                     h=FLAGS.height)
        envs.append(env)

    with tf.Session() as sess:
        agent = QlearningAgent(session=sess,
                               lr=FLAGS.lr,
                               action_size=envs[0].action_size,
                               h=FLAGS.height,
                               w=FLAGS.width,
                               channels=FLAGS.memory_len)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)
        sess.run(tf.initialize_all_variables())
        if not os.path.exists(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
            ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
            if ckpt is not None:
                saver.restore(sess, ckpt)
                print('Restoring session from %s' % ckpt)
        summary = AgentSummary(FLAGS.logdir, agent, FLAGS.env)
        for i in range(FLAGS.threads):
            processes.append(th.Thread(target=worker, args=(agent, envs[i], sess, summary, saver, i,)))
        for p in processes:
            p.daemon = True
            p.start()
        while not training_finished:
            if FLAGS.render:
                for i in range(FLAGS.threads):
                    envs[i].render()
            time.sleep(.01)
        for p in processes:
            p.join()


if __name__ == '__main__':
    if FLAGS.eval:
        evaluate()
    else:
        run(train_async_dqn)
