# Python 2 and 3 compatibility
from __future__ import print_function
from __future__ import division

try:
    range = xrange
except NameError:
    pass
import tensorflow as tf
import numpy as np
from random import random
import threading as th
import random
from datetime import datetime
from environment import GymEnvironment
from agent import QlearningAgent, AgentSummary
import time
import os

# Distribution of epsilon exploration chances (0.1 = 0.4; 0.01 = 0.3; 05 = 0.3)
EPS_MIN_SAMPLES = 4 * [0.1] + 3 * [0.01] + 3 * [0.5]

# Configurations
tf.app.flags.DEFINE_integer("threads", 8, "Number of threads to use")
tf.app.flags.DEFINE_boolean("gpu", False, "Use CPU or GPU for training (default is CPU)")
# Training settings
#tf.app.flags.DEFINE_boolean("nstep", False, "Use N-step Q-Learning instead of 1-step.")
tf.app.flags.DEFINE_boolean("play", False, "Disables training and logging, shows playing agents")
tf.app.flags.DEFINE_integer("total_frames", 80000000, "Total frames (across all threads)")
tf.app.flags.DEFINE_integer("update_interval", 40000, "Update target network after X frames")
tf.app.flags.DEFINE_float("eps_steps", 4000000.0, "Decrease epsilon over X frames")
tf.app.flags.DEFINE_float("eps_start", 1.0, "Starting epsilon (initial exploration chance)")
tf.app.flags.DEFINE_float("gamma", 0.99, "Gamma discount factor")
tf.app.flags.DEFINE_integer("t_max", 5, "Maximum batch size")
tf.app.flags.DEFINE_integer("action_repeat", 4, "Applies last action to X next frames")
tf.app.flags.DEFINE_integer("memory_len", 4, "Memory length - number of stacked input images")
# Environment settings
tf.app.flags.DEFINE_string("env_name", 'Breakout-v0', "Environment name (available all OpenAI Gym environments)")
tf.app.flags.DEFINE_boolean("render", False, "Render frames? Significantly slows training process")
tf.app.flags.DEFINE_integer("width", 84, "Screen image width")
tf.app.flags.DEFINE_integer("height", 84, "Screen image height")
# Logging
tf.app.flags.DEFINE_string("logdir", 'logs/', "Path to the directory used for checkpoints and loggings")
tf.app.flags.DEFINE_integer("log_interval", 20000, "Log and checkpoint every X frame")
# Optimizer
tf.app.flags.DEFINE_float("lr", 1e-4, "Starting learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout")
tf.app.flags.DEFINE_float("rms_decay", 0.99, "RMSProp decay factor")
tf.app.flags.DEFINE_float("gradient_clip", 40, "Gradient norm clipping threshold")
FLAGS = tf.app.flags.FLAGS

# Hide all GPUs for current process if CPU was chosen
if not FLAGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Thread shared lists for logging
episode_q = []
episode_rewards = []
global_epsilons = [0.] * FLAGS.threads
training_finished = False

def update_epsilon(frames, eps_steps, eps_min):
    """Anneals epsilon based on current frame"""
    eps = FLAGS.eps_start - (frames / eps_steps) * (FLAGS.eps_start - eps_min)
    return eps if eps > eps_min else eps_min


def play(agent, env, sess=None, summary=None, saver=None, thread_idx=None):
    """Plays without training.
    :agent: agent.QlearningAgent object or any derived object
    :env: environment.GymEnvironment object or any derived wrapper
    Rest parameters are redundant and provided for compatibility
    with async_q_learner function signature"""
    s = env.reset()
    while True:
        reward_per_action = agent.predict_rewards(s)
        action_index = np.argmax(reward_per_action)
        s, r, term, info = env.step(action_index)
        time.sleep(.1)
        if term:
            s = env.reset()


def async_q_learner(agent, env, sess, agent_summary, saver, thread_idx=0):
    """Starts asynchronous 1-step Q-Learning.
    Can be used as a worker for threading.Thread
    :agent: agent.QlearningAgent object or any derived object
    :env: environment.GymEnvironment object or any derived wrapper
    :sess: tensorflow.Session
    :agent_summary: agent.AgentSummary object
    :saver: tensorflow.train.Saver object
    :thread_idx: thread number (thread with index=0 used for target network update and logging)"""
    # Global shared lists for logging summary
    global episode_q
    global episode_rewards
    episode_reward = 0
    eps_min = random.choice(EPS_MIN_SAMPLES)
    epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
    print('Thread: %d. Sampled min epsilon: %f' % (thread_idx, eps_min))
    s = env.reset()
    last_logging = agent.frame
    last_target_update = agent.frame
    # Training loop:
    while agent.frame < FLAGS.total_frames:
        batch_states, batch_rewards, batch_actions = [], [], []
        # Batch update loop:
        while len(batch_states) < FLAGS.t_max:
            # Increment shared frame counter
            agent.frame_increment()
            batch_states.append(s)
            # Exploration vs Exploitation, E-greedy action choose
            if random.random() < epsilon:
                reward_per_action = np.random.rand(agent.action_size)
            else:
                reward_per_action = agent.predict_rewards(s)
                episode_q.append(np.max(reward_per_action))  # Logging
            # Get action index with maximum expected future reward
            action_index = np.argmax(reward_per_action)
            # Execute an action and receive new state, reward for action
            s, r, term, info = env.step(action_index)
            episode_reward += r  # Logging
            r = np.clip(r, -1, 1)
            # Check for Atari end of round
            round_end = 'round_end' in info and info['round_end']
            # 1-step Q-Learning: add discounted expected future reward
            if not term and not round_end:# and not FLAGS.nstep:
                r += FLAGS.gamma * agent.predict_target(s)
            batch_rewards.append(r)
            batch_actions.append(action_index)
            if term:
                episode_rewards.append(episode_reward)  # Logging
                s = env.reset()
                episode_reward = 0
                break
            # Used in Atari games for splitting rounds into separate games
            if round_end:
                term = True
                break
        # Apply asynchronous gradient update to shared agent
        agent.train(np.vstack(batch_states), batch_actions, batch_rewards)
        # Anneal epsilon
        epsilon = update_epsilon(agent.frame, FLAGS.eps_steps, eps_min)
        global_epsilons[thread_idx] = epsilon # Logging
        # Logging and target network update
        if thread_idx == 0:
            if agent.frame - last_target_update >= FLAGS.update_interval:
                last_target_update = agent.frame
                agent.update_target()
            if agent.frame - last_logging >= FLAGS.log_interval:
                last_logging = agent.frame
                avg_r = np.mean(episode_rewards or 0.)
                avg_q = np.mean(episode_q or 0.)
                avg_eps = np.mean(global_epsilons)
                print("%s. Avg.Ep.R: %.4f. Avg.Ep.Q: %.2f. Avg.Eps: %.2f. T: %d" %
                      (str(datetime.now())[11:19], avg_r, avg_q, epsilon, agent.frame))
                saver.save(sess, os.path.join(FLAGS.logdir, "sess.ckpt"), global_step=agent.frame)
                print('Session saved to %s' % FLAGS.logdir)
                agent_summary.write_summary({
                    'total_frame_step': agent.frame,
                    'episode_avg_reward': avg_r,
                    'avg_q_value': avg_q,
                    'epsilon': avg_eps,
                    'learning_rate': agent.lr
                })
                # Clear shared logs
                del episode_q[:]
                del episode_rewards[:]
    global training_finished
    training_finished = True
    print('Thread %d. Training finished. Total frames: %s' % (thread_idx, agent.frame))


def run(worker):
    """Launches worker asynchronously in 'FLAGS.threads' threads"""
    print('Starting. Threads:', FLAGS.threads)
    processes = []
    envs = []
    for _ in range(FLAGS.threads):
        envs.append(GymEnvironment(action_repeat=FLAGS.action_repeat,
                                   memory_len=FLAGS.memory_len,
                                   w=FLAGS.width,
                                   h=FLAGS.height,
                                   name=FLAGS.env_name))
    agent = QlearningAgent(lr=FLAGS.lr,
                           action_size=envs[0].action_size,
                           channels=FLAGS.memory_len,
                           w=FLAGS.width,
                           h=FLAGS.height,
                           gradient_clip=FLAGS.gradient_clip,
                           rms_decay=FLAGS.rms_decay,
                           total_frames=FLAGS.total_frames)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)
    init_op = tf.initialize_all_variables()
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    with tf.Session() as sess:
        sess.run(init_op)
        ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
        if ckpt != None:
            saver.restore(sess, ckpt)
            print('Restoring session from %s' % ckpt)
        agent.init(sess)
        summary = AgentSummary(FLAGS.logdir, agent, FLAGS.env_name)
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
    if FLAGS.play:
        FLAGS.render = True
        run(play)
    else:
        run(async_q_learner)
