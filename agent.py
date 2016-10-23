from __future__ import division
import tensorflow as tf
import numpy as np
from model import DQNModel
import time

class QlearningAgent:
    def __init__(self, lr, action_size, channels, w, h,
                 gradient_clip, rms_decay, total_frames):
        """Creates tensorflow Q-Learning agent"""
        self.lr_initial = lr
        self.action_size = action_size
        self.total_frames = total_frames
        self.global_step = tf.Variable(0, name='frame', trainable=False)
        self.frame_inc_op = self.global_step.assign_add(1, use_locking=True)
        self.sess = None
        with tf.variable_scope('train_model'):
            self.model = DQNModel(lr=lr, ch=channels, h=h, w=w, action_size=action_size,
                                  gradient_clip=gradient_clip, rms_decay=rms_decay)
        with tf.variable_scope('target_model'):
            self.model_target = DQNModel(lr=lr, ch=channels, h=h, w=w, action_size=action_size,
                                         gradient_clip=gradient_clip, rms_decay=rms_decay)
        with tf.variable_scope('train_to_target'):
            self.model2target_op = {}
            for k in self.model.W:
                self.model2target_op[k] = self.model_target.W[k].assign(self.model.W[k], use_locking=True)

    def init(self, session):
        """Sets session. Must be called before agent usage."""
        self.sess = session
        self.model.sess = self.sess
        self.model_target.sess = self.sess
        self.update_target()

    def update_target(self):
        """Updates target network weights"""
        for k in self.model.W.keys():
            self.model2target_op[k].eval(session=self.sess)

    def predict_rewards(self, state):
        """ Predicts epsilon greedy rewards per action for given state.
        :rtype : rewards for each action (e.g [1.2, 5.0, 0.4])"""
        return self.model.predict(state).flatten()

    def predict_target(self, state):
        """Predicts maximum action's reward for given state with target network"""
        return np.max(self.model_target.predict(state).flatten())

    def train(self, states, actions, discounted_rewards):
        """Trains network, linearly anneals learning rate"""
        return self.model.train(states, actions, discounted_rewards, lr=self.lr)

    def frame_increment(self):
        self.frame_inc_op.eval(session=self.sess)

    @property
    def lr(self):
        return self.lr_initial - self.lr_initial * (self.frame / self.total_frames)

    @property
    def frame(self):
        return self.global_step.eval(session=self.sess)


class AgentSummary:
    """Helper wrapper for summary tensorboard logging"""
    def __init__(self, logdir, agent, env_name):
        with tf.variable_scope('summary'):
            self.agent = agent
            self.last_write_time = time.time()
            self.last_write_frames = self.agent.frame
            scalar_tags = ['fps', 'episode_avg_reward', 'avg_q_value',
                           'epsilon', 'learning_rate', 'total_frame_step']
            self.writer = tf.train.SummaryWriter(logdir, self.agent.sess.graph)
            self.summary_vars = {}
            self.summary_ph = {}
            self.summary_ops = {}
            for k in scalar_tags:
                self.summary_vars[k] = tf.Variable(0.)
                self.summary_ph[k] = tf.placeholder('float32', name=k)
                self.summary_ops[k] = tf.scalar_summary("%s/%s" % (env_name, k), self.summary_vars[k])
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.merge_summary(list(self.summary_ops.values()))

    def write_summary(self, tags):
        """:tags : summary dictionary with with keys:
                   'episode_avg_reward', 'avg_q_value', 'epsilon',
                   'learning_rate', 'total_frame_step'"""
        tags['fps'] = (self.agent.frame - self.last_write_frames) / (time.time() - self.last_write_time)
        self.last_write_time = time.time()
        self.last_write_frames = self.agent.frame
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in tags.items()})
        summary = self.agent.sess.run(self.summary_op, {self.summary_vars[k]: v for k, v in tags.items()})
        self.writer.add_summary(summary, global_step=self.agent.frame)