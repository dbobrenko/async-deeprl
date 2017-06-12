from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Flatten, Dense, Input, Conv2D
from keras.models import Model

os.environ["KERAS_BACKEND"] = "tensorflow"


class QlearningAgent:
    def __init__(self, session, action_size, h, w, channels, opt=tf.train.AdamOptimizer(1e-4)):
        """Creates Q-Learning agent
        :param session: tensorflow session
        :param action_size: (int) length of action space
        :param h: (int) input image height
        :param w: (int) input image width
        :param channels: (int) number of image channels
        :param opt: tensorflow optimizer (by default: Adam optimizer)"""
        self.action_size = action_size
        self.opt = opt
        self.global_step = tf.Variable(0, name='frame', trainable=False)
        self.frame_inc_op = self.global_step.assign_add(1, use_locking=True)
        K.set_session(session)
        self.sess = session
        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.reward = tf.placeholder('float32', [None], name='reward')
            model, self.state, self.q_values = self._build_model(h, w, channels)
            self.weights = model.trainable_weights
        with tf.variable_scope('optimizer'):
            # Zero all actions, except one that was performed
            action_onehot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            # Predict expected future reward for performed action
            q_value = tf.reduce_sum(tf.multiply(self.q_values, action_onehot), reduction_indices=1)
            # Define squared mean loss function: (y - y_)^2
            self.loss = tf.reduce_mean(tf.square(self.reward - q_value))
            # Compute gradients w.r.t. weights
            grads = tf.gradients(self.loss, self.weights)
            # Apply gradient norm clipping
            grads, _ = tf.clip_by_global_norm(grads, 40.)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = opt.apply_gradients(grads_vars)
        with tf.variable_scope('target_network'):
            target_m, self.target_state, self.target_q_values = self._build_model(h, w, channels)
            target_w = target_m.trainable_weights
        with tf.variable_scope('target_update'):
            self.target_update = [target_w[i].assign(self.weights[i])
                                  for i in range(len(target_w))]

    @property
    def frame(self):
        """:return: global frame"""
        return self.global_step.eval(session=self.sess)

    def update_target(self):
        """Synchronizes shared target with local weights"""
        self.sess.run(self.target_update)

    def predict_rewards(self, state):
        """Predicts reward per action for given state.
        :param state: array with shape=[batch_size, num_channels, width, height]
        :type state: nd.array
        :return: rewards for each action (e.g. [1.2, 5.0, 0.4])
        :rtype: list"""
        return self.sess.run(self.q_values, {self.state: state}).flatten()

    def predict_target(self, state):
        """Predicts maximum action's reward for given state with target network
        :param state: array with shape=[batch_size, num_channels, width, height]
        :type state: nd.array
        :return: maximum expected reward
        :rtype: float"""
        return np.max(self.sess.run(self.target_q_values, {self.target_state: state}).flatten())

    def train(self, states, actions, rewards):
        """Trains online network on given states and rewards batch
        :param states: batch with screens with shape=[N, H, W, C]
        :param actions: batch with actions indices, e.g. [1, 4, 0, 2]
        :param rewards: batch with received rewards from given actions (e.g. [0.43, 0.5, -0.1, 1.0])
        :type states: nd.array
        :type actions: list
        :type rewards: list"""
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.reward: rewards
        })

    def frame_increment(self):
        """Increments global frame counter"""
        self.frame_inc_op.eval(session=self.sess)

    def _build_model(self, h, w, channels, fc3_size=256):
        """Builds DQN model (Mnih et al., 2015)
        :param h: input layer height
        :param w: input layer width
        :param channels: input layer number of channels
        :param fc3_size: 3rd fully connected layer size (common: 256, 512)"""
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        inputs = Input(shape=(h, w, channels,))
        model = Conv2D(activation="relu", filters=16, kernel_size=(8, 8), padding="same",
                       strides=(4, 4), data_format="channels_last")(inputs)
        model = Conv2D(kernel_size=(4, 4), data_format="channels_last", strides=(2, 2),
                       filters=32, activation="relu", padding="same")(model)
        model = Flatten()(model)
        model = Dense(units=fc3_size, activation='relu')(model)
        out = Dense(units=self.action_size, activation='linear')(model)
        model = Model(inputs=inputs, outputs=out)
        qvalues = model(state)
        return model, state, qvalues


class AgentSummary:
    """Helper wrapper for summary tensorboard logging"""

    def __init__(self, logdir, agent, env_name):
        """ :param logdir: path to the log directory
            :param agent: agent class-wrapper
            :param env_name: environment name"""
        with tf.variable_scope('summary'):
            self.agent = agent
            self.last_time = time.time()
            self.last_frames = self.agent.frame
            scalar_tags = ['fps', 'episode_avg_reward', 'avg_q_value',
                           'epsilon', 'total_frame_step']
            self.writer = tf.summary.FileWriter(logdir, self.agent.sess.graph)
            self.summary_vars = {}
            self.summary_ph = {}
            self.summary_ops = {}
            for k in scalar_tags:
                self.summary_vars[k] = tf.Variable(0.)
                self.summary_ph[k] = tf.placeholder('float32', name=k)
                self.summary_ops[k] = tf.summary.scalar("%s/%s" % (env_name, k), self.summary_vars[k])
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.summary.merge(list(self.summary_ops.values()))

    def write_summary(self, tags):
        """Writes summary to TensorBoard.
        :param tags: summary dictionary with with keys:
                     'episode_avg_reward': average episode reward;
                     'avg_q_value'       : average episode Q-value;
                     'epsilon'           : current epsilon values;
                     'total_frame_step'  : current frame step.
        :type tags: dict"""
        tags['fps'] = (self.agent.frame - self.last_frames) / (time.time() - self.last_time)
        self.last_time = time.time()
        self.last_frames = self.agent.frame
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in tags.items()})
        summary = self.agent.sess.run(self.summary_op,
                                      {self.summary_vars[k]: v for k, v in tags.items()})
        self.writer.add_summary(summary, global_step=self.agent.frame)
