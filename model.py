from __future__ import division

try:
    range = xrange
except NameError:
    pass
import tensorflow as tf
import numpy as np
from math import ceil


class DQNModel:
    def __init__(self, lr=0.001, ch=4, h=84, w=84, action_size=3, fc3_size=256, dropout=0.5,
                 gradient_clip=40, rms_decay=0.99, use_locking=True, global_step=None):
        """Creates Deep Q-Network.
        :lr: tensorflow Variable or float - learning rate.
        :action_size: size of action space.
        :ch: number of input image channels.
        :h: input image height.
        :w: input image width.
        :fc3_size: number of units in 3rd fully-connected layer
        :dropout: keep probability applied to the 3rd fc layer
        :gradient_clip: norm gradient clipping
        :rms_decay: RMSProp decay parameter
        """
        self.lr = lr
        self.dropout = dropout
        self.action_size = action_size
        self.sess = None
        with tf.variable_scope('input'):
            self._x_ph = tf.placeholder('float32', [None, h, w, ch], name='state')
            self._a_ph = tf.placeholder('int32', [None], name='action')
            self._y_ph = tf.placeholder('float32', [None], name='reward')

        with tf.variable_scope('hyperparameters'):
            self._dropout_ph = tf.placeholder('float32', name='dropout')
            self._lr_ph = tf.placeholder('float32', name='learning_rate')

        with tf.variable_scope('network'):
            self.W = {
                'W1': self._init_conv([8, 8, ch, 16], name='W1'),
                'b1': self._init_dense([16], name='b1'),
                'W2': self._init_conv([4, 4, 16, 32], name='W2'),
                'b2': self._init_dense([32], name='b2'),
                # stride steps = 4 and 2, so 4 * 2 = 8
                'W3': self._init_dense([int(ceil(w / 8) * ceil(h / 8) * 32), fc3_size], name='W3'),
                'b3': self._init_dense([fc3_size], name='b3'),
                'W4': self._init_dense([fc3_size, self.action_size], name='W4'),
                'b4': self._init_dense([self.action_size], name='b4')
            }
            self._q_values = self._create_model()

        with tf.variable_scope('optimizer'):
            action_onehot = tf.one_hot(self._a_ph, self.action_size, 1.0, 0.0, name='action_onehot')
            action_q = tf.reduce_sum(tf.mul(self._q_values, action_onehot), reduction_indices=1)
            self._loss = tf.reduce_mean(tf.square(self._y_ph - action_q))
            self._opt = tf.train.RMSPropOptimizer(self._lr_ph,
                                                  decay=rms_decay,
                                                  use_locking=use_locking)
            self._gradvals = self._opt.compute_gradients(self._loss, self.W.values())
            gradvals_clip = []
            for grad, var in self._gradvals:
                gradvals_clip.append((tf.clip_by_norm(grad, gradient_clip), var))
            self._train_op = self._opt.apply_gradients(gradvals_clip, global_step=global_step)

    def predict(self, state):
        """:state : array with shape=[batch_size, num_channels, width, height]"""
        feed = {self._x_ph: state, self._dropout_ph: 1.0}
        return self.sess.run(self._q_values, feed_dict=feed)

    def train(self, states, actions, discounted_rewards, lr=None):
        """ Minimizes cost
        :states: numpy array with input state
        :actions: vectorized action list, e.g. [1, 0, 0]
        :discounted_rewards: scalar reward value for chosen action
        :lr: float, learning rate. If None is passed, will be used lr passed to the c-tor."""
        if lr is None:
            lr = self.lr
        self.sess.run(self._train_op, feed_dict={
            self._lr_ph: lr,
            self._x_ph: states,
            self._a_ph: actions,
            self._y_ph: discounted_rewards,
            self._dropout_ph: self.dropout
        })

    def _create_model(self):
        # Convolution Layers
        with tf.variable_scope('conv1'):
            conv1 = self._conv2d(self._x_ph, self.W['W1'], self.W['b1'], 4)
        with tf.variable_scope('conv2'):
            conv2 = self._conv2d(conv1, self.W['W2'], self.W['b2'], 2)
        # Fully connected layers
        with tf.variable_scope('fc3'):
            fc3 = tf.reshape(conv2, [-1, self.W['W3'].get_shape().as_list()[0]])
            fc3 = tf.add(tf.matmul(fc3, self.W['W3']), self.W['b3'])
            fc3 = tf.nn.relu(fc3)
            fc3 = tf.nn.dropout(fc3, self._dropout_ph)
        # Output, class prediction
        with tf.variable_scope('fc4'):
            out = tf.add(tf.matmul(fc3, self.W['W4']), self.W['b4'])
        return out

    def _init_conv(self, shape, name='conv2d'):
        """ "He" init (He et al. 2015): https://arxiv.org/pdf/1502.01852v1.pdf
        :param shape conv params: [filter_h, filter_w, input_channels, output_channels]"""
        input_size = shape[0] * shape[1] * shape[2]
        variance = np.sqrt(2. / input_size)
        return tf.Variable(tf.random_normal(shape, stddev=variance), name=name)

    def _init_dense(self, shape, name='dense'):
        """'Xavier init (Glorot et al. 2010):
        http://jmlr.org/proceedings/papers/v10/glorot10a/glorot10a.pdf"""
        variance = np.sqrt(2. / shape[0])
        return tf.Variable(tf.random_normal(shape, stddev=variance), name=name)

    def _conv2d(self, x, W, b, s=1):
        """Conv2D wrapper, with bias, relu activation and padding."""
        x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x, )
