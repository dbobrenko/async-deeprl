from skimage.transform import resize
from skimage.color import rgb2gray
import gym
import numpy as np

__custom_actions__ = {'Breakout-v0': [1, 4, 5],
                      'Pong-v0': [1, 2, 3]
                      }


class GymEnvironment:
    """Small wrapper around OpenAI Gym environment"""
    def __init__(self, name, action_repeat=4, memory_len=4, w=84, h=84, is_atari=True, verbose=True):
        self.env = gym.make(name)
        self.memory_len = memory_len
        self.W = w
        self.H = h
        self.action_repeat = action_repeat
        self.verbose = verbose
        self.stacked_s = None
        self.action_space = list(range(self.env.action_space.n))
        self.action_size = len(self.action_space)
        self.is_atari = is_atari
        for key in __custom_actions__:
            if key == self.env.spec.id:
                self.set_custom_actions(__custom_actions__[key])
                break

    def set_custom_actions(self, space):
        """:space: list of allowed actions (e.g. [1, 2, 3])"""
        self.action_space = space
        self.action_size = len(self.action_space)
        if self.verbose:
            print('Environment: %s. Action space: %s' % (self.env.spec.id, self.action_space))

    def preprocess(self, state):
        """:state: numpy array image
        "rtype: stacked frames with shape=[1, h, w, memory_len]"""
        s = resize(rgb2gray(state), (self.H, self.W))
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if self.stacked_s is not None:
            self.stacked_s = np.append(s, self.stacked_s[:, :, :, :self.memory_len - 1], axis=3)
        else:
            self.stacked_s = np.repeat(s, self.memory_len, axis=3)
        return self.stacked_s

    def reset(self):
        self.stacked_s = None
        return self.preprocess(self.env.reset())

    def step(self, action_index):
        """Executes action and repeat it on the next X frames
        :param action_index: action index (0-based)
        :rtype 4 elements tuple:
               new state,
               accumulated reward over skipped frames
               if it is a terminal state,
               info"""
        action = self.action_space[action_index]
        accum_reward = 0
        round_end = False
        if self.is_atari:
            start_lives = self.env.ale.lives()
        for _ in range(self.action_repeat):
            s, r, term, info = self.env.step(action)
            s = self.preprocess(s)
            accum_reward += r
            if term:
                break
            if self.is_atari and start_lives != self.env.ale.lives():
                round_end = True
                self.stacked_s = None
                break
        info['round_end'] = round_end
        return s, accum_reward, term, info

    def render(self):
        self.env.render()