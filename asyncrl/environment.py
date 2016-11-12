from scipy.misc import imresize
import gym
import numpy as np
import random

# Predefined custom action space in given games
__custom_actions__ = {'Breakout-v0': [1, 4, 5], # NoOp,, Right, Left
                      'Pong-v0': [1, 2, 3], # NoOp,, Right, Left
                      'SpaceInvaders-v0': [1, 2, 3], # NoOp,, Right, Left
                     }

class GymWrapperFactory:
    @staticmethod
    def make(name, actrep=4, memlen=4, w=84, h=84, random_start=30):
        env = gym.make(name)
        if hasattr(env, 'ale'): # Arcade Learning Environment
            return GymALE(env, actrep=actrep, memlen=memlen, w=w, h=h, random_start=random_start)
        else: # Basic Gym environment wrapper
            return GymWrapper(env, actrep=actrep, memlen=memlen,
                              w=w, h=h, random_start=random_start)

class GymWrapper:
    """A small wrapper around OpenAI Gym ALE"""
    def __init__(self, env, actrep=4, memlen=4, w=84, h=84, random_start=30):
        print('Creating wrapper around Gym Environment')
        self.env = env
        self.memlen = memlen
        self.W = w
        self.H = h
        self.actrep = actrep
        self.random_start = random_start
        if hasattr(self.env.action_space, "n"):
            self.action_space = list(range(self.env.action_space.n))
        elif hasattr(self.env.action_space, "shape"):
            self.action_space = list(np.eye(self.env.action_space.shape))
        else:
            raise ValueError("Environment %s: Unable to get action space size." % self.env.spec.id)
        self.action_size = len(self.action_space)
        self.stacked_s = None
        for key in __custom_actions__:
            if key == self.env.spec.id:
                self.set_custom_actions(__custom_actions__[key])
                break
        print('Environment: %s. Action space: %s' % (self.env.spec.id, self.action_space))

    def set_custom_actions(self, action_space):
        """Sets custom action space for current environment.
        :param action_space: list of allowed actions (e.g. [1, 2, 3])
        :type action_space: list"""
        self.action_space = action_space
        self.action_size = len(self.action_space)

    def preprocess(self, screen, new_game=False):
        """Converts to grayscale, resizes and stacks input screen.
        :param screen: array image in [0; 255] range with shape=[H, W, C]
        :param new_game: if True - repeats passed screen `memlen` times
                   otherwise - stacks with previous screens"
        :type screen: numpy.array
        :type new_game: bool
        :return: image in [0.0; 1.0] stacked with last `memlen-1` screens; 
                shape=[1, h, w, memlen]
        :rtype: numpy.array"""
        gray = screen.astype('float32').mean(2) # no need in true grayscale, just take mean
        # convert values into [0.0; 1.0] range
        s = imresize(gray, (self.W, self.H)).astype('float32') * (1. / 255)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if new_game or self.stacked_s is None:
            self.stacked_s = np.repeat(s, self.memlen, axis=3)
        else:
            self.stacked_s = np.append(s, self.stacked_s[:, :, :, :self.memlen - 1], axis=3)
        return self.stacked_s

    def reset(self):
        """Resets current game.
        :return: preprocessed first screen of the next game
        :rtype: numpy.array"""
        return self.preprocess(self.env.reset(), new_game=True)

    def reset_random(self):
        """Resets current game and skips `self.random_start` amount of frames.
        :return: preprocessed first screen of the next game
        :rtype: numpy.array"""
        s = self.env.reset()
        skip = random.randrange(self.random_start)
        for i in range(skip):
            s, r, term, info = self.env.step(random.choice(self.action_space))
            if term:
                print('WARNING! Random start frame skip have reached terminal state. Resetting.')
                s = self.env.reset()
        return self.preprocess(s, new_game=True)

    def step(self, action_index, test=False):
        """Executes action and repeat it on the next X frames
        :param action_index: action index (0-based)
        :type action_index: int
        :return 4 elements tuple:
               preprocessed and stacked screen,
               accumulated reward over skipped frames
               is terminal,
               info
        :rtype: tuple"""
        action = self.action_space[action_index]
        accum_reward = 0
        for _ in range(self.actrep):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if term:
                break
        return self.preprocess(s), accum_reward, term, info

    def render(self):
        """Renders current frame"""
        self.env.render()

class GymALE(GymWrapper):
    def __init__(self, env, actrep=4, memlen=4, w=84, h=84, random_start=30):
        GymWrapper.__init__(self, env=env, actrep=actrep, memlen=memlen, w=w, h=h,
                            random_start=random_start)
        print('Creating a wrapper around Arcade Learning Environment')
        self.has_lives = hasattr(self.env, 'ale') and hasattr(self.env.ale, 'lives')

    def preprocess(self, screen, new_game=False):
        luminance = screen.astype('float32').mean(2) # no need in true grayscale, just take mean
        # crop top/bottom Atari specific borders
        if self.env.spec.id == 'SpaceInvaders-v0':
            # crop only bottom in SpaceInvaders, due to flying object at the top of the screen
            luminance = luminance[:-15, :]
        else:
            luminance = luminance[36:-15, :]
        # convert into [0.0; 1.0]
        s = imresize(luminance, (self.W, self.H)).astype('float32') * (1. / 255)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if new_game or self.stacked_s is None:
            self.stacked_s = np.repeat(s, self.memlen, axis=3)
        else:
            self.stacked_s = np.append(s, self.stacked_s[:, :, :, :self.memlen - 1], axis=3)
        return self.stacked_s

    def step(self, action_index, test=False):
        action = self.action_space[action_index]
        accum_reward = 0
        if self.has_lives:
            start_lives = self.env.ale.lives()
        prev_s = None
        for _ in range(self.actrep):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            # Only for training mode. Resets after Atari game round ends, according to:
            # https://github.com/deepmind/alewrap/blob/master/alewrap/GameEnvironment.lua#L102
            if not test and self.has_lives and self.env.ale.lives() < start_lives:
                term = True
            if term:
                break
            prev_s = s
        # Takes maximum value for each pixel value over the current and previous frame
        # Used to get round Atari sprites flickering (Mnih et al. (2015))
        if prev_s is not None:
            s = np.maximum.reduce([s, prev_s])
        return self.preprocess(s), accum_reward, term, info