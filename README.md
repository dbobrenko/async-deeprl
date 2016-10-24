## Asynchronous Q-Learning
An attempt to implement asynchronous 1-step Q-Learning from Google DeepMind's paper ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.](https://arxiv.org/abs/1602.01783)

## Requirements
1. Linux based OS or Mac OS X;
2. [Anaconda package with Python 2.7+ or 3.5+](https://www.continuum.io/downloads);
3. [TensorFlow](https://www.tensorflow.org/);
4. [OpenAI Gym](https://gym.openai.com/). To install, run in terminal:
```
pip install gym['all']
```

## Usage
To train your own model on 'Atari 2600 Breakout', simply run:
```
python asynq.py
```

To specify your own environment, use `[--env_name NAME]` flag, e.g:
```
python asynq.py --env_name 'Pong-v0'
```
All available environments you can check [here](https://gym.openai.com/envs). Note that current implementation supports environments only with visual raw pixel observations.
Tested OpenAI Gym environments:
* Breakout-v0
* Pong-v0

**To use pretrained model**, or change log folder, just pass`[--logdir PATH]` flag:
```
python asynq.py --logdir path/to/checkpoint/folder/
```

**To play without training** and logging, pass `[--play]` flag:
```
python asynq.py --logdir path/to/checkpoint/folder/ --threads 2 --play
```

Also, you can change amount of spawned threads with `[--threads NUMBER]` (by default = 8), use GPU instead of cpu with `[--gpu]` flag, etc.
Whole list with available flags you can check with `[--help]` flag.

## Pretrained models
*In progress*.

## Detailed implementation description and theory
*In progress*.