## Asynchronous Q-Learning
An attempt to implement asynchronous 1-step Q-Learning from Google DeepMind's paper ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.](https://arxiv.org/abs/1602.01783)

**Intuition and detailed implementation description.**
*In progress*.

## Requirements
1. Linux based OS or Mac OS X;
2. [Anaconda package with Python 2.7+ or 3.5+](https://www.continuum.io/downloads);
3. [TensorFlow](https://www.tensorflow.org/);
4. [OpenAI Gym](https://gym.openai.com/). To install, run in terminal:
```
pip install gym['all']
```

## Usage
**To train** your own model on 'Atari 2600 Breakout', simply run:
```
python asynq.py
```

To specify another environment, use `--env_name NAME` flag, e.g:
```
python asynq.py --env_name 'Pong-v0'
```
All available environments you can check [here](https://gym.openai.com/envs). Note that current implementation supports environments only with raw pixels observations.
Tested OpenAI Gym environments:
* Breakout-v0
* Pong-v0

**To play without training** and logging, pass `--play` flag:
```
python asynq.py --logdir path/to/checkpoint/folder/ --threads 2 --play
```

To change amount of spawned threads, pass `--threads NUMBER` (by default = 8); to use GPU instead of cpu, pass `--gpu` flag.
List of all available flags you can check with `python asynq.py --help`

## Pretrained models
**To use pretrained model**, or change log folder, just pass `--logdir PATH` flag:
```
python asynq.py --logdir path/to/checkpoint/folder/
```

*Pretrained models are in progress*.