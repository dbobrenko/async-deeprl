## Asynchronous Q-Learning
An attempt to implement asynchronous one-step Q-Learning from Google DeepMind's paper ["Asynchronous Methods for Deep Reinforcement Learning", Mnih et al., 2016.](https://arxiv.org/abs/1602.01783)

**Intuition and detailed implementation description.**
*In progress*.

Benchmarks for current implementation of Asynchronous one-step Q-Learning (GPU: **GTX 980 Ti**; CPU: **Core i7-3770 @ 3.40GHz (4 cores, 8 threads)**):
| **Input**      | **Device** | **FPS** |
|:--------------:|:----------:|:-------:|
| 84x84x4 screen | GPU        | 540     |
| 84x84x4 screen | CPU        | 315     |


## Requirements
1. Linux based OS or Mac OS X;
2. [Anaconda package](https://www.continuum.io/downloads) (recommended);
OR manually install python, and run in terminal:
```
pip install six
pip install future
pip install scipy
```
3. [TensorFlow](https://www.tensorflow.org/);
4. [OpenAI Gym](https://gym.openai.com/). To install, run in terminal:
```
pip install gym['all']
```
Keras and NumPy already comes with listed packages.

## Usage
**To train** your own model on 'Atari 2600 Breakout', simply run:
```
python async_dqn.py
```

To specify another environment, use `--env` flag, e.g:
```
python async_dqn.py --env Pong-v0
```
All available environments you can check [here](https://gym.openai.com/envs). Note, that current implementation supports environments only with raw pixels observations.
Tested OpenAI Gym environments:
* Breakout-v0
* Pong-v0
* SpaceInvarers-v0

To change amount of spawned threads, use `--threads` (by default = 8) flag; to use GPU instead of cpu, pass `--gpu` flag.
All available flags can be checked with `python async_dqn.py --help`

## Pretrained models
**To use pretrained agent**, or change log folder, just use `--logdir` flag:
```
python async_dqn.py --logdir path/to/checkpoint/folder/
```
*TODO: Link to pretrained models*.

## Agent evaluation
To evaluate pretrained agent, use:
```
python async_dqn.py --eval --eval_dir folder/for/evaluation/write --logdir path/to/checkpoint/folder/
```
