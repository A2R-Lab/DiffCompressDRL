# DiffCompressDRL: Differentially Encoded Observation Spaces 
<img width="720" alt="image_diff_example (1)" src="https://github.com/A2R-Lab/DiffCompressDRL/assets/8332062/7cbea8b2-225c-4414-a9f7-1fd6f6685f3c">


Source code and numerical experiments for the paper: "[Differentially Encoded Observation Spaces for Perceptive Reinforcement Learning](https://arxiv.org/pdf/2310.01767.pdf)"

## Method
DiffCompressDRL makes use of differential image encodings in order to compress and thereby drastically reduce the overall memory footprint of perceptive-based deep RL. In brief, we make use of a custom observation compressor that functions as follows:
1. Given a set of `N` input observations of shape `(N, F, H, W, C)`, where `F` is the frame-stack size, we compute and store temporal indices for each observation (this allows us to store one copy of each individual image, instead of duplicates across multiple stacks).
2. We store every `F`th raw image observation `o_r: (H, W, C)`, compressing all other images `o_i` by storing the difference between it and the reference in sparse matrix format `SparseArray(o_i - o_r)` (see diagram above).
3. To retrieve an observation, we either return it immediately (if it is stored in raw format) or uncompress it by adding its refernce image `o_r` back to it (`o_i = SparseArray(o_i - o_r) + o_r`).

We find our compression method is able to reduce the memory footprint by as much as 14.2x and 16.7x across tasks from the Atari 2600 benchmark and the DeepMind Control Suite respectively, while not affecting convergence.

<img width="300" alt="image" src="https://github.com/A2R-Lab/DiffCompressDRL/assets/8332062/212e41b6-941f-4e32-ad26-f2fa523f5131">
<img width="420" alt="image" src="https://github.com/A2R-Lab/DiffCompressDRL/assets/8332062/076eec6b-95df-4552-8699-470143b9d822">



## Quick-Start Guide

Requires Python 3.8+

**This package contains submodules make sure to run ```git submodule update --init --recursive```** after cloning!

### Atari Experiments

```
python3 -m pip install requirements.txt
cd atari_compress/
python3 train.py --config cfgs/experiment.yaml
```
Params are loaded from a supplied YAML file and overwrite `cfgs/default.yaml`. The main arguments are:
* `alg`, the algorithm to use (i.e. `PPO` or `QRDQN`)
* `env`, the Atari environment
* `seeds`, list of random seeds
* `compress`, whether to turn on observation compression

We tested on the following 10 Atari environments:
```
AsteroidsNoFrameskip-v4
BeamRiderNoFrameskip-v4
BreakoutNoFrameskip-v4
EnduroNoFrameskip-v4
MsPacmanNoFrameskip-v4
PongNoFrameskip-v4
QbertNoFrameskip-v4
RoadRunnerNoFrameskip-v4
SeaquestNoFrameskip-v4
SpaceInvadersNoFrameskip-v4
```


### DeepMind Control Suite Experiments

```
cd drqv2_compress/
# Follow install instructions in README.md
python3 train.py task=TASK full_compress=True
```
where you can turn on half or full compression with `--full_compress=True` or `--half_compress=True`.

and where `TASK` is one of:
```
quadruped_walk
walker_walk
```
or any of the many other tasks that we have not explicitly tested in the DMC (see `/drqv2_compress/cfgs/task`).

### Citing
To cite this work in your research, please use the following bibtex:
```
@misc{grossman2023differentially,
      title={Differentially Encoded Observation Spaces for Perceptive Reinforcement Learning}, 
      author={Lev Grossman and Brian Plancher},
      year={2023},
      eprint={2310.01767},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
