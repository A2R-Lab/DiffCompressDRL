# DiffCompressDRL
<img width="720" alt="image_diff_example (1)" src="https://github.com/A2R-Lab/DiffCompressDRL/assets/8332062/7cbea8b2-225c-4414-a9f7-1fd6f6685f3c">


Source code and numerical experiments for the paper: "[Differentially Encoded Observation Spaces for Perceptive Reinforcement Learning](https://arxiv.org/pdf/2310.01767.pdf)"

**This package contains submodules make sure to run ```git submodule update --init --recursive```** after cloning!

## Installation



## Atari Experiments
```
cd atari_compress
python3 train.py --env ENV --alg ALG --compress
```
where `--compress` turns on compression and leaving it off will run the standard algorithm

and where `ALG` is one of:
```
PPO
QRDQN
```
and where `ENV` is one of:
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

## DeepMind Control Suite Tasks
```
cd drqv2_compress
python3 train.py --task=TASK --full_compress=True
```
where you can turn on half or full compression with `--full_compress=True` or `--half_compress=True`.

and where `TASK` is one of:
```
quadruped_walk
walker_walk
```
or any of the many other tasks that we have not tested in the DMC (see `/drqv2_compess/cfpgs/task`)

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
