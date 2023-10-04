# DiffCompressDRL
Source code and numerical experiments for the paper: "Differentially Encoded Observation Spaces for Perceptive Reinforcement Learning"

## Atari Experiments
```
python3 train_atari.py --env BreakoutNoFrameskip-v4 --alg QRDQN --compress
```
## DeepMind Control Suite Tasks
```
git clone git@github.com:A2R-Lab/drqv2_compress.git
cd drqv2_compress
python3 train.py --task=quadruped_walk --full_compress=True
```
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
