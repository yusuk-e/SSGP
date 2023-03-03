# Symplectic Spectrum Gaussian Processes
This repository contains the code for the paper:
- [Symplectic Spectrum Gaussian Processes: Learning Hamiltonians from Noisy and Sparse Data](https://openreview.net/forum?id=W4ZlZZwsQmt)

In this work, we present a Gaussian process that incorporates the symplectic geometric structure of Hamiltonian systems, which is used as a prior distribution for estimating Hamiltonian systems with additive dissipation. Experiments on several physical systems show that SSGP offers excellent performance in predicting dynamics that follow the energy conservation or dissipation law from noisy and sparse data.

## Requirements
This code is written in Python 3, and depends on PyTorch and torchdiffeq. Only the executable file is written in Ruby 3.

## Quick example
- One can obtain the datasets of several physical systems (Pendulum, Duffing Oscillator, Double Pendulum) by running the code in [data](data):
```
ruby run.rb
```
- The training and prediction are executed by the code in [model](model):
```
ruby run.rb
```

## Citations
```
@inproceedings{tanaka2022,
	title={Symplectic Spectrum Gaussian Processes: Learning Hamiltonians from Noisy and Sparse Data},
	author={Yusuke Tanaka and Tomoharu Iwata and Naonori Ueda},
	booktitle={Advances in Neural Information Processing Systems},
	year={2022}
	}
```