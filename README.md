# BASQ_torch

The minimal implementation of BASQ with PyTorch and GPyTorch. 

## Prerequisite
- PyTorch
- GPyTorch

## Features
- GPU acceleration
- Arbitrary kernel for Bayesian Quadrature modelling
- Arbitrary prior distribution for Bayesian inference
- KISS-GP with LOVE for constant-time GP variance inference

## How to run
```python
python3 main.py
```

The example with Gaussian Mixture Likelihood (dim=3) will run.

## Comments
This main branch is only for the CPU environment.
GPU version is in the branch "bascuda".
Pytorch device assignment causes additional overhead, which is why the versions are separated.
While CPU outperforms GPU in small settings, whereas GPU performs better in large settings (e.g. larger batch iteration, batch size)
