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
