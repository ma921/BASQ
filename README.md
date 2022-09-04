# BASQ_torch

The minimal implementation of BASQ with PyTorch and GPyTorch. 

## Prerequisite
- PyTorch
- GPyTorch
- PyYAML

## Features
- GPU acceleration
- Arbitrary kernel for Bayesian Quadrature modelling
- Arbitrary prior distribution for Bayesian inference
- KISS-GP with LOVE for constant-time GP variance inference

## How to run
```python
python3 main.py
```

The example with Gaussian Mixture Likelihood (dim=10) will run.

## For developers
You can select arbitrary kernels/priors/parameters on ./BASQ/_parametetrs.py
- To change kernel: change "kernel_type", such as Matern52.
- To change prior: change the prior distribution directly as input to the BASQ. The prior must be defined by torch.distributions (see main.py)
- As WSABI and uncertainty sampling is defined only for RBF kernel and normal prior, non-Gaussian kernel or prior will change the internal setting automatically for compatibility.
