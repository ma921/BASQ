import time
import copy
import torch
import gpytorch
from torch.distributions.multivariate_normal import MultivariateNormal

import gp
from gmm import GMM
from basq import BASQ
from metric import KLdivergence

def set_basq():
    # Bayesian Inference Modelling
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_dim = 3
    mu_pi = torch.zeros(num_dim)
    cov_pi = 2*torch.eye(num_dim)
    true_likelihood = GMM(num_dim, mu_pi, cov_pi, device)

    # BQ modelling
    train_x = torch.rand(2,num_dim)
    train_y = true_likelihood(train_x)
    prior = MultivariateNormal(mu_pi,cov_pi)
    #gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
    gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    model, likelihood = gp.set_gp(train_x, train_y, gp_kernel)

    # evaluation setting
    Z_true = 1
    test_x = prior.sample(sample_shape=torch.Size([10000]))
    metric = KLdivergence(prior, test_x, Z_true, true_likelihood)
    print("True model evidence E[Z|y] is "+str(metric.Z_true))
    return prior, model, likelihood, true_likelihood, metric

if __name__ == "__main__":
    torch.manual_seed(0)
    n_batch = 10      # the number of BASQ iteration
    
    prior, model, likelihood, true_likelihood, metric = set_basq()
    basq = BASQ(
        prior,
        model,
        likelihood,
        true_likelihood,
        n_rec=20000,       # subsampling ratio for Recombination
        nys_ratio=1e-2,    # subsubsampling ratio for Nystrom approximation
        training_iter=100, # number of SDG interations for GP Type-II MLE
        batch_size=100,    # batch size
        quad_ratio=5,      # supersampling ratio for quadrature
    )
    
    results = basq.run(n_batch)
    print(
        "overhead: "+str(results[:,0].sum().item())+" [s]\n"
        + "overhead per sample: "+str(results[:,0].sum().item()/basq.batch_size/n_batch)+" [s]\n"
        + "final E[Z|y]: "+str(results[-1,1].item())+"\n"
        + "final Var[Z|y]: "+str(results[-1,2].item())+"\n"
        + "logMAE: "+str(torch.log(torch.tensor(abs(results[-1][1] - metric.Z_true))).item())+"\n"
        + "logKL: "+str(torch.log(metric(basq.model, basq.likelihood, results[-1][1])).item())+"\n"
    )