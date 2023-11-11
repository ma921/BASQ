import torch
from .._prior import Gaussian
from .._utils import TensorManager
from .._kernel import Kernel
from .._rchq import recombination
from .._gp import predict_mean
import warnings
warnings.filterwarnings("ignore")


class BASQ(TensorManager):
    def __init__(self, n_cand, n_nys, prior, warped_gp=False):
        TensorManager.__init__(self)
        self.n_cand = n_cand
        self.n_nys = n_nys
        self.prior = prior
        self.warped_gp = warped_gp
        
    def batch_uncertainty_sampling(self, model, n_batch):
        if self.warped_gp:
            kernel = model.kernel
        else:
            kernel = Kernel(model)
        
        X_cand = self.prior.sample(self.n_cand)
        X_nys = self.prior.sample(self.n_nys)
        idx_batch, w_batch = recombination(
            X_cand,             # random samples for recombination
            X_nys,              # number of samples used for approximating kernel with Nystrom method
            n_batch,            # number of samples finally returned
            kernel,             # kernel
            self.device,          # device
            self.dtype,           # dtype
        )
        X_batch = X_cand[idx_batch]
        return X_batch, w_batch
    
    def quadrature(self, model, n_batch):
        X_batch, w_batch = self.batch_uncertainty_sampling(model, n_batch)
        if self.warped_gp:
            Y_hat_batch = model.predict_mean(X_batch)
        else:
            Y_hat_batch = predict_mean(X_batch, model)
        expected_integral = w_batch @ Y_hat_batch
        self.marginal_likelihood = expected_integral
        return expected_integral
    
    def full_quadrature(self, model, n_batch):
        if self.warped_gp:
            kernel = model.kernel
        else:
            kernel = Kernel(model)
        
        X_cand = self.prior.sample(self.n_cand)
        X_nys = self.prior.sample(self.n_nys)
        idx_batch, w_batch = recombination(
            X_cand,             # random samples for recombination
            X_nys,              # number of samples used for approximating kernel with Nystrom method
            n_batch,            # number of samples finally returned
            kernel,             # kernel
            self.device,          # device
            self.dtype,           # dtype
        )
        X_batch = X_cand[idx_batch]
        
        # Mean
        if self.warped_gp:
            Y_hat_batch = model.predict_mean(X_batch)
        else:
            Y_hat_batch = predict_mean(X_batch, model)
            
        expected_integral = w_batch @ Y_hat_batch
        self.marginal_likelihood = expected_integral
        
        # Variance
        w_cand = self.ones(self.n_cand) / self.n_cand
        first = w_batch @ kernel(X_batch, X_batch) @ w_batch
        second = 2 * w_batch @ kernel(X_batch, X_cand) @ w_cand
        third = w_cand @ kernel(X_cand, X_cand) @ w_cand
        varinace_integarl = first - second + third
        return expected_integral, varinace_integarl
        
    def posterior(self, x_test, model):
        if not hasattr(self, "marginal_likelihood"):
            self.quadrature(model, 100)
            
        if self.warped_gp:
            Y_hat = model.predict_mean(x_test)
        else:
            Y_hat = predict_mean(x_test, model)
        posterior = self.prior.pdf(x_test) * Y_hat / self.marginal_likelihood
        return posterior
    
    def KLdivergence(self, Z_true, x_test, true_likelihood, model):
        if not hasattr(self, "marginal_likelihood"):
            self.quadrature(model, 100)
        
        KL = torch.zeros(len(x_test)).to(self.device, self.dtype)
        q = torch.squeeze(
            self.prior.pdf(x_test) * true_likelihood(x_test) / Z_true
        )
        if self.warped_gp:
            Y_hat = model.predict_mean(x_test)
        else:
            Y_hat = predict_mean(x_test, model)
        p = torch.squeeze(
            self.prior.pdf(x_test) * Y_hat  / self.marginal_likelihood
        )
        KL[p > 0] = p[p > 0] * torch.log(p[p > 0] / q[p > 0])
        return (KL.sum() * len(x_test) / (p > 0).sum()).abs()