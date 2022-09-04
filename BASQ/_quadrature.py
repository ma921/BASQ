import torch
from ._gp import predict
from ._rchq import recombination


class KernelQuadrature:
    def __init__(self, n_rec, n_nys, n_quad, batch_size, sampler, kernel, device, mean_predict):
        """
        Input:
           - n_rec: int, subsampling size for kernel recombination
           - nys_ratio: float, subsubsampling ratio for Nystrom.
           - n_nys: int, number of Nystrom samples; int(nys_ratio * n_rec)
           - n_quad: int, number of kernel recombination subsamples; int(quad_ratio * n_rec)
           - batch_size: int, batch size
           - sampler: function of samples = function(n_samples)
           - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)
           - device: torch.device, cpu or cuda
           - mean_predict: function of mean = function(x), the function that returns the predictive mean at given x
        """
        self.n_rec = n_rec
        self.n_nys = n_nys
        self.n_quad = n_quad
        self.batch_size = batch_size
        self.sampler = sampler
        self.kernel = kernel
        self.device = device
        self.mean_predict = mean_predict

    def rchq(self, pts_nys, pts_rec, w_IS, batch_size, kernel):
        """
        Input:
            - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
            - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
            - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
            - batch_size: int, batch size
            - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)

        Output:
            - x: torch.tensor, the sparcified samples from pts_rec. The number of samples are determined by self.batch_size
            - w: torch.tensor, the positive weights for kernel quadrature as discretised summation.
        """
        idx, w = recombination(
            pts_rec,
            pts_nys,
            batch_size,
            kernel,
            self.device,
            init_weights=w_IS,
        )
        x = pts_rec[idx]
        return x, w

    def quadrature(self):
        """
        Output:
            - EZy: float, the mean of the evidence
            - VarZy: float, the variance of the evidence
        """
        pts_nys, pts_rec, w_IS = self.sampler(self.n_quad)
        X, w = self.rchq(pts_nys, pts_rec, w_IS, self.batch_size, self.kernel)
        EZy = (w @ self.mean_predict(X)).item()
        VarZy = (w @ self.kernel(X, X) @ w).item()
        print("E[Z|y]: " + str(EZy) + "  Var[Z|y]: " + str(VarZy))
        return EZy, VarZy

    def prior_max(self, mvn_max):
        """
        Input:
            - mvn_max: torch.distributions, mutlivariate normal distribution of optimised prior distribution

        Output:
            - EZy_prior: float, the mean of the evidence when the prior is optimised to maximise the evidence
            - VarZy_prior: float, the variance of the evidence when the prior is optimised to maximise the evidence
        """
        pts_rec = mvn_max.sample(sample_shape=torch.Size([self.n_quad]))
        pts_nys = pts_rec[:self.n_nys]
        w_IS = torch.ones(self.n_quad) / self.n_quad

        X, w = self.rchq(pts_nys, pts_rec, w_IS, self.batch_size, self.kernel)
        EZy = (w @ self.mean_predict(X)).item()
        VarZy = (w @ self.kernel(X, X) @ w).item()
        print("prior maximisation")
        print("E[Z|y]: " + str(EZy) + "  Var[Z|y]: " + str(VarZy))
        return EZy, VarZy

    def uniform_trans(self, model_IS, uni_sampler):
        """
        Input:
            - model_IS: gpytorch.models, function of GP model that assumes that
                        prior is uniform distribution transformed via importance sampling
            - uni_sampler: function of samples = function(n_samples), uniform distribution sampler

        Output:
            - EZy_uni float, the mean of the evidence when the prior is transformed into uniform distribution
            - VarZy_uni: float, the variance of the evidence when the prior is transformed into uniform distribution
        """
        pts_rec = uni_sampler(self.n_quad)
        pts_nys = pts_rec[:self.n_nys]
        w_IS = torch.ones(self.n_quad) / self.n_quad

        X, w = self.rchq(pts_nys, pts_rec, w_IS, self.batch_size, model_IS.covar_module.forward)
        mean, _ = predict(X, model_IS)
        EZy = (w @ mean).item()
        VarZy = (w @ model_IS.covar_module.forward(X, X) @ w).item()
        print("uniform transformation")
        print("E[Z|y]: " + str(EZy) + "  Var[Z|y]: " + str(VarZy))
        return EZy, VarZy
