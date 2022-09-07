import copy
import torch
from ._utils import Utils
from torch.distributions.uniform import Uniform


class GaussianCalc:
    def __init__(self, prior, device):
        """
        Args:
            - prior; torch.distributions, prior distribution.
            - device; torch.device, device, cpu or cuda
        """
        self.prior = prior
        self.device = device
        self.utils = Utils(device)

    def get_cache(self, model):
        """
        Observed dataset = (Xobs, Yobs)
        woodbury_vector = K(Xobs, Xobs)^(-1) @ Yobs
        woodbury_inv = K(Xobs, Xobs)^(-1)
        S @ S.T = woodbury_inv

        Args:
            - model: gpytorch.models, function of GP model, typically self.wsabi.model in _basq.py

        Returns:
            - woodbury_vector: torch.tensor, Woodbury vector, K(Xobs, Xobs)^(-1) @ Yobs
            - woodbury_inv: torch.tensor, the inverse of Gram matrix K(Xobs, Xobs)^(-1)
        """
        try:
            woodbury_vector = model.prediction_strategy.mean_cache
            S = model.prediction_strategy.covar_cache
        except AttributeError:
            model.eval()
            mean = self.prior.loc.view(-1).unsqueeze(0)
            model(mean)
            woodbury_vector = model.prediction_strategy.mean_cache
            S = model.prediction_strategy.covar_cache
        woodbury_inv = S @ S.T
        return woodbury_vector, woodbury_inv

    def parameters_extraction(self, model):
        self.Xobs = copy.deepcopy(model.train_inputs[0])
        self.n_data, self.n_dims = self.Xobs.size()
        self.woodbury_vector, self.woodbury_inv = self.get_cache(model)
        self.outputscale = copy.deepcopy(model.covar_module.outputscale.detach())
        self.lengthscale = copy.deepcopy(model.covar_module.base_kernel.lengthscale.detach())
        self.W = torch.eye(self.Xobs.size(1)).to(self.device) * self.lengthscale ** 2
        self.v = self.outputscale * torch.sqrt(torch.linalg.det(2 * torch.pi * self.W))

    def unimodal_approximation(self, model, alpha):
        """
        approximate the GP-modelled likelihood by a unimodal multivariate normal distribution.
        https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures

        Args:
            - model: gpytorch.models, function of GP model, typically self.wsabi.model in _basq.py
            - alpha: torch.tensor, the alpha hyperparameter in WSABI-BQ modelling, ell = alpha + 0.5 ell^2

        Returns:
            - mvn_pi_max: torch.distributions, mutlivariate normal distribution of optimised prior
        """
        self.parameters_extraction(model)

        x = (self.Xobs.unsqueeze(1) - self.Xobs.unsqueeze(0)).reshape(self.n_data**2, self.n_dims)
        Npdfs = self.utils.safe_mvn_prob(
            torch.zeros(self.n_dims).to(self.device),
            2 * self.W,
            x,
        ).reshape(self.n_data, self.n_data)

        _w_m = 0.5 * (self.v**2) * (self.woodbury_vector.unsqueeze(1) * self.woodbury_vector.unsqueeze(0)) * Npdfs
        w_m = _w_m / _w_m.sum()

        mu_pi_max = alpha + (w_m.unsqueeze(2) * (self.Xobs.unsqueeze(1) + self.Xobs.unsqueeze(0)) / 2).sum(axis=0).sum(axis=0)
        Xij2 = ((self.Xobs.unsqueeze(1) + self.Xobs.unsqueeze(0)) / 2).reshape(self.n_data ** 2, self.n_dims) - mu_pi_max
        W_m = w_m.reshape(self.n_data**2, 1)
        cov_pi_max = (W_m.unsqueeze(1) * Xij2.unsqueeze(2) @ Xij2.unsqueeze(1)).sum(axis=0) + self.W / 2
        mvn_pi_max = self.utils.safe_mvn_register(mu_pi_max, cov_pi_max)
        return mvn_pi_max

    def uniform_transformation(self, model, Y_unwarp):
        """
        Args:
            - model: gpytorch.models, function of GP model, typically self.wsabi.model in _basq.py
            - Y_unwarp: torch.tensor, the raw observations without WSABI warping.

        Returns:
            - Xobs_uni: torch.tensor, the inputs transformed into uniform prior distribution
            - Yobs_uni: torch.tensor, the observation transformed into uniform prior distribution
            - uni_sampler: function of samples = function(n_samples), a uniform distribution sampler
            - uni_logpdf: function of logpdf = function(x), a log probability density function of uniform distribution
        """
        self.parameters_extraction(model)
        uni_min = self.Xobs.min(0)[0]
        uni_max = self.Xobs.max(0)[0]

        uni_sampler = lambda N: torch.stack([
            Uniform(uni_min[i], uni_max[i]).sample(torch.Size([N])) for i in range(self.n_dims)
        ]).T
        uni_logpdf = lambda X: torch.ones(X.size(0)).to(self.device) * torch.sum(-torch.log(uni_max - uni_min))

        Yobs_uni = self.utils.remove_anomalies(Y_unwarp)
        idx = self.utils.remove_anomalies_uniform(self.Xobs, uni_min, uni_max)
        Xobs_uni = self.Xobs[idx]
        Yobs_uni = Yobs_uni[idx]
        return Xobs_uni, Yobs_uni, uni_sampler, uni_logpdf
