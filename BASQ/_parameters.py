import torch
import gpytorch
from ._quadrature import KernelQuadrature
from ._sampler import PriorSampler, UncertaintySampler
from ._wsabi import WsabiGP
from ._vbq import VanillaGP


class Parameters:
    def __init__(self, Xobs, Yobs, prior, true_likelihood, device):
        """
        Args:
           - Xobs; torch.tensor, X samples, X belongs to prior measure.
           - Yobs; torch.tensor, Y observations, Y = true_likelihood(X).
           - prior; torch.distributions, prior distribution.
           - true_likelihood; function of y = function(x), true likelihood to be estimated.
           - device; torch.device, device, cpu or cuda
        """
        # BQ Modelling
        bq_model = "vbq"               # select a BQ model from ["wsabi", "vbq"], vbq stands for Vanilla BQ
        sampler_type = "prior"         # select a sampler from ["uncertainty", "prior"]
        kernel_type = "RBF"            # select a kernel from ["RBF", "Matern32", "Matern52"]

        # WSABI modelling
        wsabi_type = "wsabim"          # select a wsabi type from ["wsabil", "wsabim"]
        alpha_factor = 1               # coefficient of alpha in WSABI modelling; alpha = 0.8 * min(y)

        # GP hyperparameter training with type-II MLE
        optimiser = "BoTorch"          # select a optimiser from ["L-BFGS-B", "BoTorch", "Adam"], BoTorch is the slowest but the most accurate
        lik = 1e-10                    # centre value of GP likelihood noise.
        rng = 10                       # range of likelihood noise [lik/rng, lik*rng]
        train_lik = False              # flag whether or not to train likelihood noise. if False, the noise is fixed with lik
        training_iter = 1000           # maximum number of SDG interations
        thresh = 0.05                  # stopping criterion. threshold = last_MLL - current_MLL
        lr = 0.1                       # learning rate of Adam

        # RCHQ hyperparameters
        n_rec = 20000                  # subsampling size for kernel recombination
        nys_ratio = 1e-2               # subsubsampling ratio for Nystrom. Number of Nystrom samples is nys_ratio * n_rec
        batch_size = 100               # batch size
        quad_ratio = 5                 # supersampling ratio for quadrature. Number of recombination samples is quad_ratio * n_rec

        # Uncertainty sampling
        ratio = 0.5                    # mixing ratio of prior and uncertainty sampling, 0 < r < 1
        n_gaussians = 50               # number of Gaussians approximating the GP-modelled acquisition function
        threshold = 1e-5               # threshold to cut off the insignificant Gaussians
        sampling_method = "exact"      # select sampling method from ["exact", "approx"]

        # Utility
        show_progress = True           # flag whether or not show the quadrature result over each iteration.

        # loading
        self.n_rec = n_rec
        self.batch_size = batch_size
        self.device = device
        self.show_progress = show_progress

        self.prior = prior
        self.true_likelihood = true_likelihood

        self.bq_model = bq_model
        self.sampler_type = sampler_type
        self.check_compatibility(bq_model, sampler_type, kernel_type)
        gp_kernel = self.set_kernel(kernel_type)
        self.set_model(bq_model, Xobs, Yobs, gp_kernel, wsabi_type, alpha_factor, lik, training_iter, thresh, lr, rng, train_lik, optimiser)
        self.set_sampler(sampler_type, sampling_method, prior, n_rec, nys_ratio, ratio, n_gaussians, threshold)
        self.set_quadrature(nys_ratio, int(n_rec * nys_ratio), int(quad_ratio * n_rec))
        self.verbose(bq_model, sampler_type, kernel_type, sampling_method, optimiser)

    def verbose(self, bq_model, sampler_type, kernel_type, sampling_method, optimiser):
        print(
            "BQ model: " + bq_model
            + " | kernel: " + kernel_type
            + " | sampler: " + sampler_type
            + " | sampling_method: " + sampling_method
            + " | optimiser: " + optimiser
        )

    def set_sampler(self, sampler_type, sampling_method, prior, n_rec, nys_ratio, ratio, n_gaussians, threshold):
        """
        Args:
           - sampler_type; string, ["uncertainty", "prior"]
           - sampling_method; string, ["exact", "approx"]
           - prior: torch.distributions, prior distribution.
           - n_rec: int, subsampling size for kernel recombination
           - nys_ratio: float, subsubsampling ratio for Nystrom. Number of Nystrom samples is nys_ratio * n_rec
           - ratio: float, mixing ratio of prior and uncertainty sampling, 0 < r < 1
           - n_gaussians: int, number of Gaussians approximating the GP-modelled acquisition function
           - threshold: float, threshold to cut off the insignificant Gaussians
        """
        self.prior_sampler = PriorSampler(prior, n_rec, nys_ratio, self.device)

        if sampler_type == "uncertainty":
            self.sampler = UncertaintySampler(
                prior,
                self.gp.model,
                n_rec,
                nys_ratio,
                self.device,
                sampling_method=sampling_method,
                ratio=ratio,
                n_gaussians=n_gaussians,
                threshold=threshold,
            )
            self.kernel_quadrature = self.gp.predictive_kernel
        elif sampler_type == "prior":
            self.sampler = PriorSampler(prior, n_rec, nys_ratio, self.device)
            self.kernel_quadrature = self.kernel
        else:
            raise Exception("The given sampler_type is undefined.")

    def set_model(self, bq_model, Xobs, Yobs, gp_kernel, wsabi_type, alpha_factor, lik, training_iter, thresh, lr, rng, train_lik, optimiser):
        """
        Args:
           - bq_model: string, ["wsabi", "vbq"]
           - Xobs: torch.tensor, X samples, X belongs to prior measure.
           - Yobs: torch.tensor, Y observations, Y = true_likelihood(X).
           - gp_kernel: gpytorch.kernels, GP kernel function
           - wsabi_type: string, ["wsabil", "wsabim"]
           - alpha_factor: float, coefficient of alpha in WSABI modelling; alpha = 0.8 * min(y)
           - lik: float, the initial value of GP likelihood noise variance
           - train_iter: int, the maximum iteration for GP hyperparameter training.
           - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
           - lr: float, the learning rate of Adam optimiser
           - rng: int, tne range coefficient of GP likelihood noise variance
           - train_like: bool, flag whether or not to update GP likelihood noise variance
           - optimiser: string, select the optimiser ["L-BFGS-B", "BoTorch", "Adam"]
        """
        if bq_model == "wsabi":
            self.gp = WsabiGP(
                Xobs,
                Yobs,
                gp_kernel,
                self.device,
                label=wsabi_type,
                alpha_factor=alpha_factor,
                lik=lik,
                training_iter=training_iter,
                thresh=thresh,
                lr=lr,
                rng=rng,
                train_lik=train_lik,
                optimiser=optimiser,
            )
            self.kernel = self.gp.kernel
            self.update = self.gp.update_wsabi_gp
            self.unimodal_approx = self.gp.unimodal_approximation
            self.uniform_trans = self.gp.uniform_transformation

        elif bq_model == "vbq":
            self.gp = VanillaGP(
                Xobs,
                Yobs,
                gp_kernel,
                self.device,
                lik=lik,
                training_iter=training_iter,
                thresh=thresh,
                lr=lr,
                rng=rng,
                train_lik=train_lik,
                optimiser=optimiser,
            )
            self.kernel = self.gp.predictive_kernel
            self.update = self.gp.update_gp
            self.retrain = self.gp.retrain_gp
        else:
            raise Exception("The given bq_model is undefined.")

        self.predict_mean = self.gp.predict_mean
        self.predict = self.gp.predict
        self.retrain = self.gp.retrain_gp

    def set_quadrature(self, nys_ratio, n_nys, n_quad):
        """
        Args:
           - nys_ratio: float, subsubsampling ratio for Nystrom.
           - n_nys: int, number of Nystrom samples; int(nys_ratio * n_rec)
           - n_quad: int, number of kernel recombination subsamples; int(quad_ratio * n_rec)
        """
        self.kq = KernelQuadrature(
            self.n_rec,
            n_nys,
            n_quad,
            self.batch_size,
            self.prior_sampler,
            self.kernel_quadrature,
            self.device,
            self.predict_mean,
        )

    def set_kernel(self, kernel_type):
        """
        Args:
           - kernel_type: string, ["RBF", "Matern32", "Matern52"]

        Returns:
           - gp_kernel: gpytorch.kernels, function of GP kernel
        """
        if kernel_type == "RBF":
            gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == "Matern32":
            gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_type == "Matern52":
            gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        else:
            raise Exception("The given kernel_type is undefined.")
        return gp_kernel

    def check_compatibility(self, bq_model, sampler_type, kernel_type):
        """
        Args:
           - bq_model: string, ["wsabi", "vbq"]
           - sampler_type; string, ["uncertainty", "prior"]
           - kernel_type: string, ["RBF", "Matern32", "Matern52"]
        """
        if bq_model == "wsabi":
            if not kernel_type == "RBF":
                raise AssertionError("WSABI model requires RBF kernel.")
        else:
            if not sampler_type == "prior":
                raise AssertionError("Uncertainty sampling requires WSABI-L modelling with RBF kernel.")

        if sampler_type == "uncertainty":
            if not bq_model == "wsabi" and kernel_type == "RBF":
                raise AssertionError("Uncertainty sampling requires WSABI-L modelling with RBF kernel.")

        if not type(self.prior) == torch.distributions.multivariate_normal.MultivariateNormal:
            if not bq_model == "vbq" and sampler_type == "prior":
                raise AssertionError("Non-Gaussian prior requires prior sampling with VBQ modelling.")

        if not kernel_type == "RBF":
            if not bq_model == "vbq" and sampler_type == "prior":
                raise AssertionError("Non-RBF kernel requires prior sampling with VBQ modelling.")
