import torch
import gpytorch
from ._quadrature import KernelQuadrature
from ._sampler import PriorSampler, UncertaintySampler
from ._wsabi import WsabiGP
from ._vbq import VanillaGP
    
class Parameters:
    def __init__(self, Xobs, Yobs, prior, true_likelihood):
        ##### defining ######
        # BQ Modelling
        bq_model="wsabi"             # select a BQ model from ["wsabi", "vbq"]
        sampler_type="uncertainty"   # select a sampler from ["uncertainty", "prior"]
        kernel_type="RBF"            # select a kernel from ["RBF", "Matern32", "Matern52"]

        # WSABI modelling
        wsabi_type="wsabil"          # select a wsabi type from ["wsabil", "wsabim", "vbq"]
        alpha_factor=0.8             # coefficient of alpha in WSABI modelling; alpha = 0.8 * min(y)

        # GP hyperparameter training with type-II MLE
        lik=1e-10                    # centre value of GP likelihood noise.
        rng=10                       # range of likelihood noise [lik/rng, lik*rng]
        train_lik=False              # flag whether or not to train likelihood noise. if False, the noise is fixed with lik
        training_iter=1000           # maximum number of SDG interations
        thresh=0.01                  # stopping criterion. threshold = last_MLL - current_MLL
        lr=0.1                       # learning rate of Adam

        # RCHQ hyperparameters
        n_rec=20000                  # subsampling size for Recombination
        nys_ratio=1e-2               # subsubsampling ratio for Nystrom. Number of Nystrom samples is nys_ratio * n_rec
        batch_size=100               # batch size
        quad_ratio=5                 # supersampling ratio for quadrature. Number of recombination samples is quad_ratio * n_rec

        # Uncertainty sampling
        ratio=0.5                    # mixing ratio of prior and uncertainty sampling
        n_gaussians=100              # Number of Gaussians approximating the GP-modelled acquisition function 
        threshold=1e-5               # Threshold to cut off the insignificant Gaussians

        # Utility
        show_progress=True           # flag whether or not show the quadrature result over each iteration.

        ##### loading #####
        self.n_rec = n_rec
        self.batch_size = batch_size
        self.show_progress = show_progress

        self.prior = prior
        self.true_likelihood = true_likelihood

        self.bq_model = bq_model
        self.sampler_type = sampler_type
        bq_model, sampler_type, kernel_type = self.check_compatibility(bq_model, sampler_type, kernel_type)
        gp_kernel = self.set_kernel(kernel_type)
        self.set_model(bq_model, Xobs, Yobs, gp_kernel, wsabi_type, alpha_factor, lik, training_iter, thresh, lr, rng, train_lik)
        self.set_sampler(sampler_type, prior, n_rec, nys_ratio, ratio, n_gaussians, threshold)
        self.set_quadrature(nys_ratio, int(n_rec*nys_ratio), int(quad_ratio*n_rec))
        
    def set_sampler(self, sampler_type, prior, n_rec, nys_ratio, ratio, n_gaussians, threshold):
        self.prior_sampler = PriorSampler(prior, n_rec, nys_ratio)

        if sampler_type == "uncertainty":
            self.sampler = UncertaintySampler(
                prior,
                self.wsabi.model,
                n_rec,
                nys_ratio,
                ratio=ratio,
                n_gaussians=n_gaussians,
                threshold=threshold,
            )
        elif sampler_type == "prior":
            self.sampler = PriorSampler(prior, n_rec, nys_ratio)
        else:
            raise Exception("The given sampler_type is undefined.")

    def set_model(self, bq_model, Xobs, Yobs, gp_kernel, wsabi_type, alpha_factor, lik, training_iter, thresh, lr, rng, train_lik):
        if bq_model=="wsabi":
            self.wsabi = WsabiGP(
                Xobs,
                Yobs,
                gp_kernel,
                label=wsabi_type,
                alpha_factor=alpha_factor,
                lik=lik,
                training_iter=training_iter,
                thresh=thresh,
                lr=lr,
                rng=rng,
                train_lik=train_lik,
            )
            self.kernel = self.wsabi.kernel
            self.predict_mean = self.wsabi.predict_mean
            self.predict = self.wsabi.predict
            self.update = self.wsabi.update_wsabi_gp
            self.unimodal_approx = self.wsabi.unimodal_approximation
            self.uniform_trans = self.wsabi.uniform_transformation
            self.retrain = self.wsabi.retrain_gp
        elif bq_model=="vbq":
            self.vbq = VanillaGP(
                Xobs, 
                Yobs,
                gp_kernel,
                lik=lik,
                training_iter=training_iter,
                thresh=thresh,
                lr=lr,
                rng=rng,
                train_lik=train_lik,
            )
            self.kernel = self.vbq.model.covar_module.forward
            self.predict_mean = self.vbq.predict_mean
            self.predict = self.vbq.predict
            self.update = self.vbq.update_gp
            self.retrain = self.vbq.retrain_gp
        else:
            raise Exception("The given bq_model is undefined.")

    def set_quadrature(self, nys_ratio, n_nys, n_quad):
        self.kq = KernelQuadrature(
            self.n_rec, 
            n_nys,
            n_quad,
            self.batch_size, 
            self.prior_sampler, 
            self.kernel, 
            self.predict_mean,
        )

    def set_kernel(self, kernel_type):
        if kernel_type == "Matern32":
            gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_type == "Matern52":
            gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        elif kernel_type == "RBF":
            gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            raise Exception("The given kernel_type is undefined.")
        return gp_kernel

    def check_compatibility(self, bq_model, sampler_type, kernel_type):
        if bq_model == "wsabi":
            kernel_type = "RBF"
        else:
            sampler_type == "prior"
            
        if sampler_type == "uncertainty":
            bq_model = "wsabi"
            kernel_type = "RBF"
            
        if not type(self.prior) == torch.distributions.multivariate_normal.MultivariateNormal:
            bq_model = "vbq"
            sampler_type = "prior"
            
        if not kernel_type == "RBF":
            bq_model = "vbq"
            sampler_type = "prior"
        print("BQ model: "+bq_model+" | kernel: "+kernel_type+" | sampler: "+sampler_type)
        
        return bq_model, sampler_type, kernel_type