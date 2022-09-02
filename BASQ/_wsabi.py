import time
import copy
import torch
from ._gp import update_gp, predict
from ._utils import Utils
from ._rchq import recombination
from ._gaussian_calc import GaussianCalc
from torch.distributions.multivariate_normal import MultivariateNormal

class WsabiGP:
    def __init__(
        self, 
        Xobs, 
        Yobs,
        gp_kernel,
        device,
        label="wsabil",
        alpha_factor=0.8, 
        lik=1e-10, 
        training_iter=10000,
        thresh=0.01,
        lr=0.1,
        rng=10,
        train_lik=False,
    ):
        self.gp_kernel = gp_kernel
        self.device = device
        self.alpha_factor = alpha_factor
        self.lik = lik
        self.training_iter = training_iter
        self.thresh = thresh
        self.lr = lr
        self.rng = rng
        self.train_lik=train_lik
        
        self.jitter = 1e-6
        self.Y_unwarp = copy.deepcopy(Yobs)
        self.utils = Utils(device)
        
        self.model = update_gp(
            Xobs,
            self.process_y_warping(Yobs),
            gp_kernel,
            self.device,
            lik=self.lik, 
            training_iter=self.training_iter, 
            thresh=self.thresh, 
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )
        self.setting(label)
        self.gauss = GaussianCalc(self.model, self.device)
        
    def setting(self, label):
        if label == "wsabil":
            self.kernel = self.wsabil_kernel
            self.predict = self.wsabil_predict
            self.predict_mean = self.wsabil_mean_predict
        elif label == "wsabim":
            self.kernel = self.wsabim_kernel
            self.predict = self.wsabim_predict
            self.predict_mean = self.wsabim_mean_predict
        
    def warp_y(self, y):
        return torch.sqrt(2*(y - self.alpha))
    
    def unwarp_y(self, y):
        return self.alpha + 0.5*y**2
    
    def process_y_warping(self, y):
        y = self.utils.remove_anomalies(y)
        self.alpha = self.alpha_factor * torch.min(y)
        y = self.warp_y(y)
        return y
    
    def cat_observations(self, X, Y):
        Xobs = self.model.train_inputs[0]
        Yobs = copy.deepcopy(self.Y_unwarp)
        if len(self.model.train_targets.shape) == 0:
            Yobs = Yobs.unsqueeze(0)
        Xall = torch.cat([Xobs, X])
        _Yall = torch.cat([Yobs, Y])
        self.Y_unwarp = copy.deepcopy(_Yall)
        Yall = self.process_y_warping(_Yall)
        return Xall, Yall
    
    def update_wsabi_gp(self, X, Y):
        X_warp, Y_warp = self.cat_observations(X, Y)
        self.model = update_gp(
            X_warp,
            Y_warp,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter, 
            thresh=self.thresh, 
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )
        
    def retrain_gp(self):
        X_warp = self.model.train_inputs[0]
        Y_warp = self.process_y_warping(copy.deepcopy(self.Y_unwarp))
        self.model = update_gp(
            X_warp,
            Y_warp,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter, 
            thresh=self.thresh, 
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )
    
    def wsabil_kernel(self, x, y):
        mu_x, _ = predict(x, self.model)
        mu_y, _ = predict(y, self.model)
        cov_xy = self.model.covar_module.forward(x, y)
        CLy = mu_x.unsqueeze(1) * cov_xy * mu_y.unsqueeze(0)

        d = min(len(x), len(y))
        CLy[range(d), range(d)] = CLy[range(d), range(d)] + self.jitter
        return CLy

    def wsabim_kernel(self, x, y):
        mu_x, _ = predict(x, self.model)
        mu_y, _ = predict(y, self.model)
        cov_xy = self.model.covar_module.forward(x, y)
        CLy = mu_x.unsqueeze(1) * cov_xy * mu_y.unsqueeze(0) + 0.5*cov_xy**2

        d = min(len(x), len(y))
        CLy[range(d), range(d)] = CLy[range(d), range(d)] + self.jitter
        return CLy

    def wsabil_predict(self, x):
        mu_warp, var_warp = predict(x, self.model)
        mu = self.alpha + 0.5 * mu_warp**2
        var = var_warp * mu_warp * var_warp
        return mu, var

    def wsabim_predict(self, x):
        mu_warp, var_warp = predict(x, self.model)
        mu = self.alpha + 0.5 * (mu_warp**2 + var_warp)
        var = var_warp * mu_warp * var_warp + 0.5*var_warp**2
        return mu, var
    
    def wsabil_mean_predict(self, x):
        mu_warp, _ = predict(x, self.model)
        mu = self.alpha + 0.5 * mu_warp**2
        return mu

    def wsabim_mean_predict(self, x):
        mu_warp, var_warp = predict(x, self.model)
        mu = self.alpha + 0.5 * (mu_warp**2 + var_warp)
        return mu
    
    def unimodal_approximation(self):
        return self.gauss.unimodal_approximation(self.model, self.alpha)
    
    def uniform_transformation(self, prior):
        Xobs_uni, Yobs_uni, uni_sampler, uni_logpdf = self.gauss.uniform_transformation(
            self.model, 
            self.Y_unwarp,
        )
    
        Y_IS = torch.exp(torch.log(Yobs_uni) + prior.log_prob(Xobs_uni) - uni_logpdf(Xobs_uni))
        Y_IS = self.utils.remove_anomalies(Y_IS)
        model_IS = update_gp(
            Xobs_uni, 
            Y_IS.detach(), 
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter, 
            thresh=self.thresh, 
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )
        return model_IS, uni_sampler