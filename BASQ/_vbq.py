import copy
import torch
from ._gp import update_gp, predict
from ._utils import Utils

class VanillaGP:
    def __init__(
        self, 
        Xobs, 
        Yobs,
        gp_kernel,
        lik=1e-10, 
        training_iter=10000,
        thresh=0.01,
        lr=0.1,
        rng=10,
        train_lik=False,
    ):
        self.gp_kernel = gp_kernel
        self.lik = lik
        self.training_iter = training_iter
        self.thresh = thresh
        self.lr = lr
        self.rng = rng
        self.train_lik=train_lik
        
        self.jitter = 1e-6
        self.Y_unwarp = copy.deepcopy(Yobs)
        self.utils = Utils()
        
        self.model = update_gp(
            Xobs,
            Yobs,
            gp_kernel,
            lik=self.lik,
            training_iter=self.training_iter, 
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )
    
    def cat_observations(self, X, Y):
        Xobs = self.model.train_inputs[0]
        Yobs = self.model.train_targets
        if len(self.model.train_targets.shape) == 0:
            Yobs = Yobs.unsqueeze(0)
        Xall = torch.cat([Xobs, X])
        Yall = torch.cat([Yobs, Y])
        return Xall, Yall
    
    def update_gp(self, X, Y):
        Xall, Yall = self.cat_observations(X, Y)
        self.model = update_gp(
            Xall,
            Yall,
            self.gp_kernel,
            lik=self.lik,
            training_iter=self.training_iter, 
            thresh=self.thresh, 
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )
        
    def retrain_gp(self):
        Xobs = self.model.train_inputs[0]
        Yobs = self.model.train_targets
        self.model = update_gp(
            Xobs,
            Yobs,
            self.gp_kernel,
            lik=self.lik,
            training_iter=self.training_iter, 
            thresh=self.thresh, 
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )
        
    def predict(self, x):
        mu, var = predict(x, self.model)
        return mu, var
    
    def predict_mean(self, x):
        mu, _ = predict(x, self.model)
        return mu