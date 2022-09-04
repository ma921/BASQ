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
        device,
        lik=1e-10,
        training_iter=10000,
        thresh=0.01,
        lr=0.1,
        rng=10,
        train_lik=False,
    ):
        """
        Input:
           - Xobs: torch.tensor, X samples, X belongs to prior measure.
           - Yobs: torch.tensor, Y observations, Y = true_likelihood(X).
           - gp_kernel: gpytorch.kernels, GP kernel function
           - device: torch.device, device, cpu or cuda
           - lik: float, the initial value of GP likelihood noise variance
           - train_iter: int, the maximum iteration for GP hyperparameter training.
           - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
           - lr: float, the learning rate of Adam optimiser
           - rng: int, tne range coefficient of GP likelihood noise variance
           - train_like: bool, flag whether or not to update GP likelihood noise variance
        """
        self.gp_kernel = gp_kernel
        self.device = device
        self.lik = lik
        self.training_iter = training_iter
        self.thresh = thresh
        self.lr = lr
        self.rng = rng
        self.train_lik = train_lik

        self.jitter = 1e-6
        self.Y_unwarp = copy.deepcopy(Yobs)
        self.utils = Utils(device)

        self.model = update_gp(
            Xobs,
            Yobs,
            gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )

    def cat_observations(self, X, Y):
        """
        Input:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, Y observations to be added to the existing data Yobs

        Output:
           - Xall: torch.tensor, X samples that contains all samples
           - Yall: torch.tensor, Y observations that contains all observations
        """
        Xobs = self.model.train_inputs[0]
        Yobs = self.model.train_targets
        if len(self.model.train_targets.shape) == 0:
            Yobs = Yobs.unsqueeze(0)
        Xall = torch.cat([Xobs, X])
        Yall = torch.cat([Yobs, Y])
        return Xall, Yall

    def update_gp(self, X, Y):
        """
        Input:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, Y observations to be added to the existing data Yobs
        """
        Xall, Yall = self.cat_observations(X, Y)
        self.model = update_gp(
            Xall,
            Yall,
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
        Xobs = self.model.train_inputs[0]
        Yobs = self.model.train_targets
        self.model = update_gp(
            Xobs,
            Yobs,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
        )

    def predict(self, x):
        """
        Input:
           - x: torch.tensor, x locations to be predicted

        Output:
           - mu: torch.tensor, predictive mean at given locations x.
           - var: torch.tensor, predictive variance at given locations x.
        """
        mu, var = predict(x, self.model)
        return mu, var

    def predict_mean(self, x):
        """
        Input:
           - x: torch.tensor, x locations to be predicted

        Output:
           - mu: torch.tensor, predictive mean at given locations x.
        """
        mu, _ = predict(x, self.model)
        return mu
