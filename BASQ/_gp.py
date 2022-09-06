import torch
import gpytorch
from BASQ._lbfgs import FullBatchLBFGS
from botorch.fit import fit_gpytorch_model
from gpytorch.priors.torch_priors import GammaPrior

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, gp_kernel):
        """
        Input:
            - train_x: torch.tensor, inputs. torch.Size(n_data, n_dims)
            - train_y: torch.tensor, observations
            - likelihood: gpytorch.likelihoods, GP likelihood model
            - gp_kernel: gpytorch.kernels, GP kernel model
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gp_kernel

    def forward(self, x):
        """
        Input:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Output:
            - torch.distributions, predictive posterior distribution at given x
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def set_gp(train_x, train_y, gp_kernel, device, lik=1e-10, rng=10, train_lik=False):
    """
    We can select whether or not to train likelihood variance.
    The true_likelihood query must be noiseless, so learning GP likelihood noise variance could be redundant.
    However, likelihood noise variance plays an important role in a limited number of samples in the early stage.
    So, setting interval constraints keeps the likelihood noise variance within a safe area.
    Otherwise, GP could confuse the meaningful multimodal peaks of true_likelihood as noise.

    Input:
        - train_x: torch.tensor, inputs. torch.Size(n_data, n_dims)
        - train_y: torch.tensor, observations
        - gp_kernel: gpytorch.kernels, GP kernel model
        - device: torch.device, cpu or cuda
        - lik: float, the initial value of GP likelihood noise variance
        - rng: int, tne range coefficient of GP likelihood noise variance
        - train_like: bool, flag whether or not to update GP likelihood noise variance

    Output:
        - model: gpytorch.models, function of GP model.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(lik / rng, lik * rng))
    model = ExactGPModel(train_x, train_y, likelihood, gp_kernel)
    model.covar_module.base_kernel.lengthscale_prior=GammaPrior(3.0, 6.0)
    model.covar_module.outputscale_prior=GammaPrior(2.0, 0.15)
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(lik),
    }
    
    model.initialize(**hypers)
    if not train_lik:
        model.likelihood.raw_noise.requires_grad = False

    if device.type == 'cuda':
        model = model.cuda()
        model.likelihood = model.likelihood.cuda()
    return model


class Closure:
    """
    Input:
        - mll: gpytorch.mlls.ExactMarginalLogLikelihood, marginal log likelihood
        - optimiser: torch.optim, L-BFGS-B optimizer from FullBatchLBFGS

    Output:
        - loss: torch.tensor, negative log marginal likelihood of GP
    """
    def __init__(self, mll, optimizer):
        self.mll = mll
        self.optimizer = optimizer
        self.train_inputs, self.train_targets = mll.model.train_inputs, mll.model.train_targets
    
    def __call__(self):
        self.optimizer.zero_grad()
        with gpytorch.settings.fast_computations(log_prob=True):
            output = self.mll.model(*self.train_inputs)
            args = [output, self.train_targets]
            l = -self.mll(*args).sum()
        return l

def train_GP(model, training_iter=50, thresh=0.01, lr=0.1, optimiser="L-BFGS-B"):
    """
    L-BFGS-B implementation is from https://github.com/hjmshi/PyTorch-LBFGS

    Input:
        - model: gpytorch.models, function of GP model.
        - train_iter: int, the maximum iteration for GP hyperparameter training.
        - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
        - lr: float, the learning rate of Adam optimiser
        - optimiser: string, select the optimiser ["L-BFGS-B", "BoTorch", "Adam"]

    Output:
        - model: gpytorch.models, function of GP model.
    """
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    if optimiser == "BoTorch":
        fit_gpytorch_model(mll)
    
    elif optimiser == "L-BFGS-B":
        # Use full-batch L-BFGS optimizer
        optimizer = FullBatchLBFGS(model.parameters())
        closure = Closure(mll, optimizer)
        loss = closure()
        loss.backward()
        loss_best = torch.tensor(1e10)
        
        for i in range(training_iter):
            # perform step and update curvature
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
            
            if loss.item() < loss_best:
                delta = torch.abs(loss_best - loss.detach())
                loss_best = loss.item()
                if delta < thresh:
                    break
            
    elif optimiser == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
        train_x = model.train_inputs[0]
        train_y = model.train_targets
        loss_best = torch.tensor(1e10)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if loss.item() < loss_best:
                delta = torch.abs(loss_best - loss.detach())
                loss_best = loss.item()
                if delta < thresh:
                    break
    else:
        raise Exception("The given optimiser is not defined")
    return model
    

def update_gp(train_x, train_y, gp_kernel, device, lik=1e-10, training_iter=50, thresh=0.01, lr=0.1, rng=10, train_lik=False, optimiser="L-BFGS-B"):
    """
    Input:
        - train_x: torch.tensor, inputs. torch.Size(n_data, n_dims)
        - train_y: torch.tensor, observations
        - gp_kernel: gpytorch.kernels, GP kernel model
        - device: torch.device, cpu or cuda
        - lik: float, the initial value of GP likelihood noise variance
        - train_iter: int, the maximum iteration for GP hyperparameter training.
        - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
        - lr: float, the learning rate of Adam optimiser
        - rng: int, tne range coefficient of GP likelihood noise variance
        - train_like: bool, flag whether or not to update GP likelihood noise variance
        - optimiser: string, select the optimiser ["L-BFGS-B", "BoTorch", "Adam"]

    Output:
        - model: gpytorch.models, function of GP model.
    """
    model = set_gp(train_x, train_y, gp_kernel, device, lik=lik, rng=rng, train_lik=train_lik)
    model = train_GP(model, training_iter=training_iter, thresh=thresh, lr=lr, optimiser=optimiser)
    return model


def predict(test_x, model):
    """
    Fast variance inference is made with LOVE via fast_pred_var().
    For accurate variance inference, you can just comment out the part.

    Input:
        - model: gpytorch.models, function of GP model.

    Output:
        - pred.mean; torch.tensor, the predictive mean
        - pred.variance; torch.tensor, the predictive variance
    """
    model.eval()
    model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(test_x))
    return pred.mean, pred.variance

def get_cov_cache(model):
    """
    woodbury_inv = K(Xobs, Xobs)^(-1)
    S @ S.T = woodbury_inv

    Input:
        - model: gpytorch.models, function of GP model, typically self.wsabi.model in _basq.py

    Output:
        - woodbury_inv: torch.tensor, the inverse of Gram matrix K(Xobs, Xobs)^(-1)
        - Xobs: torch.tensor, the observed inputs X
        - lik_var: torch.tensor, the GP likelihood noise variance
    """
    Xobs = model.train_inputs[0]
    lik_var = model.likelihood.noise
    try:
        S = model.prediction_strategy.covar_cache
    except AttributeError:
        model.eval()
        mean = Xobs[0].unsqueeze(0)
        model(mean)
        S = model.prediction_strategy.covar_cache
    woodbury_inv = S @ S.T
    return woodbury_inv, Xobs, lik_var

def predictive_covariance(x, y, model):
    """
    Input:
        - x: torch.tensor, inputs x
        - y: torch.tensor, inputs y
        - model: gpytorch.models, function of GP model.

    Output:
        - cov_xy: torch.tensor, predictive covariance matrix
    """
    woodbury_inv, Xobs, lik_var = get_cov_cache(model)
    Kxy = model.covar_module.forward(x, y)
    KxX = model.covar_module.forward(x, Xobs)
    KXy = model.covar_module.forward(Xobs, y)
    cov_xy = Kxy - KxX @ woodbury_inv @ KXy
    
    d = min(len(x), len(y))
    cov_xy[range(d), range(d)] = cov_xy[range(d), range(d)] + lik_var
    return cov_xy
