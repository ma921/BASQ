import torch
import gpytorch


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
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(lik),
        'covar_module.base_kernel.lengthscale': torch.tensor(2),
        'covar_module.outputscale': torch.tensor(2.),
    }
    model.initialize(**hypers)
    if not train_lik:
        model.likelihood.raw_noise.requires_grad = False

    if device.type == 'cuda':
        model = model.cuda()
        model.likelihood = model.likelihood.cuda()
    return model


def train_GP(model, training_iter=50, thresh=0.01, lr=0.1):
    """
    Input:
        - model: gpytorch.models, function of GP model.
        - train_iter: int, the maximum iteration for GP hyperparameter training.
        - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
        - lr: float, the learning rate of Adam optimiser

    Output:
        - model: gpytorch.models, function of GP model.
    """
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
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
    return model


def update_gp(train_x, train_y, gp_kernel, device, lik=1e-10, training_iter=50, thresh=0.01, lr=0.1, rng=10, train_lik=False):
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

    Output:
        - model: gpytorch.models, function of GP model.
    """
    model = set_gp(train_x, train_y, gp_kernel, device, lik=lik, rng=rng, train_lik=train_lik)
    model = train_GP(model, training_iter=training_iter, thresh=thresh, lr=lr)
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