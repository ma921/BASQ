import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, gp_kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gp_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def set_gp(train_x, train_y, gp_kernel, lik=1e-10, rng=10, train_lik=False):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(lik/rng, lik*rng))
    model = ExactGPModel(train_x, train_y, likelihood, gp_kernel)
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(lik),
        'covar_module.base_kernel.lengthscale': torch.tensor(2),
        'covar_module.outputscale': torch.tensor(2.),
    }
    model.initialize(**hypers)
    if not train_lik:
        model.likelihood.raw_noise.requires_grad = False
    
    if torch.cuda.is_available():
        model = model.cuda()
        model.likelihood = model.likelihood.cuda()
    return model
    
def train_GP(model, training_iter=50, thresh=0.01, lr=0.1):
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    loss_best = torch.tensor(1e8)

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

def update_gp(train_x, train_y, gp_kernel, lik=1e-10 , training_iter=50, thresh=0.01, lr=0.1, rng=10, train_lik=False):
    model = set_gp(train_x, train_y, gp_kernel, lik=lik, rng=rng, train_lik=train_lik)
    model = train_GP(model, training_iter=training_iter, thresh=thresh, lr=lr)
    return model

def predict(test_x, model):
    model.eval()
    model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(test_x))
    return pred.mean, pred.variance