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

def set_gp(train_x, train_y, gp_kernel):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, gp_kernel)
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    return model, likelihood
    
def train_GP(model, likelihood, training_iter=50, thresh=0.01):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
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
    return model, likelihood

def update_gp(train_x, train_y, model, training_iter=50):
    model, likelihood = set_gp(train_x, train_y, model.covar_module)
    model, likelihood = train_GP(model, likelihood, training_iter=training_iter)
    return model, likelihood

def predict(test_x, model, likelihood):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
    return pred.mean, pred.variance