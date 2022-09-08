import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def random_choice(pop_tensor, num_samples, device):
    """Use torch.randperm to generate indices on a GPU tensor."""
    return pop_tensor[torch.randperm(len(pop_tensor), device=device)[:num_samples]]


class GMM:
    def __init__(self, dim, mu_pi, cov_pi, device):
        self.dim = dim
        self.device = device
        self.mu_pi = mu_pi
        self.cov_pi = cov_pi
        self.initialising()

    def initialising(self):
        self.n_comp = self.component_generator()
        self.means = torch.stack([self.mean_generator() for _ in range(self.n_comp)]).to(self.device)
        self.cov = self.cov_generator()
        self.weights = self.weights_calc()

    def IO(self, x):
        if len(x.size()) == 1:
            return x.unsqueeze(1)
        else:
            return x

    def mean_generator(self):
        return (3 * (2 * torch.rand(self.dim) - 1)).to(self.device)

    def cov_generator(self):
        return torch.diag(3 * torch.rand(self.dim) + 1).to(self.device)

    def component_generator(self):
        return random_choice(torch.arange(10, 16), 1, self.device).item()

    def weights_calc(self):
        Npdfs = MultivariateNormal(
            self.mu_pi,
            self.cov_pi + self.cov,
        ).log_prob(self.means).exp().to(self.device)
        return (1 / Npdfs) / self.n_comp

    def __call__(self, x):
        x = self.IO(x)
        d_x = len(x)
        x = (torch.tile(self.means, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(self.n_comp * d_x, self.dim).to(self.device)
        Npdfs = MultivariateNormal(
            torch.zeros(self.dim).to(self.device),
            self.cov,
        ).log_prob(x).exp().reshape(d_x, self.n_comp).to(self.device)

        f = self.weights.unsqueeze(0) * Npdfs
        return torch.sum(f, axis=1).to(self.device)
