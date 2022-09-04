import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from BASQ.experiment.gmm import GMM
from BASQ._basq import BASQ
from BASQ._metric import KLdivergence
import warnings
warnings.filterwarnings('ignore')


def set_basq():
    """
    Bayesian Inference Modelling

    Goal: estimation of both evidence and posterior in one go.

    Bayesian inference to be solved:
        - true_likelihood: a likelihood modelled with Gaussian mixture
                           We wish to estimate this function only from the queries to this.
        - prior: a unimodal multivariate normal distribution.
                 mean: mu_pi
                 covariance matrix: cov_pi
        - true evidence: 1

    Note:
    - initial guess: (X, Y) = (train_x, train_y)
    - metric for posterior inference: the KL divergence between true and estimated posterior.
              posterior = E[GP-modelled-likelihood] * prior / marginal-likelihood
    - metric for evidence: logarithmic mean absolute error between true and estimated evidence.
              logMAE = torch.log(Z_estimated - Z_true)

    Output:
        - prior: torch.distributions, prior distribution
        - train_x: torch.tensor, initial sample(s)
        - train_y: torch.tensor, initial observation of true likelihood query
        - true_likelihood: function of y = function(x), true likelihood to be estimated.
        - metric: function of KL = metric(basq, EZy), where basq is the trained BQ model, and EZy is the mean of the evidence
        - device: torch.device, cpu or cuda
    """
    # device = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_dim = 10
    mu_pi = torch.zeros(num_dim).to(device)
    cov_pi = 2 * torch.eye(num_dim).to(device)
    true_likelihood = GMM(num_dim, mu_pi, cov_pi, device)

    # BQ modelling
    train_x = torch.rand(2, num_dim).to(device)
    train_y = true_likelihood(train_x)
    prior = MultivariateNormal(mu_pi, cov_pi)

    # evaluation setting
    Z_true = 1
    test_x = prior.sample(sample_shape=torch.Size([10000]))
    metric = KLdivergence(prior, test_x, Z_true, device, true_likelihood)
    return prior, train_x, train_y, true_likelihood, metric, device


if __name__ == "__main__":
    torch.manual_seed(0)
    n_batch = 10           # the number of BASQ iteration

    prior, train_x, train_y, true_likelihood, metric, device = set_basq()
    basq = BASQ(
        train_x,
        train_y,
        prior,
        true_likelihood,
        device,
    )

    results = basq.run(n_batch)
    print(
        "overhead: " + str(results[:, 0].sum().item()) + " [s]\n"
        + "overhead per sample: " + str(results[:, 0].sum().item() / basq.batch_size / n_batch)
        + " [s]\n"
        + "final E[Z|y]: " + str(results[-1, 1].item()) + "\n"
        + "final Var[Z|y]: " + str(results[-1, 2].item()) + "\n"
        + "logMAE: " + str(torch.log(torch.tensor(abs(results[-1][1] - metric.Z_true))).item()) + "\n"
        + "logKL: " + str(torch.log(metric(basq, results[-1][1])).item()) + "\n"
    )
