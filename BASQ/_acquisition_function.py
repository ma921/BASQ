import torch
from ._gaussian_calc import GaussianCalc
from torch.distributions.multivariate_normal import MultivariateNormal


class SquareRootAcquisitionFunction(GaussianCalc):
    def __init__(self, prior, model, device, n_gaussians=100, threshold=1e-5):
        """
        Inherited the functions from GaussianCalc.

        Args:
        - prior: torch.distributions, prior distribution
        - device: torch.device, cpu or cuda
        """
        super().__init__(prior, device)
        self.n_gaussians = n_gaussians    # number of Gaussians for uncertainty sampling
        self.threshold = threshold        # threshold to cut off the small weights
        self.update(model)

    def update(self, model):
        """
        Args:
        - model: gpytorch.models, function of GP model, typically model = self.wsabi.model in _basq.py
        """
        self.parameters_extraction(model)
        self.wA, self.wAA, self.mu_AA, self.sigma_AA = self.sparseGMM()
        self.d_AA = len(self.mu_AA)
        self.w_mean, self.mu_mean, self.sig_mean = self.sparseGMM_mean()
        self.d_mean = len(self.mu_mean)

    def sparseGMM(self):
        """
        See details on factorisation trick and sparse GMM sampler in Sapplementary.
        https://arxiv.org/abs/2206.04734

        Returns:
            - w1: torch.tensor, the weight of prior distribution
            - w2: torch.tensor, the weights of other normal distributions
            - mu2: torch.tensor, the mean vectors of other normal distributions
            - sigma2: torch.tensor, the covariance matrix of other normal distributions
        """
        i, j = torch.where(self.woodbury_inv < 0)
        _w1_ = self.outputscale
        _w2_ = torch.abs((self.v**2) * self.woodbury_inv[i, j])
        _Z = _w1_ + torch.sum(_w2_)
        _w1, _w2 = _w1_ / _Z, _w2_ / _Z

        Winv = self.W.inverse()
        Sinv = self.prior.covariance_matrix.inverse()
        sigma2 = (2 * Winv + Sinv).inverse()

        _idx = _w2.argsort(descending=True)[:self.n_gaussians]
        idx = _idx[_w2[_idx] > self.threshold]
        Xi = self.Xobs[i[idx]]
        Xj = self.Xobs[j[idx]]

        w2 = _w2[idx]
        mu2 = (sigma2 @ Winv @ (Xi + Xj).T).T + sigma2 @ Sinv @ self.prior.loc

        zA = _w1 + torch.sum(w2)
        w1, w2 = _w1 / zA, w2 / zA
        return w1, w2, mu2, sigma2

    def joint_pdf(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - first/first+second: torch.tensor, the values of probability density function of approximated A(x)
        """
        d_x = len(x)

        # calculate the first term
        Npdfs_A = self.utils.safe_mvn_prob(
            self.prior.loc,
            self.prior.covariance_matrix,
            x,
        )
        first = self.wA * Npdfs_A

        # calulate the second term
        if len(self.wAA) == 0:
            return first
        else:
            x_AA = (torch.tile(self.mu_AA, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(
                self.d_AA * d_x, self.n_dims
            )
            Npdfs_AA = self.utils.safe_mvn_prob(
                torch.zeros(self.n_dims).to(self.device),
                self.sigma_AA,
                x_AA,
            ).reshape(d_x, self.d_AA)

            f_AA = self.wAA.unsqueeze(0) * Npdfs_AA
            second = f_AA.sum(axis=1)
            return first + second

    def sampling(self, n):
        """
        Args:
            - n: int, number of samples to be sampled.

        Returns:
            - samplesA/samplesAA: torch.tensor, the samples from approximated A(x)
        """
        cntA = (n * self.wA).type(torch.int)
        samplesA = self.prior.sample(torch.Size([cntA])).to(self.device)

        if len(self.wAA) == 0:
            return samplesA
        else:
            cntAA = (n * self.wAA).type(torch.int)
            samplesAA = torch.cat([
                MultivariateNormal(
                    self.mu_AA[i],
                    self.sigma_AA,
                ).sample(torch.Size([cnt])).to(self.device)
                for i, cnt in enumerate(cntAA)
            ])
            return torch.cat([samplesA, samplesAA])

    def sparseGMM_mean(self):
        """
        Returns:
            - weights: torch.tensor, the weight of approximated GP mean functions
            - mu_mean: torch.tensor, the mean vectors of approximated GP mean functions
            - sig_prime: torch.tensor, the covariance matrix of approximated GP mean functions
        """
        Winv = self.W.inverse()
        Sinv = self.prior.covariance_matrix.inverse()
        sig_prime = (Winv + Sinv).inverse()
        mu_prime = (sig_prime @ (
            (Winv @ self.Xobs.T).T + Sinv @ self.prior.loc
        ).T).T
        npdfs = MultivariateNormal(
            self.prior.loc,
            self.W + self.prior.covariance_matrix,
        ).log_prob(self.Xobs).exp()
        omega_prime = self.woodbury_vector * npdfs

        weights = omega_prime / omega_prime.sum()
        W_prime = weights * MultivariateNormal(
            self.prior.loc,
            sig_prime,
        ).log_prob(mu_prime).exp()

        W_pos = W_prime[W_prime > 0].sum()
        W_neg = W_prime[W_prime < 0].sum().abs()
        N_pos = int(W_pos / (W_pos + W_neg) * self.n_gaussians)
        N_neg = self.n_gaussians - N_pos
        idx_pos = W_prime[W_prime > 0].argsort(descending=True)[:N_pos]
        idx_neg = W_prime[W_prime < 0].argsort()[:N_neg]
        weights_pos = weights[W_prime > 0][idx_pos]
        weights_neg = weights[W_prime < 0][idx_neg].abs()
        weights = torch.cat([weights_pos, weights_neg])
        mu_pos = mu_prime[W_prime > 0][idx_pos]
        mu_neg = mu_prime[W_prime < 0][idx_neg]
        mu_mean = torch.cat([mu_pos, mu_neg])

        idx_weights = weights > (self.threshold * weights.sum())
        weights = weights[idx_weights]
        mu_mean = mu_mean[idx_weights]
        weights = weights / weights.sum()
        return weights, mu_mean, sig_prime

    def joint_pdf_mean(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - first/first+second: torch.tensor, the values of probability density function of approximated GP mean functions
        """
        d_x = len(x)

        x_AA = (torch.tile(self.mu_mean, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(
            self.d_mean * d_x, self.n_dims
        )
        Npdfs_AA = self.utils.safe_mvn_prob(
            torch.zeros(self.n_dims).to(self.device),
            self.sig_mean,
            x_AA,
        ).reshape(d_x, self.d_mean)

        f_AA = self.w_mean.unsqueeze(0) * Npdfs_AA
        pdf = f_AA.sum(axis=1)
        return pdf

    def sampling_mean(self, n):
        """
        Args:
            - n: int, number of samples to be sampled.

        Returns:
            - samples: torch.tensor, the samples from approximated GP mean functions
        """
        cnts = (n * self.w_mean).type(torch.int)
        samples = torch.cat([
            MultivariateNormal(
                self.mu_mean[i],
                self.sig_mean,
            ).sample(torch.Size([cnt])).to(self.device)
            for i, cnt in enumerate(cnts)
        ])
        return samples
