import torch
from ._gp import predict
from ._utils import Utils
from ._acquisition_function import SquareRootAcquisitionFunction


class PriorSampler:
    def __init__(self, prior, n_rec, nys_ratio, device):
        """
        Args:
           - prior: torch.distributions, prior distribution
           - n_rec: int, number of subsamples for empirical measure of kernel recomnbination
           - nys_ratio: float, subsubsampling ratio for Nystrom.
           - device: torch.device, cpu or cuda
        """
        self.prior = prior
        self.n_rec = n_rec
        self.nys_ratio = nys_ratio
        self.device = device

    def __call__(self, n_rec):
        """
        Args:
           - n_rec: int, number of subsamples for empirical measure of kernel recomnbination

        Returns:
           - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
           - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
           - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        """
        pts_rec = self.prior.sample(sample_shape=torch.Size([n_rec])).to(self.device)
        pts_nys = pts_rec[:int(self.n_rec * self.nys_ratio)]
        w = torch.ones(n_rec) / n_rec
        return pts_nys, pts_rec, w.to(self.device)


class UncertaintySampler(SquareRootAcquisitionFunction):
    def __init__(
        self,
        prior,
        model,
        n_rec,
        nys_ratio,
        device,
        sampling_method="approx",
        ratio=0.5,
        ratio_super=100,
        n_gaussians=100,
        threshold=1e-5,
    ):
        """
        Args:
           - prior: torch.distributions, prior distribution.
           - model: gpytorch.models, function of GP model
           - n_rec: int, subsampling size for kernel recombination
           - nys_ratio: float, subsubsampling ratio for Nystrom. Number of Nystrom samples is nys_ratio * n_rec
           - device: torch.device, cpu or cuda
           - sampling_method; string, ["exact", "approx"]
           - ratio: float, mixing ratio of prior and uncertainty sampling, 0 < r < 1
           - n_gaussians: int, number of Gaussians approximating the GP-modelled acquisition function
           - threshold: float, threshold to cut off the insignificant Gaussians
        """
        super().__init__(prior, model, device, n_gaussians=n_gaussians, threshold=threshold)
        self.model = model
        self.ratio = ratio
        self.nys_ratio = nys_ratio
        self.ratio_super = ratio_super
        self.device = device
        self.sampling_method = sampling_method
        self.utils = Utils(device)

    def pdf(self, X):
        """
        Args:
           - X: torch.tensor, inputs

        Returns:
           - pdf: the value at given X of probability density function of approximated sparse Gaussian Mixture Model (GMM)
        """
        if self.ratio == 0:
            return self.prior.log_prob(X).exp()
        elif self.ratio == 1:
            return self.joint_pdf(X)
        else:
            g_pdf = self.joint_pdf(X)
            f_pdf = self.prior.log_prob(X).exp()
            return ((1 - self.ratio) * f_pdf + self.ratio * g_pdf) / f_pdf

    def SIR(self, X, weights, n_return):
        """
        Sequentail Importance Resample (SIR).
        Resample from the weighted samples via importance sampling.

        Args:
           - X: torch.tensor, inputs
           - weights: torch.tensor, weights for importance sampling. This is not necessarily required to be normalised.
           - n_return: torch.tensor, inputs, number of samples to be returned.

        Returns:
           - samples: resampled samples.
        """
        draw = torch.multinomial(weights, n_return)
        return X[draw]

    def approx(self, n):
        """
        Proposal distribution g(x) = (1-r) f(x) + r A(x),
                              f(x) = π(x),
        where r is self.ratio, m(x) and C(x) are the mean and varinace of square-root kernel.
        weights w_IS = f(x) / g(x)
        samples for Nystrom should be sampled from f(x), thus we adopt SIR.
        pts_nys <- SIR(pts_rec, weights) is equivalent to be sampled from f(x).

        Args:
           - n: int, number of samples to be returned

        Returns:
           - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
           - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
           - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        """
        if self.ratio == 0:
            pts_rec = self.prior.sample(torch.Size([n]))
        elif self.ratio == 1:
            pts_rec = self.sampling(n)
        else:
            pts_rec = torch.cat([
                self.sampling(int(self.ratio * n)),
                self.prior.sample(torch.Size([int((1 - self.ratio) * n)])),
            ])

        mean, var = predict(pts_rec, self.model)
        w = torch.exp(torch.log(torch.abs(mean)) + self.prior.log_prob(pts_rec) - torch.nan_to_num(self.pdf(pts_rec)))
        w = torch.nan_to_num(w)
        if torch.sum(w) == 0:
            weights = torch.ones(len(w)) / len(w)
        else:
            weights = w / torch.sum(w)

        n_nys = int(n * self.nys_ratio)
        pts_nys = self.SIR(pts_rec, weights, n_nys)
        return pts_nys, pts_rec, weights

    def SIR_from_mean(self, n_super, n):
        """
        Proposal distribution g(x) = B(x),
                              f(x) = |m(x)| π(x),
        weights w_IS = f(x) / g(x)

        Args:
           - n_super: int, number of supersamples for SIR
           - n: int, number of samples to be returned

        Returns:
           - samples: resampled samples.
        """
        X_pi = self.sampling_mean(n_super)
        mean, _ = predict(X_pi, self.model)
        mean_log = mean.abs().log()
        prior_log = self.utils.safe_mvn_prob(self.prior.loc, self.prior.covariance_matrix, X_pi).log()
        sampler_log = torch.nan_to_num(self.joint_pdf_mean(X_pi))

        w_mpi_B = torch.exp(
            mean_log + prior_log - sampler_log
        )
        X_f = self.SIR(X_pi, w_mpi_B, n)
        return X_f

    def SIR_from_AF(self, n_super, n):
        """
        Proposal distribution g(x) = A(x),
                              f(x) = C(x)π(x),
        weights w_IS = f(x) / g(x)

        Args:
           - n_super: int, number of supersamples for SIR
           - n: int, number of samples to be returned

        Returns:
           - samples: resampled samples.
        """
        X_A = self.sampling(n_super)
        _, var_A = predict(X_A, self.model)
        var_log = var_A.log()
        prior_log = self.utils.safe_mvn_prob(self.prior.loc, self.prior.covariance_matrix, X_A).log()
        sampler_log = torch.nan_to_num(self.joint_pdf(X_A)).log()

        w_C_A = torch.exp(
            var_A.log() + prior_log - sampler_log
        )
        X_rec = self.SIR(X_A, w_C_A, n)
        return X_rec

    def calc_weights(self, pts_rec):
        """
        weights w_IS = f(x) / g(x)

        Args:
           - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination

        Returns:
           - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        """
        mean_rec, var_rec = predict(pts_rec, self.model)
        f_rec = torch.exp(torch.abs(mean_rec).log() + self.prior.log_prob(pts_rec))
        if self.ratio < 1:
            g_rec = torch.exp(
                torch.log(self.ratio * var_rec + (1 - self.ratio) * torch.abs(mean_rec))
                + self.prior.log_prob(pts_rec)
            )
        else:
            g_rec = torch.exp(
                torch.tensor(self.ratio).log() + var_rec.log() + self.prior.log_prob(pts_rec)
            )
        w_IC = f_rec / g_rec
        w_IC = w_IC / w_IC.sum()
        return w_IC

    def exact(self, n):
        """
        Proposal distribution g(x) = (1-r) f(x) + r C(x)π(x),
                              f(x) = |m(x)| π(x),
        where r is self.ratio, m(x) and C(x) are the mean and varinace of square-root kernel.
        weights w_IS = f(x) / g(x)
        samples for Nystrom should be sampled from f(x), thus we adopt SIR.
        pts_nys <- SIR(pts_rec, weights) is equivalent to be sampled from f(x).
        If r = 0, this simply samples from f(x) = |m(x)| π(x).
        If r = 1, this becomes pure uncertainty sampling.

        Args:
           - n: int, number of samples to be returned

        Returns:
           - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
           - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
           - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        """
        n_nys = int(n * self.nys_ratio)

        if self.ratio == 0:
            n_super = int(self.ratio_super * n)
            pts_rec = self.SIR_from_mean(n_super, n)
            pts_nys = pts_rec[:n_nys]
            w_IC = torch.ones(n).to(self.device) / n
            return pts_nys, pts_rec, w_IC
        elif self.ratio == 1:
            n_super = int(self.ratio_super * n)
            pts_rec = self.SIR_from_AF(n_super, n)
            w_IC = self.calc_weights(pts_rec)
            pts_nys = self.SIR_from_mean(n, n_nys)
            return pts_nys, pts_rec, w_IC
        else:
            n_super = int(self.ratio_super * (1 - self.ratio) * n)
            n_pi = int((1 - self.ratio) * n)
            X_f = self.SIR_from_mean(n_super, n_pi)
            pts_nys = X_f[:n_nys]

            n_super = int(self.ratio_super * self.ratio * n)
            n_rec = int(self.ratio * n)
            X_rec = self.SIR_from_AF(n_super, n_rec)
            pts_rec = torch.cat([X_rec, X_f])
            w_IC = self.calc_weights(pts_rec)
            return pts_nys, pts_rec, w_IC

    def __call__(self, n):
        """
        Args:
           - n: int, number of samples to be returned

        Returns:
           - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
           - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
           - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        """
        if self.sampling_method == "approx":
            return self.approx(n)
        elif self.sampling_method == "exact":
            return self.exact(n)
        else:
            raise Exception("The given sampling method is undefined.")
