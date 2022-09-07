import time
import torch
from ._rchq import recombination
from ._parameters import Parameters


class BASQ(Parameters):
    def __init__(self, Xobs, Yobs, prior, true_likelihood, device):
        """
        Goal: Estimate both evidence and posterior in one go with minimal queries.

        Args:
           - Xobs; torch.tensor, X samples, X belongs to prior measure.
           - Yobs; torch.tensor, Y observations, Y = true_likelihood(X).
           - prior; torch.distributions, prior distribution.
           - true_likelihood; function of y = function(x), true likelihood to be estimated.
           - device; torch.device, device, cpu or cuda

        Results:
           - evidence (a.k.a. marginal likelihood);
             EZy, VarZy = self.kq.quadrature()
             EZy; the mean of evidence
             VarZy; the variance of evidence
           - posterior; self.joint_posterior(x, EZy)
        """
        super().__init__(Xobs, Yobs, prior, true_likelihood, device)

    def joint_posterior(self, x, EZy):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)
            - EZy: float, the mean of the evidence

        Returns:
            - torch.tensor, the posterior of given x
        """
        return self.predict_mean(x) * self.prior.log_prob(x).exp() / EZy

    def quadratures(self):
        """
        Calculate two additional quadratures.
        The following quadratures are applicable only for WSABI-BQ.
        - Prior maximisation; the prior distribution is optimised to maximise the evidence
        - Uniform transformation; the prior distribution is transformed into uniform
                                  distrubution via impotance sampling.

        Args:
            - EZy_prior: float, the mean of the evidence when the prior is optimised to maximise the evidence
            - VarZy_prior: float, the variance of the evidence when the prior is optimised to maximise the evidence
            - EZy_uni float, the mean of the evidence when the prior is transformed into uniform distribution
            - VarZy_uni: float, the variance of the evidence when the prior is transformed into uniform distribution
        """
        mvn_max = self.unimodal_approx()
        EZy_prior, VarZy_prior = self.kq.prior_max(mvn_max)
        model_IS, uni_sampler = self.uniform_trans(mvn_max)
        EZy_uni, VarZy_uni = self.kq.uniform_trans(model_IS, uni_sampler)
        return EZy_prior, VarZy_prior, EZy_uni, VarZy_uni

    def run_rchq(self, pts_nys, pts_rec, w_IS, kernel):
        """
        Args:
            - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
            - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
            - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
            - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)

        Returns:
            - x: torch.tensor, the sparcified samples from pts_rec. The number of samples are determined by self.batch_size
            - w: torch.tensor, the positive weights for kernel quadrature as discretised summation.
        """
        idx, w = recombination(
            pts_rec,
            pts_nys,
            self.batch_size,
            kernel,
            self.device,
            init_weights=w_IS,
        )
        x = pts_rec[idx]
        return x, w

    def run_basq(self):
        if self.sampler_type == "uncertainty":
            self.sampler.update(self.wsabi.model)
        pts_nys, pts_rec, w_IS = self.sampler(self.n_rec)
        X, _ = self.run_rchq(pts_nys, pts_rec, w_IS, self.kernel)
        Y = self.true_likelihood(X)
        self.update(X, Y)

    def run(self, n_batch):
        """
        Args:
            - n_batch: int, number of iteration. The total query is n_batch * self.batch_size

        Returns:
            - results: torch.tensor, [overhead, EZy, VarZy]
        """
        results = []
        overhead = 0
        for _ in range(n_batch):
            s = time.time()
            self.run_basq()
            _overhead = time.time() - s
            if self.show_progress:
                EZy, VarZy = self.kq.quadrature()
                results.append([_overhead, EZy, VarZy])
            else:
                overhead += _overhead
        if not self.show_progress:
            EZy, VarZy = self.kq.quadrature()
            results.append([overhead, EZy, VarZy])

        if self.bq_model == "wsabi":
            self.wsabi.memorise_parameters()
            EZy_prior, VarZy_prior, EZy_uni, VarZy_uni = self.quadratures()
            self.wsabi.remind_parameters()
            self.retrain()
        return torch.tensor(results)
