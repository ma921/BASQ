import time
import torch
from ._rchq import recombination
from ._parameters import Parameters

class BASQ(Parameters):
    def __init__(self, Xobs, Yobs, prior, true_likelihood, device):
        super().__init__(Xobs, Yobs, prior, true_likelihood, device)
    
    def joint_posterior(self, x, EZy):
        return self.predict_mean(x) * self.prior.log_prob(x).exp() / EZy
        
    def quadratures(self):
        mvn_max = self.unimodal_approx()
        EZy_prior, VarZy_prior = self.kq.prior_max(mvn_max)
        model_IS, uni_sampler = self.uniform_trans(mvn_max)
        EZy_uni, VarZy_uni = self.kq.uniform_trans(model_IS, uni_sampler)
        return EZy_prior, VarZy_prior, EZy_uni, VarZy_uni
    
    def run_rchq(self, pts_nys, pts_rec, w_IS, kernel):
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
        self.update(X,Y)
        
    def run(self, n_batch):
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
            EZy_prior, VarZy_prior, EZy_uni, VarZy_uni = self.quadratures()
            self.retrain()
        return torch.tensor(results)