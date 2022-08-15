import gp
import time
import torch
from rchq_torch import recombination

class BASQ:
    def __init__(
        self,
        prior,
        model,
        likelihood,
        true_likelihood,
        n_rec, 
        nys_ratio, 
        training_iter,
        batch_size,
        quad_ratio,
        show_progress=True
    ):
        self.n_rec = n_rec
        self.nys_ratio = nys_ratio
        self.training_iter = training_iter
        self.batch_size = batch_size
        self.quad_ratio = quad_ratio
        self.show_progress = show_progress
        
        self.prior = prior
        self.model = model
        self.likelihood = likelihood
        self.true_likelihood = true_likelihood
        self.kernel = lambda x,y: self.model.covar_module.forward(x,y)
        
    def Sampler(self, n_rec):
        pts_rec = self.prior.sample(sample_shape=torch.Size([n_rec]))
        pts_nys = pts_rec[:int(self.n_rec*self.nys_ratio)]
        w = torch.ones(n_rec) / n_rec
        return pts_nys, pts_rec, w

    def run_basq(self):
        pts_nys, pts_rec, w_IS = self.Sampler(self.n_rec)
        X, _ = self.run_rchq(pts_nys, pts_rec, w_IS)
        Y = self.true_likelihood(X)
        self.model, self.likelihood = gp.update_gp(X,Y, self.model, training_iter=self.training_iter)
        #return model, likelihood, kernel

    def run_rchq(self, pts_nys, pts_rec, w_IS):
        idx, w = recombination(
            pts_rec,
            pts_nys,
            self.batch_size,
            self.kernel,
            init_weights=w_IS,
        )
        x = pts_rec[idx]
        return x, w

    def quadrature(self):
        n_batch = int(self.batch_size*self.quad_ratio)
        pts_nys, pts_rec, w_IS = self.Sampler(int(self.quad_ratio*self.n_rec))
        X, w = self.run_rchq(pts_nys, pts_rec, w_IS)
        mean, var = gp.predict(X, self.model, self.likelihood)
        EZy = (w@mean).item()
        VarZy = (w@self.kernel(X,X)@w).item()
        print("E[Z|y]: "+str(EZy)+"  Var[Z|y]: "+str(VarZy))
        return EZy, VarZy
    
    def run(self, n_batch):
        results = []
        overhead = 0
        for _ in range(n_batch):
            s = time.time()
            self.run_basq()
            _overhead = time.time() - s
            if self.show_progress:
                EZy, VarZy = self.quadrature()
                results.append([_overhead, EZy, VarZy])
            else:
                overhead += _overhead
        if not self.show_progress:
            EZy, VarZy = self.quadrature()
            results.append([overhead, EZy, VarZy])
        return torch.tensor(results)

    ### Post Process ###
    def joint_posterior_WSABIL(self, x, EZy):
        Xobs = self.model.train_inputs[0]
        Yobs = self.model.train_targets
        alpha = min(Yobs)
        y_warp = torch.sqrt(2*(Yobs-alpha))
        model_warp, likelihood_warp = gp.update_gp(Xobs, y_warp, self.model, training_iter=self.training_iter)
        _mean, _ = gp.predict(x, model_warp, likelihood_warp)
        likelihood_mean = alpha + 1/2 * _mean**2
        return likelihood_mean * self.prior.log_prob(x).exp() / EZy

    def joint_posterior_WSABIM(self, x, EZy):
        Xobs = self.model.train_inputs[0]
        Yobs = self.model.train_targets
        alpha = min(Yobs)
        y_warp = torch.sqrt(2*(Yobs-alpha))
        model_warp, likelihood_warp = gp.update_gp(Xobs, y_warp, self.model, training_iter=self.training_iter)
        _mean, _var = gp.predict(x, model_warp, likelihood_warp)
        likelihood_mean = alpha + 1/2 * _mean**2 + 1/2 * _var
        return likelihood_mean * self.prior.log_prob(x).exp() / EZy