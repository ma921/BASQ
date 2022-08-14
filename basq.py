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
    ):
        self.n_rec = n_rec
        self.nys_ratio = nys_ratio
        self.training_iter = training_iter
        self.batch_size = batch_size
        self.quad_ratio = quad_ratio
        
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
        for _ in range(n_batch):
            s = time.time()
            self.run_basq()
            overhead = time.time() - s
            EZy, VarZy = self.quadrature()
            results.append([overhead, EZy, VarZy])
        return torch.tensor(results)