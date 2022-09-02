import torch
from ._gp import predict
from ._rchq import recombination

class KernelQuadrature:
    def __init__(self, n_rec, n_nys, n_quad, batch_size, sampler, kernel, mean_predict):
        self.n_rec = n_rec
        self.n_nys = n_nys
        self.n_quad = n_quad
        self.batch_size = batch_size
        self.sampler = sampler
        self.kernel = kernel
        self.mean_predict = mean_predict

    def rchq(self, pts_nys, pts_rec, w_IS, batch_size, kernel):
        idx, w = recombination(
            pts_rec,
            pts_nys,
            batch_size,
            kernel,
            init_weights=w_IS,
        )
        x = pts_rec[idx]
        return x, w

    def quadrature(self):
        pts_nys, pts_rec, w_IS = self.sampler(self.n_rec)
        X, w = self.rchq(pts_nys, pts_rec, w_IS, self.batch_size, self.kernel)
        EZy = (w@self.mean_predict(X)).item()
        VarZy = (w@self.kernel(X,X)@w).item()
        print("E[Z|y]: "+str(EZy)+"  Var[Z|y]: "+str(VarZy))
        return EZy, VarZy

    def prior_max(self, mvn_max):
        pts_rec = mvn_max.sample(sample_shape=torch.Size([self.n_quad]))
        pts_nys = pts_rec[:self.n_nys]
        w_IS = torch.ones(self.n_quad) / self.n_quad

        X, w = self.rchq(pts_nys, pts_rec, w_IS, self.batch_size, self.kernel)
        EZy = (w@self.mean_predict(X)).item()
        VarZy = (w@self.kernel(X,X)@w).item()
        print("prior maximisation")
        print("E[Z|y]: "+str(EZy)+"  Var[Z|y]: "+str(VarZy))
        return EZy, VarZy
    
    def uniform_trans(self, model_IS, uni_sampler):
        pts_rec = uni_sampler(self.n_quad)
        pts_nys = pts_rec[:self.n_nys]
        w_IS = torch.ones(self.n_quad) / self.n_quad

        X, w = self.rchq(pts_nys, pts_rec, w_IS, self.batch_size, model_IS.covar_module.forward)
        mean, _ = predict(X, model_IS)
        EZy = (w@mean).item()
        VarZy = (w@model_IS.covar_module.forward(X,X)@w).item()
        print("uniform transformation")
        print("E[Z|y]: "+str(EZy)+"  Var[Z|y]: "+str(VarZy))
        return EZy, VarZy