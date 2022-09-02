import copy
import torch
from ._gaussian_calc import GaussianCalc
from torch.distributions.multivariate_normal import MultivariateNormal

class SquareRootAcquisitionFunction(GaussianCalc):
    def __init__(self, prior, model, n_gaussians=100, threshold=1e-5):
        super().__init__(prior)
        self.n_gaussians = n_gaussians
        self.threshold = threshold   # the threshold to cut off the small weights
        self.update(model)
        
    def update(self, model):
        self.parameters_extraction(model)
        self.wA, self.wAA, self.mu_AA, self.sigma_AA = self.sparseGMM()
        self.d_AA = len(self.mu_AA)
        
    def sparseGMM(self):
        i, j = torch.where(self.woodbury_inv < 0)
        _w1_ = self.outputscale
        _w2_ = torch.abs((self.v**2) * self.woodbury_inv[i,j])
        _Z = _w1_ + torch.sum(_w2_)
        _w1, _w2 = _w1_/_Z, _w2_/_Z

        Winv = self.W.inverse()
        Sinv = self.prior.covariance_matrix.inverse()
        sigma2 = (2*Winv + Sinv).inverse()

        _idx = _w2.argsort(descending=True)[:self.n_gaussians]
        idx = _idx[_w2[_idx] > self.threshold]
        Xi = self.Xobs[i[idx]]
        Xj = self.Xobs[j[idx]]

        w2 = _w2[idx]
        mu2 = (sigma2 @ Winv @ (Xi + Xj).T).T + sigma2 @ Sinv @ self.prior.loc

        zA = _w1 + torch.sum(w2)
        w1, w2 = _w1/zA, w2/zA
        return w1, w2, mu2, sigma2
    
    def joint_pdf(self, x):
        d_x = len(x)

        # calculate the first term
        Npdfs_A = self.prior.log_prob(x).exp()
        first = self.wA * Npdfs_A

        # calulate the second term
        if len(self.wAA) == 0:
            return first
        else:
            x_AA = (torch.tile(self.mu_AA, (d_x,1,1)) - x.unsqueeze(1)).reshape(
                self.d_AA * d_x, self.n_dims
            )
            Npdfs_AA = MultivariateNormal(
                torch.zeros(self.n_dims),
                self.sigma_AA,
            ).log_prob(x_AA).exp().reshape(d_x, self.d_AA)

            f_AA = self.wAA.unsqueeze(0) * Npdfs_AA
            second = f_AA.sum(axis=1)
            return first + second
        
    def sampling(self, n):
        cntA = (n * self.wA).type(torch.int)
        samplesA = self.prior.sample(torch.Size([cntA]))

        if len(self.wAA) == 0:
            return samplesA
        else:
            cntAA = (n * self.wAA).type(torch.int)
            samplesAA = torch.cat([
                MultivariateNormal(
                    self.mu_AA[i],
                    self.sigma_AA,
                ).sample(torch.Size([cnt]))
                for i, cnt in enumerate(cntAA)
            ])
            return torch.cat([samplesA, samplesAA])