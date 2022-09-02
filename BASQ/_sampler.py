import torch
from ._gp import predict
from ._utils import Utils
from ._acquisition_function import SquareRootAcquisitionFunction

class PriorSampler:
    def __init__(self, prior, n_rec, nys_ratio):
        self.prior = prior
        self.n_rec = n_rec
        self.nys_ratio = nys_ratio

    def __call__(self, n_rec):
        pts_rec = self.prior.sample(sample_shape=torch.Size([n_rec]))
        pts_nys = pts_rec[:int(self.n_rec*self.nys_ratio)]
        w = torch.ones(n_rec) / n_rec
        return pts_nys, pts_rec, w
    
class UncertaintySampler(SquareRootAcquisitionFunction):
    def __init__(
        self,
        prior,
        model,
        n_rec,
        nys_ratio,
        ratio=0.5,
        n_gaussians=100,
        threshold=1e-5,
    ):
        super().__init__(prior, model, n_gaussians=n_gaussians, threshold=threshold)
        self.model = model
        self.ratio = ratio
        self.nys_ratio = nys_ratio
        self.utils = Utils()
        
    def pdf(self, X, var):
        if self.ratio == 0:
            return self.prior.log_prob(X).exp()
        elif self.ratio == 1:
            #return var * self.prior.log_prob(X).exp()
            return self.joint_pdf(X)
        else:
            #g_pdf = var * self.prior.log_prob(X).exp()
            g_pdf = self.joint_pdf(X)
            f_pdf = self.prior.log_prob(X).exp()
            return ((1-self.ratio) * f_pdf + self.ratio * g_pdf) / f_pdf
    
    def SIR(self, X, weights):
        n_weights = int(len(weights)*self.nys_ratio)
        draw = torch.multinomial(weights, n_weights)
        return X[draw]
    
    def __call__(self, n):
        if self.ratio == 0:
            pts_rec = self.prior.sample(torch.Size([n]))
        elif self.ratio == 1:
            pts_rec = self.sampling(n)
        else:
            pts_rec = torch.cat([
                self.sampling(int(self.ratio*n)),
                self.prior.sample(torch.Size([int((1-self.ratio)*n)])),
            ])
        
        mean, var = predict(pts_rec, self.model)
        w = torch.exp(torch.log(torch.abs(mean)) + self.prior.log_prob(pts_rec) - torch.nan_to_num(self.pdf(pts_rec, var)))
        w = torch.nan_to_num(w)
        if torch.sum(w) == 0:
            weights = torch.ones(len(w)) / len(w)
        else:
            weights = w / torch.sum(w) # importance sampling
        pts_nys = self.SIR(pts_rec, weights)
        return pts_nys, pts_rec, weights