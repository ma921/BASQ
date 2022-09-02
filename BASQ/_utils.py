import torch
import warnings
from torch.distributions.multivariate_normal import MultivariateNormal

class Utils:
    def __init__(self, device):
        self.eps = -torch.sqrt(torch.tensor(torch.finfo().max)).item()
        self.gpu_lim = int(5e5)
        self.device = device
        
    def remove_anomalies(self, y):
        y[y.isnan()] = self.eps
        y[y.isinf()] = self.eps
        y[y < self.eps] = self.eps
        return y
    
    def remove_anomalies_uniform(self, X, uni_min, uni_max):
        logic = torch.sum(torch.stack([torch.logical_or(
            X[:,i] < uni_min[i], 
            X[:,i] > uni_max[i],
        ) for i in range(X.size(1))]), axis=0)
        return (logic == 0)
    
    def is_psd(self, mat):
        return bool((mat == mat.T).all() and (torch.eig(mat)[0][:,0]>=0).all())

    def safe_mvn_register(self, mu, cov):
        if self.is_psd(cov):
            return MultivariateNormal(mu, cov)
        else:
            warnings.warn("Estimated covariance matrix was not positive semi-definite. Conveting...")
            cov = torch.nan_to_num(cov)
            cov = torch.sqrt(cov * cov.T)
            if not self.is_psd(cov):
                cov = cov + torch.eye(cov.size(0)).to(self.device)
            return MultivariateNormal(mu, cov)
    
    def safe_mvn_prob(self, mu, cov, X):
        mvn = self.safe_mvn_register(mu, cov)
        if X.size(0) > self.gpu_lim:
            warnings.warn("The matrix size exceeds the GPU limit. Splitting.")
            n_split = torch.tensor(X.size(0) / self.gpu_lim).ceil().long()
            _X = torch.tensor_split(X, n_split)
            Npdfs = torch.cat(
                [
                    mvn.log_prob(_X[i]).exp()
                    for i in range(n_split)
                ]
            )
        else:
            Npdfs = mvn.log_prob(X).exp()
        return Npdfs
