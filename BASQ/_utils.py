import torch

class Utils:
    def __init__(self):
        self.eps = -torch.sqrt(torch.tensor(torch.finfo().max)).item()
        
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