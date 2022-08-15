import gp
import torch

class KLdivergence:
    def __init__(self, prior, test_data, Z_true, true_function):
        self.prior = prior
        self.test_data = test_data
        self.Z_true = Z_true
        self.true_function = true_function
    
    def __call__(self, basq_model, EZy):
        KL = torch.zeros(len(self.test_data))
        q = torch.squeeze(
            self.prior.log_prob(self.test_data).exp() * self.true_function(self.test_data) / self.Z_true
        )
        p = basq_model.joint_posterior_WSABIM(self.test_data, EZy)
        KL[p > 0] = p[p > 0] * torch.log(p[p > 0] / q[p > 0])
        return torch.abs(torch.sum(KL) * len(self.test_data) / sum(p > 0))