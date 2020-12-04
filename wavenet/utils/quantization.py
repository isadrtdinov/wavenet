import torch
from torch import nn


class MuLawQuantization(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([mu]), requires_grad=False)
        self.eps = 1e-4

    def forward(self, x):
        # waveform [-1.0, 1.0] to mu law [-1.0, 1.0]
        mu_x = torch.sign(x) * torch.log(1.0 + self.mu * torch.abs(x)) / \
               torch.log(1.0 + self.mu)
        return mu_x

    def inverse(self, mu_x):
        # mu law [-1.0, 1.0] to waveform [-1.0, 1.0]
        x = torch.sign(mu_x) / self.mu * \
            (torch.pow(1.0 + self.mu, torch.abs(mu_x)) - 1.0)
        return x

    def quantize(self, mu_x):
        # mu law [-1.0, 1.0] to quants [0, mu)
        quants = ((mu_x + 1.0) * (self.mu / 2) - self.eps).to(torch.long)
        return quants

    def dequantize(self, quants):
        # quants [0, mu) to mu law [-1.0, 1.0]
        mu_x = 2 * quants / self.mu - 1.0
        return mu_x
