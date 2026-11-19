"""The block gate of LAGD-Net (Eq. 1): a learnable logit g applied as sigmoid(g / T)."""

import torch
import torch.nn as nn

TAU = 0.1  # threshold on the gate value at the end of the search


class BlockGate(nn.Module):
    """Scales the residual branch of a block by sigmoid(logit / temperature)."""

    def __init__(self, temperature=5.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(5.0))
        self.temperature = temperature

    def forward(self, x):
        return x * torch.sigmoid(self.logit / self.temperature)

    def value(self):
        return torch.sigmoid(self.logit / self.temperature).item()
