"""Sparse Structure Selection (Huang and Wang, ECCV 2018) at block level, the baseline of Table 2.

Each residual block gets one scaling factor, initialized to 1 and updated by the accelerated
proximal gradient step of the SSS paper (their Eqs. 10-12). A block is removed when its factor is
exactly zero.
"""

import torch
import torch.nn as nn

from lagdnet.utils import gated_blocks


class SSSFactor(nn.Module):
    """y = factor * x. The factor is stored as `logit`, so SSS networks save and load like the
    gated ones."""

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(1.0))
        self.temperature = 1.0

    def forward(self, x):
        return x * self.logit

    def value(self):
        return float(self.logit.item())


def use_sss_factors(model):
    """Replace the gate of every block with an SSS factor."""
    for b in gated_blocks(model):
        b.gate = SSSFactor()
        b.snapped = False
    return model


def soft_threshold(z, t):
    return torch.sign(z) * torch.clamp(z.abs() - t, min=0.0)


@torch.no_grad()
def apg_step(factors, velocities, lr, gamma, mu):
    """One update of the factors (SSS Eqs. 10-12), made after the weight step."""
    for f, v in zip(factors, velocities):
        g = f.grad if f.grad is not None else torch.zeros_like(f)
        s = soft_threshold(f - lr * g, lr * gamma)
        v_new = s - f + mu * v
        f.copy_(s + mu * v_new)
        v.copy_(v_new)


def remove_block(block):
    """Bypass a block whose factor is exactly zero. As trained, the block computes
    relu(0 * F(x) + S(x)) = relu(S(x)), so a projection shortcut keeps its ReLU. After an identity
    shortcut the ReLU is not needed, since the block input is already non-negative."""
    block.snapped = True
    if isinstance(block.shortcut, nn.Sequential):
        block.shortcut.append(nn.ReLU())
