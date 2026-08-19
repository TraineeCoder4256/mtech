"""The velocity field v_theta(x, t, c).

Architecture (ours -- the paper's is in non-public supplementary material):

  * Gaussian Fourier features for the flow time t, fixed at init.
  * A small MLP embeds (t_features, c) into one conditioning vector.
  * The trunk is a stack of pre-LayerNorm residual MLP blocks; conditioning
    enters every block through FiLM (per-block scale and shift predicted
    from the embedding), so it modulates the whole depth rather than only
    the input layer.

Sized for the problem, not for show: the target is 6-d and the conditioning
5-d, so a ~0.6 M-parameter network is plenty and stays trainable on CPU.
"""

import math

import torch
import torch.nn as nn


class FourierTime(nn.Module):
    """Random Gaussian Fourier features of the scalar flow time."""

    def __init__(self, dim=64, scale=16.0):
        super().__init__()
        # fixed projection: registered as buffer so it ships with state_dict
        self.register_buffer("freq", torch.randn(dim // 2) * scale)

    def forward(self, t):
        # t: (n, 1) in [0, 1]
        ang = 2.0 * math.pi * t * self.freq[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class FiLMBlock(nn.Module):
    """Pre-LN residual MLP block, modulated by the conditioning embedding."""

    def __init__(self, width, emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.film = nn.Linear(emb_dim, width * 2)  # -> scale, shift

    def forward(self, h, emb):
        scale, shift = self.film(emb).chunk(2, dim=1)
        x = self.norm(h) * (1.0 + scale) + shift
        x = self.fc2(torch.nn.functional.silu(self.fc1(x)))
        return h + x


class VelocityField(nn.Module):
    """v_theta(x, t, c): (n,xdim)+(n,1)+(n,cdim) -> (n,xdim)."""

    def __init__(self, x_dim=6, c_dim=5, width=256, depth=5,
                 emb_dim=128, t_dim=64):
        super().__init__()
        self.x_dim, self.c_dim = x_dim, c_dim
        self.time = FourierTime(t_dim)
        self.embed = nn.Sequential(
            nn.Linear(t_dim + c_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
        )
        self.proj_in = nn.Linear(x_dim, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, emb_dim)
                                    for _ in range(depth))
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, x_dim)
        # zero-init the last layer: the flow starts as the identity map's
        # velocity (0), which stabilises early CFM training
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, t, c):
        emb = self.embed(torch.cat([self.time(t), c], dim=1))
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.proj_out(self.norm_out(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
