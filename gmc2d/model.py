"""The velocity field, and the loss that trains it.

Flow matching turns "sample from a hard distribution" into "solve an ODE".
The network answers one question: at point x, at time t along the path from
noise to data, for a cell described by c -- which way do I move?

Training: draw noise z and a real sample y, pick a time t, put a point on
the straight line between them.  The velocity along that line is y - z, so
regress on it.  Each x_t lies on many such lines, so the mean-squared error
makes the network learn the AVERAGE velocity there -- and that field is
exactly the one whose flow carries noise onto data.

Three choices that are not obvious:
  * Fourier features for t: a raw scalar through a linear layer can only
    vary slowly with t, but the field changes character along the path.
  * FiLM rather than concatenating c: concatenated conditioning fades with
    depth, and c is the whole point here (a 0.1 mfp cell and a 20 mfp cell
    have completely different exit distributions).
  * Zero-init output layer: the flow starts as the identity map.
"""

import copy
import math

import torch
import torch.nn as nn


class FourierTime(nn.Module):
    """t -> sines and cosines of t at fixed random frequencies."""

    def __init__(self, dim=64, scale=16.0):
        super().__init__()
        self.register_buffer("freq", torch.randn(dim // 2) * scale)

    def forward(self, t):
        a = 2.0 * math.pi * t * self.freq[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class FiLMBlock(nn.Module):
    """Residual MLP block; the condition supplies a scale and a shift."""

    def __init__(self, width, emb):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)
        self.film = nn.Linear(emb, 2 * width)

    def forward(self, h, emb):
        scale, shift = self.film(emb).chunk(2, dim=1)
        x = self.norm(h) * (1.0 + scale) + shift
        return h + self.fc2(torch.nn.functional.silu(self.fc1(x)))


class VelocityField(nn.Module):
    """v(x, t, c): (n,6) + (n,1) + (n,5) -> (n,6)."""

    def __init__(self, x_dim=6, c_dim=5, width=256, depth=5, emb=128, t_dim=64):
        super().__init__()
        self.x_dim, self.c_dim = x_dim, c_dim
        self.time = FourierTime(t_dim)
        self.embed = nn.Sequential(nn.Linear(t_dim + c_dim, emb), nn.SiLU(),
                                   nn.Linear(emb, emb), nn.SiLU())
        self.proj_in = nn.Linear(x_dim, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, emb) for _ in range(depth))
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, x_dim)
        nn.init.zeros_(self.proj_out.weight)      # start at zero velocity
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, t, c):
        emb = self.embed(torch.cat([self.time(t), c], dim=1))
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.proj_out(self.norm_out(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def cfm_loss(model, y, c):
    """Conditional flow-matching loss on a batch of targets y, conditions c."""
    t = torch.rand(y.shape[0], 1, device=y.device)
    z = torch.randn_like(y)
    return torch.mean((model((1 - t) * z + t * y, t, c) - (y - z)) ** 2)


class EMA:
    """Moving average of the weights; sampling uses it, training does not."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.shadow.parameters(), model.parameters()):
            a.mul_(self.decay).add_(b, alpha=1 - self.decay)
        for a, b in zip(self.shadow.buffers(), model.buffers()):
            a.copy_(b)                            # buffers are constants
