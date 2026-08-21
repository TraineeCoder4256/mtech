"""The velocity field, and the loss that trains it.

Flow matching turns "sample from a hard distribution" into "solve an ODE".
The network answers one question: at point x, at time t along the path from
noise to data, for a cell described by c -- which way do I move?

Training: draw noise z and a real sample y, pick a time t, and put a point
on the straight line between them.  The velocity along that line is y - z,
so regress on it.  Each x_t lies on many such lines, so the mean-squared
error makes the network learn the AVERAGE velocity there -- and that field
is exactly the one whose flow carries noise onto data.

Three choices that are not obvious:
  * Fourier features for t: a raw scalar through a linear layer can only
    vary slowly with t, but the field changes character along the path.
  * FiLM instead of concatenating c: concatenated conditioning fades with
    depth, and here c is the whole point (W=0.1 and W=20 are different
    distributions), so it modulates every block.
  * Zero-init output layer: the flow starts as the identity, which is a
    stable place to begin.
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

    def __init__(self, width, emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.film = nn.Linear(emb_dim, width * 2)

    def forward(self, h, emb):
        scale, shift = self.film(emb).chunk(2, dim=1)
        x = self.norm(h) * (1.0 + scale) + shift        # condition steers
        x = self.fc2(torch.nn.functional.silu(self.fc1(x)))
        return h + x                                    # learn a correction


class VelocityField(nn.Module):
    """v(x, t, c): (n,6) + (n,1) + (n,5) -> (n,6)."""

    def __init__(self, x_dim=6, c_dim=5, width=256, depth=5,
                 emb_dim=128, t_dim=64):
        super().__init__()
        self.x_dim, self.c_dim = x_dim, c_dim
        self.time = FourierTime(t_dim)
        self.embed = nn.Sequential(
            nn.Linear(t_dim + c_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU())
        self.proj_in = nn.Linear(x_dim, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, emb_dim)
                                    for _ in range(depth))
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, x_dim)
        nn.init.zeros_(self.proj_out.weight)            # start at zero velocity
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
    x_t = (1.0 - t) * z + t * y
    return torch.mean((model(x_t, t, c) - (y - z)) ** 2)


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
            a.mul_(self.decay).add_(b, alpha=1.0 - self.decay)
        for a, b in zip(self.shadow.buffers(), model.buffers()):
            a.copy_(b)                                  # buffers are constants
