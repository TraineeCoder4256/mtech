"""Family 1: a normalizing flow (neural spline flow).

Moved here unchanged from ball/network.py (gmc3d v1.6.1), which now only
re-exports it.  Same network, same weights, same draws: shown before this
file was pushed (training with the same seed gives identical weights, and
the same weights give identical draws).

WHAT IT LEARNS.  p(mu, s | R): the joint distribution of the exit cosine
and the path length of a walk in a ball of radius R that scatters at least
once, in the encoded space of encoding.py.

HOW.  A neural spline flow, copied from gmc2d/generators/nflow.py and cut
down to 2 targets and 1 condition:

  * A chain of invertible steps bends Gaussian noise z into a target y.
    Each step ("coupling") leaves one of the two targets alone and bends
    the other with a monotone spline whose shape a small network picks
    from the untouched target and from R.  The next step swaps roles.
  * Every step can be undone exactly, so the probability the model gives a
    real walk is computable, and training simply maximises it (negative
    log-likelihood, in nats per sample).
  * Sampling runs the steps backwards from noise: ONE pass (nfe = 1).
    That is why it was chosen first: the cost per draw decides whether a
    ball step pays.

SIZE.  6 couplings of width 64, 16 spline bins, 146,554 weights: an
engineering default, far smaller than gmc2d's 1.7M because the target is
2-D.  It is the fairness reference (fairness.REFERENCE).

The coupling picks "the other target" as column 1 - bend, which assumes
X_DIM = 2.  A wider interface needs a mask instead (gmc2d's flow had six
targets and cycled through masks); that is a model-side change to make
when Innovation's interface grows.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import C_DIM, X_DIM
from .base import Family

# ---- shape of the flow ------------------------------------------------------
LAYERS, WIDTH, BINS, EMB = 6, 64, 16, 32
BOUND = 5.0                    # splines act on [-5, 5]; identity outside
MIN_WIDTH = MIN_HEIGHT = MIN_SLOPE = 1e-3


# ------------------------------------------------------------------ the curve
def _knots(raw, bound, minimum):
    """Unconstrained numbers -> K positive bin sizes summing to 2*bound, and
    the K+1 knot positions from -bound to bound."""
    k = raw.shape[-1]
    size = minimum + (1 - minimum * k) * torch.softmax(raw, dim=-1)
    edges = F.pad(torch.cumsum(size, dim=-1), (1, 0)) * 2 * bound - bound
    edges[..., 0], edges[..., -1] = -bound, bound
    return edges[..., 1:] - edges[..., :-1], edges


def rq_spline(x, uw, uh, ud, inverse=False, bound=BOUND):
    """Monotone rational-quadratic spline (Durkan et al. 2019), elementwise.

    x: (...)  uw, uh: (..., K)  ud: (..., K-1)  ->  (y, log|dy/dx|).
    inverse=True undoes the forward map.  Outside [-bound, bound] it is the
    identity.  (Copied from gmc2d/generators/nflow.py.)
    """
    inside = (x >= -bound) & (x <= bound)
    xc = x.clamp(-bound, bound)
    w, xk = _knots(uw, bound, MIN_WIDTH)
    h, yk = _knots(uh, bound, MIN_HEIGHT)
    shift = math.log(math.expm1(1 - MIN_SLOPE))
    d = F.pad(MIN_SLOPE + F.softplus(ud + shift), (1, 1), value=1.0)
    knots = yk if inverse else xk
    b = (xc[..., None] >= knots[..., 1:-1]).sum(-1, keepdim=True)
    g = lambda t: t.gather(-1, b)[..., 0]
    x0, y0, wb, hb, d0, d1 = g(xk), g(yk), g(w), g(h), g(d), g(d[..., 1:])
    s = hb / wb
    if not inverse:
        t = (xc - x0) / wb
    else:
        dy = xc - y0
        a = hb * (s - d0) + dy * (d1 + d0 - 2 * s)
        bq = hb * d0 - dy * (d1 + d0 - 2 * s)
        c = -s * dy
        disc = (bq * bq - 4 * a * c).clamp_min(0)
        t = (2 * c) / (-bq - torch.sqrt(disc))
    tt = t * (1 - t)
    den = s + (d1 + d0 - 2 * s) * tt
    ld = (torch.log(s * s * (d1 * t * t + 2 * s * tt + d0 * (1 - t) ** 2))
          - 2 * torch.log(den))
    if not inverse:
        out = y0 + hb * (s * t * t + d0 * tt) / den
    else:
        out, ld = x0 + t * wb, -ld
    return (torch.where(inside, out, x),
            torch.where(inside, ld, torch.zeros_like(ld)))


# ---------------------------------------------------------------- the network
class FiLMBlock(nn.Module):
    """Residual MLP block; the condition supplies a scale and a shift.
    (Copied from gmc2d/model.py.)"""

    def __init__(self, width, emb):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)
        self.film = nn.Linear(emb, 2 * width)

    def forward(self, h, emb):
        scale, shift = self.film(emb).chunk(2, dim=1)
        x = self.norm(h) * (1.0 + scale) + shift
        return h + self.fc2(F.silu(self.fc1(x)))


class Coupling(nn.Module):
    """One step: bend target `bend` with a spline chosen from the other
    target and the condition."""

    def __init__(self, bend, width, emb, bins, bound):
        super().__init__()
        self.bend, self.keep = bend, 1 - bend
        self.bins, self.bound = bins, bound
        self.proj_in = nn.Linear(1, width)
        self.block = FiLMBlock(width, emb)
        self.norm = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, 3 * bins - 1)
        nn.init.zeros_(self.proj_out.weight)     # start as the identity
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, emb, inverse=False):
        p = self.proj_out(self.norm(self.block(
            self.proj_in(x[:, self.keep:self.keep + 1]), emb)))
        k = self.bins
        yb, ld = rq_spline(x[:, self.bend], p[:, :k], p[:, k:2 * k],
                           p[:, 2 * k:], inverse=inverse, bound=self.bound)
        out = x.clone()
        out[:, self.bend] = yb
        return out, ld


class SplineFlow(nn.Module):
    """p(y | c) for 2 targets y and 1 condition c (both standardised)."""

    def __init__(self, layers=LAYERS, width=WIDTH, bins=BINS, emb=EMB,
                 bound=BOUND):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(C_DIM, emb), nn.SiLU(),
                                   nn.Linear(emb, emb), nn.SiLU())
        self.steps = nn.ModuleList(Coupling(i % 2, width, emb, bins, bound)
                                   for i in range(layers))

    def log_prob(self, y, c):
        emb, z, logdet = self.embed(c), y, 0.0
        for step in self.steps:
            z, ld = step(z, emb)
            logdet = logdet + ld
        return -0.5 * (z * z).sum(1) - math.log(2 * math.pi) + logdet

    def sample(self, z, c):
        emb = self.embed(c)
        for step in reversed(self.steps):
            z, _ = step(z, emb, inverse=True)
        return z


# ------------------------------------------------------------------ the family
class NFlow(Family):
    name = "nflow"
    nfe = 1
    loss_name = "NLL"
    CONFIG = {"layers": LAYERS, "width": WIDTH, "bins": BINS, "emb": EMB}

    @classmethod
    def build(cls, config):
        return SplineFlow(config["layers"], config["width"], config["bins"],
                          config["emb"])

    @staticmethod
    def loss(net, y, c):
        return -net.log_prob(y, c).mean()

    @classmethod
    def draw(cls, net, c, generator):
        z = torch.randn(c.shape[0], X_DIM, generator=generator,
                        device=c.device)
        return net.sample(z, c)
