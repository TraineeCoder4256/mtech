"""A normalizing flow -- the second model family.

WHAT IT IS.  A chain of invertible steps that bends a plain Gaussian into
the exit-state distribution of one cell.  Every step can be run both ways
exactly, so the same network does two jobs:

  sampling   noise z  --(steps run backwards)-->  exit state y   (one pass)
  training   exit state y  --(steps run forwards)-->  z, and the exact
             probability the model gives y, from the change-of-variables
             rule:  log p(y|c) = log N(z) + sum of log |stretch| of each step

So it is trained by maximum likelihood: make the real Monte Carlo exits as
probable as possible.  No ODE, no noise schedule, no adversary.

HOW ONE STEP WORKS (a "coupling layer", Dinh et al. 2017).  Split the six
numbers into two halves.  Leave half A alone; bend each number in half B
with a monotone curve whose shape is chosen by a small network that looks at
half A and the cell condition c.  Because half A passes through untouched,
the network can be recomputed on the way back, and because each curve is
monotone it can be undone.  The next step swaps which half is bent, so after
a few steps every number has been bent depending on every other one.

THE CURVE (a rational-quadratic spline, Durkan et al. 2019, "Neural Spline
Flows").  On [-BOUND, BOUND] the curve is K pieces, each a ratio of two
quadratics, joined smoothly.  The network picks the K widths, K heights and
the slopes at the joins; outside the interval the curve is the straight
line y = x.  Both directions have closed forms (the inverse solves one
quadratic), so sampling is exact and needs no iteration.  Splines matter
here because the exit distribution is lumpy -- four faces, a pile-up of
short paths -- and a straight-line (affine) coupling bends too gently.

ONE PROBLEM PARTICULAR TO THIS DATA, and its fix.  Two groups of the six
targets are not free numbers: (cos, sin) of the exit angle always lie on a
circle, and the direction (ox, oy, oz) always lies on a sphere.  So the data
fill a thin 4-dimensional sheet inside 6-dimensional space.  A density on a
sheet of zero thickness is infinite, and a flow trained on it pours all its
effort into squeezing ever thinner instead of learning where on the sheet the
particles go.  Flow matching never notices, because it does not measure
density; a flow does.

The fix is RADIAL JITTER: during training only, multiply each circle point
and each direction by a random length 1 + JITTER * noise.  The circle
becomes a thin ring and the sphere a thin shell, both with honest thickness.
Nothing physical is blurred, because the jitter only changes LENGTHS, and
CellSampler already throws the lengths away: it reads the angle with
arctan2 and divides the direction by its norm.  So the decoded exit states
the model learns are exactly the ones in the data.  JITTER = 0.05 keeps the
lengths within about +-15%, far from zero.

COST.  One draw is ONE pass through LAYERS small networks; together they
hold about as many weights as cfm's single velocity network, so one flow
draw costs roughly one cfm network call, against cfm's ten.  nfe is 1.

FAIRNESS (generators/base.py).  The shared loop, the shared data and split,
the 3 h cap, cfm's own optimiser settings, and a parameter count within
+-20% of cfm's (this one is 1,716,648, cfm's is 1,676,678, with the settings below: 8 layers of
width 192, K = 8).  It even reuses cfm's residual block (model.FiLMBlock),
so the two families are built from the same bricks and differ in how the
bricks are used.

TUNING TRIALS (at most base.TUNING_RUNS = 4, recorded here):
  (filled in below once the trials are run)
"""
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import FiLMBlock
from generators.base import Generator
from generators.training import fit_loop

# ---- shape of the flow -----------------------------------------------------
LAYERS, WIDTH, BINS = 8, 192, 8        # coupling steps, block width, spline pieces
EMB = 128                              # size of the condition embedding
BOUND = 5.0                            # splines act on [-5, 5]; identity outside
JITTER = 0.05                          # radial jitter, training only (see above)
MIN_WIDTH = MIN_HEIGHT = MIN_SLOPE = 1e-3

# which of the six targets each step bends; the rest steer it.  Targets:
# 0,1 = cos, sin of the exit angle; 2,3,4 = ox, oy, oz; 5 = log(s / chord).
# Consecutive pairs cover all six, and the pattern mixes the three groups.
MASKS = ((3, 4, 5), (0, 1, 2), (0, 2, 4), (1, 3, 5))

# ---- training settings: cfm's, so neither family gets a kinder optimiser ---
STEPS, BATCH, LR = 20000, 4096, 2e-3
WARMUP, EMA_DECAY, VAL_EVERY = 500, 0.999, 500
SEED = 0


# ----------------------------------------------------------------- the curve
def _knots(raw, bound, minimum):
    """Unconstrained numbers -> K bin sizes that are positive and sum to 2*bound,
    and the K+1 knot positions from -bound to bound."""
    k = raw.shape[-1]
    size = minimum + (1 - minimum * k) * torch.softmax(raw, dim=-1)
    edges = F.pad(torch.cumsum(size, dim=-1), (1, 0)) * 2 * bound - bound
    edges[..., 0], edges[..., -1] = -bound, bound
    return edges[..., 1:] - edges[..., :-1], edges


def rq_spline(x, uw, uh, ud, inverse=False, bound=BOUND):
    """Monotone rational-quadratic spline, applied elementwise.

    x: (...)  uw, uh: (..., K)  ud: (..., K-1)  ->  (y, log|dy/dx|) at x.
    With inverse=True it undoes the forward map instead, and the log-slope
    returned is that of the inverse (the negative of the forward one).
    Outside [-bound, bound] the map is y = x with slope 1.
    """
    # Every value goes through the same arithmetic, clamped into the box so
    # nothing overflows; the ones that started outside are swapped back for
    # the identity at the end.  (Cheaper than picking out the inside ones.)
    inside = (x >= -bound) & (x <= bound)
    xc = x.clamp(-bound, bound)

    w, xk = _knots(uw, bound, MIN_WIDTH)
    h, yk = _knots(uh, bound, MIN_HEIGHT)
    # slopes at the K+1 knots; the two ends are 1 so the curve joins the
    # straight line outside smoothly.  The shift makes ud = 0 mean slope 1,
    # so a freshly built flow (zero outputs) is exactly the identity.
    shift = math.log(math.expm1(1 - MIN_SLOPE))
    d = F.pad(MIN_SLOPE + F.softplus(ud + shift), (1, 1), value=1.0)

    # which bin each value falls in, and that bin's corner, size and slopes
    knots = yk if inverse else xk
    b = (xc[..., None] >= knots[..., 1:-1]).sum(-1, keepdim=True)
    g = lambda t: t.gather(-1, b)[..., 0]
    x0, y0, wb, hb, d0, d1 = g(xk), g(yk), g(w), g(h), g(d), g(d[..., 1:])
    s = hb / wb                                  # the bin's average slope

    if not inverse:                              # t = how far across the bin
        t = (xc - x0) / wb
    else:                                        # solve a t^2 + b t + c = 0
        dy = xc - y0
        a = hb * (s - d0) + dy * (d1 + d0 - 2 * s)
        bq = hb * d0 - dy * (d1 + d0 - 2 * s)
        c = -s * dy
        disc = (bq * bq - 4 * a * c).clamp_min(0)
        t = (2 * c) / (-bq - torch.sqrt(disc))   # the root that lies in [0, 1]
    tt = t * (1 - t)
    den = s + (d1 + d0 - 2 * s) * tt
    ld = (torch.log(s * s * (d1 * t * t + 2 * s * tt + d0 * (1 - t) ** 2))
          - 2 * torch.log(den))                  # log of the forward slope
    if not inverse:
        out = y0 + hb * (s * t * t + d0 * tt) / den
    else:
        out, ld = x0 + t * wb, -ld
    return torch.where(inside, out, x), torch.where(inside, ld, torch.zeros_like(ld))
    x_in, uw, uh, ud = x[inside], uw[inside], uh[inside], ud[inside]

    w, xk = _knots(uw, bound, MIN_WIDTH)
    h, yk = _knots(uh, bound, MIN_HEIGHT)
    # slopes at the K+1 knots; the two ends are 1 so the curve joins the
    # straight line outside smoothly.  The shift makes ud = 0 mean slope 1,
    # so a freshly built flow (zero outputs) is exactly the identity.
    shift = math.log(math.expm1(1 - MIN_SLOPE))
    d = F.pad(MIN_SLOPE + F.softplus(ud + shift), (1, 1), value=1.0)

    # which bin each value falls in
    knots = yk if inverse else xk
    b = (x_in[..., None] >= knots[..., 1:-1]).sum(-1, keepdim=True)
    g = lambda t: t.gather(-1, b)[..., 0]
    x0, y0, wb, hb, d0, d1 = g(xk), g(yk), g(w), g(h), g(d), g(d[..., 1:])
    s = hb / wb                                  # the bin's average slope

    if not inverse:
        t = (x_in - x0) / wb
        tt = t * (1 - t)
        den = s + (d1 + d0 - 2 * s) * tt
        out = y0 + hb * (s * t * t + d0 * tt) / den
        slope = s * s * (d1 * t * t + 2 * s * tt + d0 * (1 - t) ** 2) / den ** 2
        ld = torch.log(slope)
    else:                                        # solve a t^2 + b t + c = 0
        dy = x_in - y0
        a = hb * (s - d0) + dy * (d1 + d0 - 2 * s)
        bq = hb * d0 - dy * (d1 + d0 - 2 * s)
        c = -s * dy
        disc = (bq * bq - 4 * a * c).clamp_min(0)
        t = (2 * c) / (-bq - torch.sqrt(disc))   # the root that lies in [0, 1]
        out = x0 + t * wb
        tt = t * (1 - t)
        den = s + (d1 + d0 - 2 * s) * tt
        slope = s * s * (d1 * t * t + 2 * s * tt + d0 * (1 - t) ** 2) / den ** 2
        ld = -torch.log(slope)
    y[inside], logdet[inside] = out, ld
    return y, logdet


# --------------------------------------------------------------- the network
class Coupling(nn.Module):
    """One step: bend the targets in `bend` with splines chosen from the rest."""

    def __init__(self, bend, x_dim, width, emb, bins, bound):
        super().__init__()
        self.bound = bound
        self.register_buffer("bend", torch.tensor(bend))
        self.register_buffer("keep", torch.tensor([i for i in range(x_dim) if i not in bend]))
        self.bins = bins
        self.proj_in = nn.Linear(x_dim - len(bend), width)
        self.block = FiLMBlock(width, emb)
        self.norm = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, len(bend) * (3 * bins - 1))
        nn.init.zeros_(self.proj_out.weight)     # start as the identity map
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, emb, inverse=False):
        p = self.proj_out(self.norm(self.block(self.proj_in(x[:, self.keep]), emb)))
        p = p.view(x.shape[0], len(self.bend), 3 * self.bins - 1)
        k = self.bins
        yb, ld = rq_spline(x[:, self.bend], p[..., :k], p[..., k:2 * k],
                           p[..., 2 * k:], inverse=inverse, bound=self.bound)
        out = x.clone()
        out[:, self.bend] = yb
        return out, ld.sum(1)


class SplineFlow(nn.Module):
    """p(y | c) for (n, 6) targets y and (n, 5) conditions c."""

    def __init__(self, x_dim=6, c_dim=5, layers=LAYERS, width=WIDTH, bins=BINS,
                 emb=EMB, bound=BOUND):
        super().__init__()
        self.x_dim, self.c_dim = x_dim, c_dim
        # the same condition embedding as cfm's, minus the time input
        self.embed = nn.Sequential(nn.Linear(c_dim, emb), nn.SiLU(),
                                   nn.Linear(emb, emb), nn.SiLU())
        self.steps = nn.ModuleList(Coupling(MASKS[i % len(MASKS)], x_dim, width, emb, bins, bound)
                                   for i in range(layers))

    def log_prob(self, y, c):
        """Exact log-density of y: run the steps forwards to noise z."""
        emb, z, logdet = self.embed(c), y, 0.0
        for step in self.steps:
            z, ld = step(z, emb)
            logdet = logdet + ld
        base = -0.5 * (z * z).sum(1) - 0.5 * self.x_dim * math.log(2 * math.pi)
        return base + logdet

    def sample(self, z, c):
        """Noise -> exit states: run the steps backwards, last step first."""
        emb = self.embed(c)
        for step in reversed(self.steps):
            z, _ = step(z, emb, inverse=True)
        return z


# ----------------------------------------------------------------- the loss
def radial_jitter(y, ynorm, jitter=JITTER):
    """Give the circle and the sphere a little thickness (see the docstring).
    y is standardised, so undo the standardisation, rescale the lengths, redo
    it.  The random numbers come from torch's global generator, as cfm_loss's."""
    mean = torch.as_tensor(ynorm.mean, dtype=y.dtype, device=y.device)
    std = torch.as_tensor(ynorm.std, dtype=y.dtype, device=y.device)
    raw = y * std + mean
    r = 1 + jitter * torch.randn(y.shape[0], 2, dtype=y.dtype, device=y.device)
    raw = torch.cat([raw[:, 0:2] * r[:, :1], raw[:, 2:5] * r[:, 1:], raw[:, 5:]], 1)
    return (raw - mean) / std


def nll_loss(net, y, c, ynorm):
    """Average negative log-likelihood per sample, in nats."""
    return -net.log_prob(radial_jitter(y, ynorm), c).mean()


# ----------------------------------------------------------------- the family
class NFlow(Generator):
    name = "nflow"
    nfe = 1                     # one pass through the chain of coupling steps

    def __init__(self, net=None, device="cpu", seed=SEED):
        self.net = None if net is None else net.to(device).eval()
        self.device, self.seed = device, seed
        self.config = {}

    def fit(self, data, out_dir, time_cap=float("inf")):
        x_dim, c_dim = data["y_train"].shape[1], data["c_train"].shape[1]
        ynorm = data["ynorm"]
        print(f"nflow  {LAYERS} layers  width {WIDTH}  bins {BINS}  jitter {JITTER}")
        self.net, facts = fit_loop(
            lambda: SplineFlow(x_dim, c_dim, LAYERS, WIDTH, BINS, EMB, BOUND),
            lambda net, y, c: nll_loss(net, y, c, ynorm),
            data, out_dir, time_cap, steps=STEPS, batch=BATCH, lr=LR,
            warmup=WARMUP, ema_decay=EMA_DECAY, val_every=VAL_EVERY,
            seed=self.seed, device=self.device, loss_name="NLL (nats)")
        self.config = {"layers": LAYERS, "width": WIDTH, "bins": BINS, "emb": EMB,
                       "bound": BOUND, "jitter": JITTER, "x_dim": int(x_dim),
                       "c_dim": int(c_dim), "steps": STEPS, "batch": BATCH,
                       "lr": LR, "seed": self.seed}
        return facts

    @torch.no_grad()
    def sample(self, c, generator, cond=None):
        z = torch.randn(c.shape[0], self.net.x_dim, generator=generator).to(self.device)
        return self.net.sample(z, c.to(self.device)).cpu()

    def save(self, out_dir):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"ema": {k: v.cpu() for k, v in self.net.state_dict().items()},
                    "config": self.config}, out_dir / "model.pt")

    @classmethod
    def load(cls, out_dir, device="cpu"):
        st = torch.load(pathlib.Path(out_dir) / "model.pt", map_location="cpu",
                        weights_only=True)
        c = st["config"]
        net = SplineFlow(c["x_dim"], c["c_dim"], c["layers"], c["width"],
                         c["bins"], c["emb"], c["bound"])
        net.load_state_dict(st["ema"])
        g = cls(net, device)
        g.config = c
        return g

    def describe(self):
        d = {"name": self.name, "nfe": self.nfe, "train_config": self.config}
        if self.net is not None:
            d["params"] = int(sum(p.numel() for p in self.net.parameters()))
            d["layers"] = len(self.net.steps)
        return d
