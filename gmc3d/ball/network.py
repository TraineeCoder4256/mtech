"""Backend 3 of 3: a trained generative network (a normalizing flow).

Used by: core/transport.py (as `backend`), checks/ball_metrics.py, run.py
(`python run.py train`).
Uses: data.py (training data and the encoding).

WHAT IT LEARNS.  p(mu, s | R): the joint distribution of the exit cosine
and the path length of a walk in a ball of radius R that scatters at least
once.  One input, two outputs -- the whole problem the ball formulation
leaves for a model (gmc2d's model had 5 inputs and 6 outputs).

WHICH MODEL.  A neural spline flow, copied from gmc2d/generators/nflow.py
and cut down to 2 targets and 1 condition (gmc2d is frozen, so the code is
copied, not imported).  The flow was chosen because it draws a sample in
ONE pass (no ODE steps, unlike flow matching), and the cost per draw is
what decides whether a ball step pays.  How it works, briefly:

  * A chain of invertible steps bends Gaussian noise z into a target y.
    Each step ("coupling") leaves one of the two targets alone and bends
    the other with a monotone spline whose shape a small network picks
    from the untouched target and from R.  The next step swaps roles.
  * Every step can be undone exactly, so the probability the model gives a
    real walk is computable, and training simply maximises it (negative
    log-likelihood, in nats per sample).
  * Sampling runs the steps backwards from noise: one pass.

The size here (6 steps of width 64, ~150k weights) is an engineering
default, far smaller than gmc2d's 1.7M: the target is 2-D, and research
item R2 (does a network beat the table, and at what size) is for Phani to
settle with the measurements in checks/.

TRAINING follows gmc2d's shared loop (generators/training.py): AdamW,
warm-up then cosine decay, gradient clipping, an exponential moving
average (EMA) of the weights that is what gets saved and sampled.
"""

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D

MODEL_DIR = D.DATA_DIR.parent / "models"

# ---- shape of the flow ------------------------------------------------------
LAYERS, WIDTH, BINS, EMB = 6, 64, 16, 32
BOUND = 5.0                    # splines act on [-5, 5]; identity outside
MIN_WIDTH = MIN_HEIGHT = MIN_SLOPE = 1e-3

# ---- training ---------------------------------------------------------------
R_MIN, R_MAX = 1.0, 30.0
N_TRAIN, N_VAL = 4_000_000, 500_000
STEPS, BATCH, LR = 8000, 4096, 2e-3
WARMUP, EMA_DECAY, VAL_EVERY = 300, 0.999, 500
SEED = 0


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
        self.embed = nn.Sequential(nn.Linear(1, emb), nn.SiLU(),
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


class EMA:
    """Moving average of the weights; sampling uses it, training does not."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.shadow.parameters(), model.parameters()):
            a.mul_(self.decay).add_(b, alpha=1 - self.decay)


# ------------------------------------------------------------------- training
def train(out_dir=None, steps=STEPS, seed=SEED, r_min=R_MIN, r_max=R_MAX,
          n_train=N_TRAIN, n_val=N_VAL, log=print):
    """Make (or reuse) the data, train, save.  Returns the Network."""
    out_dir = Path(out_dir or MODEL_DIR / f"flow_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = D.dataset(n_train, n_val, r_min, r_max, seed=1)
    log(f"data   {n_train:,} + {n_val:,} walks, R in [{r_min:g}, {r_max:g}],"
        f" mean {d['mean_scatters']:.1f} scatters each, "
        f"{time.time() - t0:.1f} s")

    torch.manual_seed(seed)
    y, c = torch.from_numpy(d["y_train"]), torch.from_numpy(d["c_train"])
    yv, cv = torch.from_numpy(d["y_val"]), torch.from_numpy(d["c_val"])
    net = SplineFlow()
    n_params = sum(p.numel() for p in net.parameters())
    log(f"model  {LAYERS} couplings, width {WIDTH}, {BINS} spline bins, "
        f"{n_params:,} weights")
    ema = EMA(net, EMA_DECAY)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    gen = torch.Generator().manual_seed(seed)
    hist, smooth, t1 = [], None, time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, y.shape[0], (BATCH,), generator=gen)
        loss = -net.log_prob(y[idx], c[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(net)
        smooth = loss.item() if smooth is None else \
            0.98 * smooth + 0.02 * loss.item()
        if step % VAL_EVERY == 0 or step == steps:
            with torch.no_grad():
                v = torch.randint(0, yv.shape[0], (65536,), generator=gen)
                vloss = -ema.shadow.log_prob(yv[v], cv[v]).mean().item()
            hist.append((step, smooth, vloss))
            log(f"step {step:6d}  train NLL {smooth:.4f}  val NLL (EMA) "
                f"{vloss:.4f}  {time.time() - t1:.0f} s")
    info = {"steps": steps, "seed": seed, "params": int(n_params),
            "train_seconds": time.time() - t1, "history": hist,
            "layers": LAYERS, "width": WIDTH, "bins": BINS, "emb": EMB,
            "r_min": r_min, "r_max": r_max, "n_train": n_train}
    torch.save({"ema": ema.shadow.state_dict(), "info": info,
                "cnorm": (d["cnorm"].mean, d["cnorm"].std),
                "ynorm": (d["ynorm"].mean, d["ynorm"].std)},
               out_dir / "model.pt")
    return Network.load(out_dir)


# -------------------------------------------------------------- the backend
class Network:
    """The backend.  `device` is where the network runs: "cpu" (default)
    or "cuda" for a GPU.  The transport loop stays on the CPU either way;
    each backend call sends the batch of radii over and gets (mu, s) back.
    """
    name = "network"

    def __init__(self, net, cnorm, ynorm, info, seed=12345, device="cpu"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.cnorm, self.ynorm, self.info = cnorm, ynorm, info
        self.r_min, self.r_max = info["r_min"], info["r_max"]
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.draws = 0

    @classmethod
    def load(cls, out_dir, seed=12345, device="cpu"):
        st = torch.load(Path(out_dir) / "model.pt", map_location="cpu",
                        weights_only=False)
        i = st["info"]
        net = SplineFlow(i["layers"], i["width"], i["bins"], i["emb"])
        net.load_state_dict(st["ema"])
        return cls(net, D.Norm(*st["cnorm"]), D.Norm(*st["ynorm"]), i, seed,
                   device)

    @torch.no_grad()
    def sample(self, R, seeds=None, idx=None):
        """(mu, s) for each R.  Uses its own seeded torch generator (the
        particle streams are not needed), so a run is still reproducible
        on a given device (CPU and GPU draw different noise)."""
        R = np.asarray(R, np.float64)
        c = torch.from_numpy(self.cnorm(np.log(R)[:, None]).astype(
            np.float32)).to(self.device)
        z = torch.randn(R.size, 2, generator=self.gen, device=self.device)
        y = self.net.sample(z, c).cpu().numpy().astype(np.float64)
        self.draws += R.size
        return D.decode(R, self.ynorm.undo(y))

    def describe(self):
        return {"name": self.name, **{k: v for k, v in self.info.items()
                                      if k != "history"}}
