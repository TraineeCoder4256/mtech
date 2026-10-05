"""Training data for the network: exact walks at random radii.

Used by: network.py (trains on it), checks/ball_metrics.py.
Uses: walk.py.

One row per walk: the radius R and what came out, (mu, s).  Radii are drawn
log-uniformly between r_min and r_max, so every factor of two in R gets the
same share of rows.  Every row is an independent walk, so there is no
"held-out cell" subtlety as in gmc2d: the validation rows are simply other
walks, and the network is also scored at radii it never saw exactly
(R is continuous).

Making the data is cheap -- a walk costs about R^2 scatters, all inside a
sphere with no geometry -- so a few million rows take seconds to minutes.
That is one of the practical gains of the ball formulation over gmc2d,
where data generation needed a sweep over cell shapes and entry states.

ENCODING (what the network actually sees), in one place:

    condition   c  = log R
    target 1    y0 = log(mu / (1 - mu))       exit cosine, 0 < mu < 1,
                                              stretched onto the real line
    target 2    y1 = log(s / R - 1)           path beyond the straight line
                                              (s > R always: it scattered)

Then each is standardised (minus mean, over standard deviation, from the
training rows).  `decode` undoes all of it.  Both targets are unbounded
and smooth, which suits a flow with a Gaussian base.
"""

from pathlib import Path

import numpy as np

from core import rng
from . import walk

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ball"
EPS = 1.0e-7


def make(n, r_min, r_max, seed=1):
    """n walks at log-uniform radii.  Returns dict R, mu, s, k."""
    state = rng.streams(np.uint64(seed), 0, n)
    gen = np.random.default_rng(seed)
    R = np.exp(gen.uniform(np.log(r_min), np.log(r_max), n))
    mu, s, k = walk.walk_many(R, state, np.arange(n))
    return {"R": R, "mu": mu, "s": s, "k": k}


def encode(R, mu, s):
    """Physical (R, mu, s) -> raw network inputs/targets (not standardised)."""
    mu = np.clip(mu, EPS, 1.0 - EPS)
    c = np.log(R)[:, None]
    y = np.stack([np.log(mu / (1.0 - mu)),
                  np.log(np.maximum(s / R - 1.0, EPS))], axis=1)
    return c, y


def decode(R, y):
    """Raw targets -> (mu, s).  The inverse of encode."""
    mu = 1.0 / (1.0 + np.exp(-y[:, 0]))
    s = R * (1.0 + np.exp(y[:, 1]))
    return mu, s


class Norm:
    """Standardisation: z = (x - mean) / std, per column."""

    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean), np.asarray(std)

    @classmethod
    def fit(cls, x):
        return cls(x.mean(axis=0), x.std(axis=0))

    def __call__(self, x):
        return (x - self.mean) / self.std

    def undo(self, z):
        return z * self.std + self.mean


def dataset(n_train, n_val, r_min, r_max, seed=1, cache=True):
    """Standardised train/validation arrays, cached in data/ball/."""
    path = DATA_DIR / f"walks_{n_train}_{n_val}_{r_min:g}_{r_max:g}_{seed}.npz"
    if cache and path.exists():
        d = dict(np.load(path))
    else:
        d = make(n_train + n_val, r_min, r_max, seed)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **d)
    c, y = encode(d["R"], d["mu"], d["s"])
    cn, yn = Norm.fit(c[:n_train]), Norm.fit(y[:n_train])
    f32 = lambda a: np.ascontiguousarray(a, np.float32)
    return {"c_train": f32(cn(c[:n_train])), "y_train": f32(yn(y[:n_train])),
            "c_val": f32(cn(c[n_train:])), "y_val": f32(yn(y[n_train:])),
            "cnorm": cn, "ynorm": yn, "mean_scatters": float(d["k"].mean()),
            "r_min": r_min, "r_max": r_max}
