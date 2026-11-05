"""Training data for the network: exact walks at random radii.

Used by: models/pipeline.py (every family trains on it),
checks/ball_metrics.py.
Uses: walk.py, models/encoding.py (the encoding).

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

ENCODING (what a network actually sees) is the model side's choice and
lives in models/encoding.py since model version m1.0.  This file makes the
walks; `dataset` hands them to that encoding.  The encoding names (encode,
decode, Norm, EPS) are re-exported here so older code keeps working.
"""

from pathlib import Path

import numpy as np

from core import rng
from . import walk
from .models.encoding import EPS, Norm, decode, encode, prepare  # noqa: F401

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ball"


def make(n, r_min, r_max, seed=1):
    """n walks at log-uniform radii.  Returns dict R, mu, s, k."""
    state = rng.streams(np.uint64(seed), 0, n)
    gen = np.random.default_rng(seed)
    R = np.exp(gen.uniform(np.log(r_min), np.log(r_max), n))
    mu, s, k = walk.walk_many(R, state, np.arange(n))
    return {"R": R, "mu": mu, "s": s, "k": k}


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
    return prepare(d, n_train, r_min, r_max)
