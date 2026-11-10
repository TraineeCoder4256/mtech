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

import hashlib

from core import rng
from core import scattering as SC
from . import laws as LW
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


# ------------------------------------------------- interface v2 (anisotropic)
def make_aniso(n, r_min, r_max, seed=1, pool=20_000):
    """n anisotropic walks (interface v2, interface.ANISO_*): radii
    log-uniform as make(), and each row's own law drawn from a pool of
    `pool` laws of the training family (ball/laws.py).  Returns dict
    R, cos_n, mu, phi, s, k, law (the row's pool index) and g_pool
    (pool, N_MOMENTS): a row's moments are g_pool[law] (kept apart so 40M
    rows do not carry 12 floats each)."""
    laws, g = LW.family(pool, seed)
    ang = SC.pack(laws)
    gen = np.random.default_rng([seed, 0xDA7A])
    R = np.exp(gen.uniform(np.log(r_min), np.log(r_max), n))
    law = gen.integers(0, pool, n)
    state = rng.streams(np.uint64(seed), 0, n)
    cn, mu, phi, s, k = walk.walk_many_aniso(R, law.astype(np.int64), ang,
                                             state, np.arange(n))
    return {"g_pool": g, "R": R, "cos_n": cn, "mu": mu, "phi": phi, "s": s,
            "k": k, "law": law}


def data_hash(d):
    """sha256 over the arrays of a make()/make_aniso() dict, by key."""
    h = hashlib.sha256()
    for key in sorted(d):
        a = np.ascontiguousarray(d[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def dataset_aniso(n, r_min, r_max, seed=1, cache=True):
    """make_aniso rows, cached in data/ball/ (raw physical values: the
    encoding is the model side's, as for v1)."""
    path = DATA_DIR / f"aniso_walks_{n}_{r_min:g}_{r_max:g}_{seed}.npz"
    if cache and path.exists():
        return dict(np.load(path))
    d = make_aniso(n, r_min, r_max, seed)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **d)
    return d
