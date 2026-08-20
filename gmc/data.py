"""Dataset -> training tensors for the boundary model.

Implements the required preprocessing of docs/boundary_model_training.md:

  * p_exit -> (cos, sin) of the normalised perimeter angle   (Sec 2.1)
  * s -> a log path length; see PATH-LENGTH PARAMETERISATION  (Sec 2.2)
  * W -> log W in the conditioning                           (Sec 2.3)
  * train/val split BY ENTRY CONDITION, never by row         (Sec 2.4)
  * uncollided branch (k == 0) removed from the training set (Sec 3.1);
    it is a Dirac component handled analytically at sampling time.

Encodings (before standardisation):

  target  y (6):  cos(2 pi p/4W), sin(2 pi p/4W), Ox, Oy, Oz, u
  cond    c (5):  log W~, log H~, 2 xi - 1, Ox_in, Oy_in

H~ gets its own slot even though the current data is square (H = W): the
paper conditions on W~ and H~ independently, and this keeps the interface
ready for rectangular data without an architecture change.

PATH-LENGTH PARAMETERISATION (the sixth target, u)
--------------------------------------------------
"logW"    u = log(s / W~)          -- the original choice.
"detour"  u = log(s / s_min(p))    -- the default.

s_min(p) is the straight-line distance from the entry point to the exit
point implied by p, so s >= s_min is a hard physical constraint that
couples two of the six outputs.  Under "logW" the network has to learn
that constraint, and where it fails the sampler has to clamp; the clamp
puts a Dirac spike at s = s_min that the reference has no counterpart
for, and a classifier two-sample test picks it up immediately (it fired
on 4.22% of samples for boundary_v1).  Under "detour" the constraint is
absorbed into the decoder -- s = s_min(p) exp(u) satisfies it for any u
>= 0 -- so it cannot be violated by construction.  The variable is also
better conditioned: on the lattice dataset u has std 1.05 and spans
[0.0003, 15.8], against std 1.45 and [-16.6, 4.2] for log(s/W~).

Read u as a log "detour factor": 0 means the particle went straight from
entry to exit, 1 means it travelled e times further than it had to.
"""

import json
import pathlib

import numpy as np

from .geometry import s_min_of

S_PARAMS = ("detour", "logW")


def path_length_target(p, W, H, xi, s, s_param="detour"):
    """Path length -> the sixth target component u (see module docstring)."""
    if s_param == "logW":
        return np.log(s / W)
    if s_param == "detour":
        return np.log(s / np.maximum(s_min_of(p, W, H, xi), 1e-12))
    raise ValueError(f"unknown s_param {s_param!r}; expected one of {S_PARAMS}")


def path_length_decode(u, p, W, H, xi, s_param="detour"):
    """Inverse of path_length_target.

    Returns (s, violated), where ``violated`` flags samples the decoder had
    to push back up to the straight-line bound.  Under "detour" that can
    only happen for u < 0, which the true density essentially never visits
    (0.05% of training rows lie below u = 0.01), so the rate is a direct
    diagnostic of how hard the flow is pressing against the boundary.
    """
    smin = s_min_of(p, W, H, xi)
    if s_param == "logW":
        s = W * np.exp(u)
    elif s_param == "detour":
        s = np.maximum(smin, 1e-12) * np.exp(u)
    else:
        raise ValueError(f"unknown s_param {s_param!r}; "
                         f"expected one of {S_PARAMS}")
    violated = s < smin
    return np.maximum(s, smin), violated


def encode_targets(p, W, ox, oy, oz, s, H=None, xi=None, s_param="detour"):
    """Raw exit state -> 6-d target vector (un-normalised).

    H and xi are only consulted by the "detour" parameterisation, which
    needs the entry point to work out the straight-line bound.
    """
    theta = 2.0 * np.pi * p / (4.0 * W)
    if H is None:
        H = W
    u = path_length_target(p, W, H, xi, s, s_param)
    return np.stack([np.cos(theta), np.sin(theta), ox, oy, oz,
                     u], axis=1).astype(np.float32)


def encode_conditions(W, H, xi, oxi, oyi):
    """Raw entry state -> 5-d conditioning vector (un-normalised)."""
    return np.stack([np.log(W), np.log(H), 2.0 * xi - 1.0, oxi, oyi],
                    axis=1).astype(np.float32)


class Normalizer:
    """Per-component standardisation with save/load, applied after encoding.

    Everything is standardised uniformly -- including the bounded (cos, sin)
    components -- so the flow's Gaussian prior sees O(1) coordinates in every
    dimension; the stats are stored with the checkpoint and inverted before
    decoding, so no information is lost.
    """

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, d):
        return cls(np.asarray(d["mean"], np.float32),
                   np.asarray(d["std"], np.float32))


def load_boundary_dataset(path, val_frac=0.05, seed=0, max_rows=None,
                          s_param="detour"):
    """Load an npz produced by make_lattice_dataset.py into training arrays.

    Returns dict with train/val encoded+normalised tensors, the fitted
    normalizers, and bookkeeping (condition counts, W values).
    """
    d = np.load(path)
    W, y0 = d["W"], d["y0"]
    oxi, oyi = d["oxi"], d["oyi"]
    p, s, k = d["p"], d["s"], d["k"]
    oxo, oyo, ozo = d["oxo"], d["oyo"], d["ozo"]

    # --- drop the uncollided Dirac branch (Sec 3.1) ---------------------
    m = k > 0
    n_total, n_coll = len(k), int(m.sum())

    # --- condition IDs: rows sharing (W, y0, oxi, oyi) are one condition --
    cond_key = np.stack([W[m], y0[m], oxi[m], oyi[m]], axis=1)
    _, cond_id = np.unique(cond_key, axis=0, return_inverse=True)
    n_cond = int(cond_id.max()) + 1

    # --- split BY CONDITION (Sec 2.4) -----------------------------------
    rng = np.random.default_rng(seed)
    val_conds = rng.random(n_cond) < val_frac
    is_val = val_conds[cond_id]

    y = encode_targets(p[m], W[m], oxo[m], oyo[m], ozo[m], s[m],
                       H=W[m], xi=y0[m], s_param=s_param)   # H = W (square)
    c = encode_conditions(W[m], W[m], y0[m], oxi[m], oyi[m])

    tr, va = ~is_val, is_val
    if max_rows is not None and tr.sum() > max_rows:
        keep = np.zeros(tr.sum(), bool)
        keep[rng.choice(tr.sum(), max_rows, replace=False)] = True
        idx = np.flatnonzero(tr)
        tr = np.zeros_like(tr)
        tr[idx[keep]] = True

    ynorm = Normalizer().fit(y[tr])
    cnorm = Normalizer().fit(c[tr])

    return {
        "y_train": ynorm.transform(y[tr]), "c_train": cnorm.transform(c[tr]),
        "y_val": ynorm.transform(y[va]), "c_val": cnorm.transform(c[va]),
        "ynorm": ynorm, "cnorm": cnorm,
        "info": {
            "path": str(path), "n_total": n_total, "n_collided": n_coll,
            "n_conditions": n_cond, "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "n_val_conditions": int(val_conds.sum()),
            "s_param": s_param,
            "W_values": sorted(float(w) for w in np.unique(W)),
        },
    }


def save_normalizers(path, ynorm, cnorm, config):
    with open(path, "w") as f:
        json.dump({"ynorm": ynorm.state(), "cnorm": cnorm.state(),
                   "config": config}, f, indent=1)


def load_normalizers(path):
    with open(path) as f:
        d = json.load(f)
    return (Normalizer.from_state(d["ynorm"]),
            Normalizer.from_state(d["cnorm"]), d["config"])
