"""Dataset -> training tensors for the boundary model.

Encodings, applied before standardisation:

  target  y (6):  cos(2 pi p/4W), sin(2 pi p/4W), Ox, Oy, Oz, log(s/W)
  cond    c (5):  log W, log H, 2 xi - 1, Ox_in, Oy_in

Why each one:

  p is a coordinate around the cell perimeter, so it wraps: p = 0 and
  p = 4W are the same corner.  Feeding it raw would ask the network to
  learn a discontinuity, so it goes in as a point on a circle.

  s is positive and spans several decades, so it goes in as a log.
  Dividing by W first makes it dimensionless, which is what lets one
  network cover every cell size.

  H has its own slot even though the data is square (H = W).  The paper
  conditions on both independently and the code paths are all there, so
  rectangular data would need no architecture change.

Two preprocessing rules matter for correctness:

  * Rows with k == 0 are dropped.  Those are uncollided particles, which
    are a Dirac component (the exit is an exact function of the entry) that
    a smooth flow cannot represent.  They are sampled analytically instead;
    see gmc/sampler.py.

  * The train/validation split is BY ENTRY CONDITION, never by row.  Every
    condition has many sampled exits; splitting by row would put siblings
    of a training row in validation and report a validation loss that is
    partly memorisation.
"""

import json

import numpy as np


def encode_targets(p, W, ox, oy, oz, s):
    """Raw exit state -> 6-d target vector (un-normalised)."""
    theta = 2.0 * np.pi * p / (4.0 * W)
    return np.stack([np.cos(theta), np.sin(theta), ox, oy, oz,
                     np.log(s / W)], axis=1).astype(np.float32)


def encode_conditions(W, H, xi, oxi, oyi):
    """Raw entry state -> 5-d conditioning vector (un-normalised)."""
    return np.stack([np.log(W), np.log(H), 2.0 * xi - 1.0, oxi, oyi],
                    axis=1).astype(np.float32)


class Normalizer:
    """Per-component standardisation, applied after encoding.

    Every component is standardised, including the bounded (cos, sin) pair,
    so the flow's Gaussian prior sees O(1) coordinates in each dimension.
    The statistics ship with the checkpoint and are inverted before
    decoding, so nothing is lost.
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


def load_dataset(path, val_frac=0.05, seed=0):
    """Load an npz from scripts/make_data.py into training arrays."""
    d = np.load(path)
    W, y0 = d["W"], d["y0"]
    oxi, oyi = d["oxi"], d["oyi"]
    p, s, k = d["p"], d["s"], d["k"]
    oxo, oyo, ozo = d["oxo"], d["oyo"], d["ozo"]

    m = k > 0                                   # drop the uncollided Dirac
    n_total, n_coll = len(k), int(m.sum())

    # rows sharing (W, y0, oxi, oyi) are repeated samples of ONE condition
    cond_key = np.stack([W[m], y0[m], oxi[m], oyi[m]], axis=1)
    _, cond_id = np.unique(cond_key, axis=0, return_inverse=True)
    n_cond = int(cond_id.max()) + 1

    rng = np.random.default_rng(seed)
    val_conds = rng.random(n_cond) < val_frac
    is_val = val_conds[cond_id]

    y = encode_targets(p[m], W[m], oxo[m], oyo[m], ozo[m], s[m])
    c = encode_conditions(W[m], W[m], y0[m], oxi[m], oyi[m])   # H = W

    tr, va = ~is_val, is_val
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
