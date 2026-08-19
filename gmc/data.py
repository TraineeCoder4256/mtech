"""Dataset -> training tensors for the boundary model.

Implements the required preprocessing of docs/boundary_model_training.md:

  * p_exit -> (cos, sin) of the normalised perimeter angle   (Sec 2.1)
  * s -> log(s / W)                                          (Sec 2.2)
  * W -> log W in the conditioning                           (Sec 2.3)
  * train/val split BY ENTRY CONDITION, never by row         (Sec 2.4)
  * uncollided branch (k == 0) removed from the training set (Sec 3.1);
    it is a Dirac component handled analytically at sampling time.

Encodings (before standardisation):

  target  y (6):  cos(2 pi p/4W), sin(2 pi p/4W), Ox, Oy, Oz, log(s/W)
  cond    c (5):  log W~, log H~, 2 xi - 1, Ox_in, Oy_in

H~ gets its own slot even though the current data is square (H = W): the
paper conditions on W~ and H~ independently, and this keeps the interface
ready for rectangular data without an architecture change.
"""

import json
import pathlib

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


def load_boundary_dataset(path, val_frac=0.05, seed=0, max_rows=None):
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

    y = encode_targets(p[m], W[m], oxo[m], oyo[m], ozo[m], s[m])
    c = encode_conditions(W[m], W[m], y0[m], oxi[m], oyi[m])  # H = W (square)

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
