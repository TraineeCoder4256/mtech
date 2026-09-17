"""Encoding the exit state for the network, and loading the dataset.

    target y (6):  cos(theta), sin(theta), Ox, Oy, Oz, log(s / chord)
    cond   c (5):  log W, log H, 2*xi - 1, Ox_in, Oy_in

matching the paper's conditioning on (W, H, xi_in, Omega_in).

Why each piece:

  theta = 2*pi*p / (2(W+H)).  The exit position p wraps around the cell
  perimeter, so p = 0 and p = perimeter are the same corner.  Fed raw, the
  network would have to learn a discontinuity; as a point on a circle there
  isn't one.  Note the perimeter is 2(W+H), not 4W -- cells are rectangles.

  chord = 2WH/(W+H) is the mean chord length of the cell (Dirac: 4*volume /
  surface).  Dividing s by it makes the path length dimensionless and
  O(1) whatever the cell's size or shape, which is what lets one network
  cover all of them.  For a square it reduces to W.

  log W and log H, not W and H, because the sizes span three decades.

Two rules that do real work:

  Rows with k == 0 are dropped.  Those particles crossed without scattering,
  so their exit is an exact function of their entry -- a Dirac delta a
  smooth flow cannot represent.  They are sampled analytically instead
  (see sampler.py).

  The train/validation split is BY ENTRY CONDITION, not by row.  Every
  condition has many sampled exits; splitting by row would put siblings of
  a training row into validation and report memorisation as generalisation.
"""

import json

import numpy as np


def mean_chord(W, H):
    """Dirac mean chord of a W x H cell: 4 * area / perimeter."""
    return 2.0 * W * H / (W + H)


def encode_targets(p, W, H, ox, oy, oz, s):
    theta = 2.0 * np.pi * p / (2.0 * (W + H))
    return np.stack([np.cos(theta), np.sin(theta), ox, oy, oz,
                     np.log(s / mean_chord(W, H))], axis=1).astype(np.float32)


def encode_conditions(W, H, xi, ox_in, oy_in):
    return np.stack([np.log(W), np.log(H), 2.0 * xi - 1.0, ox_in, oy_in],
                    axis=1).astype(np.float32)


class Normalizer:
    """Standardise each component, so the flow's Gaussian prior sees O(1)
    numbers everywhere.  Stats ship with the checkpoint and are inverted
    before decoding, so nothing is lost."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x):
        self.mean, self.std = x.mean(0), x.std(0) + 1e-8
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
    """npz from make_data.py -> normalised train/val tensors."""
    d = np.load(path)
    collided = d["k"] > 0
    take = {k: d[k][collided] for k in d.files}

    # rows sharing an entry condition are repeated samples of one setup
    key = np.stack([take["W"], take["H"], take["xi"],
                    take["ox_in"], take["oy_in"]], axis=1)
    _, cond_id = np.unique(key, axis=0, return_inverse=True)
    n_cond = int(cond_id.max()) + 1

    rng = np.random.default_rng(seed)
    val_cond = rng.random(n_cond) < val_frac      # held out by CONDITION
    is_val = val_cond[cond_id]                    # ... expanded to rows

    y = encode_targets(take["p"], take["W"], take["H"],
                       take["ox"], take["oy"], take["oz"], take["s"])
    c = encode_conditions(take["W"], take["H"], take["xi"],
                          take["ox_in"], take["oy_in"])

    tr, va = ~is_val, is_val
    ynorm, cnorm = Normalizer().fit(y[tr]), Normalizer().fit(c[tr])
    return {
        "y_train": ynorm.transform(y[tr]), "c_train": cnorm.transform(c[tr]),
        "y_val": ynorm.transform(y[va]), "c_val": cnorm.transform(c[va]),
        "ynorm": ynorm, "cnorm": cnorm,
        "info": {"rows": len(d["k"]), "collided": int(collided.sum()),
                 "conditions": n_cond, "train": int(tr.sum()),
                 "val": int(va.sum()), "val_conditions": int(val_cond.sum())},
    }


def save_norm(path, ynorm, cnorm, config):
    path.write_text(json.dumps({"y": ynorm.state(), "c": cnorm.state(),
                                "config": config}, indent=1))


def load_norm(path):
    d = json.loads(path.read_text())
    return Normalizer.from_state(d["y"]), Normalizer.from_state(d["c"]), d["config"]
