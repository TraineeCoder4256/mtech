"""How physical quantities become numbers a network can learn, and back.

Used by: every family (through backend.py and pipeline.py), ball/data.py
(re-exports these names so older code keeps working).
Uses: nothing but NumPy.

The interface (ball/interface.py, owned by the Innovation side) says WHAT
goes in and out, in physical units: R in mean free paths in, (mu, s) out.
This file says HOW those are shown to a network.  That choice belongs to
the model side, because it changes how easy the distribution is to learn
but never what the transport receives: `decode` always hands back plain
(mu, s).

The encoding (moved here unchanged from ball/data.py, v1.6.1):

    condition   c  = log R
    target 1    y0 = log(mu / (1 - mu))       exit cosine, 0 < mu < 1,
                                              stretched onto the real line
    target 2    y1 = log(s / R - 1)           path beyond the straight line
                                              (s > R always: it scattered)

Then each column is standardised (minus its mean, over its standard
deviation, both taken from the training rows only).  Both targets end up
unbounded and smooth, which suits any family that starts from Gaussian
noise.  An analogy: the walk speaks in metres and degrees; the network is
happiest with numbers near zero.  This file is the interpreter, and every
family uses the same interpreter, so no family can win the comparison by
being handed an easier version of the problem.
"""

import hashlib

import numpy as np

from .. import interface as I

EPS = 1.0e-7

# sizes of the ENCODED condition and target, under the encoding below:
# one column per interface input (log R) and per output (two logits/logs).
# A different encoding may change these; families read them from here.
C_DIM, X_DIM = I.N_IN, I.N_OUT


def condition(R):
    """Physical R -> raw network condition (not standardised)."""
    return np.log(R)[:, None]


def encode(R, mu, s):
    """Physical (R, mu, s) -> raw network inputs/targets (not standardised)."""
    mu = np.clip(mu, EPS, 1.0 - EPS)
    c = condition(R)
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


def prepare(raw, n_train, r_min, r_max):
    """Raw walks (dict R, mu, s, k; train rows first) -> the standardised
    arrays every family trains on.  The norms are fitted on the training
    rows only, so the validation rows stay unseen."""
    c, y = encode(raw["R"], raw["mu"], raw["s"])
    cn, yn = Norm.fit(c[:n_train]), Norm.fit(y[:n_train])
    f32 = lambda a: np.ascontiguousarray(a, np.float32)
    return {"c_train": f32(cn(c[:n_train])), "y_train": f32(yn(y[:n_train])),
            "c_val": f32(cn(c[n_train:])), "y_val": f32(yn(y[n_train:])),
            "cnorm": cn, "ynorm": yn, "mean_scatters": float(raw["k"].mean()),
            "r_min": r_min, "r_max": r_max, "sha256": fingerprint(raw)}


def fingerprint(raw):
    """A short hash of the walks themselves (R, mu, s), so a manifest can
    say exactly which data a model saw, whatever file it came from."""
    h = hashlib.sha256()
    for k in ("R", "mu", "s"):
        h.update(np.ascontiguousarray(raw[k], np.float64).tobytes())
    return h.hexdigest()[:16]
