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


# ------------------------------------------------------- interface v2 (m2.0)
# Anisotropic balls (ball/interface.py ANISO_*): 13 inputs, 4 outputs.  The
# same idea as above, one line per quantity:
#
#     condition   c  = (log R, g_1 .. g_12)     the moments are already
#                                               numbers near zero; they are
#                                               only standardised
#     target 1    y0 = logit((1 + cos_n) / 2)   -1..1 stretched onto the line
#     target 2    y1 = logit(mu)                as v1
#     target 3    y2 = logit(phi / pi)          0..pi stretched onto the line
#     target 4    y3 = log(s / R - 1)           as v1
#
# so v1's two targets are v2's middle and last ones, encoded identically.

C_DIM_ANISO, X_DIM_ANISO = I.ANISO_N_IN, I.ANISO_N_OUT


def _logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(y):
    return 1.0 / (1.0 + np.exp(-y))


def condition_aniso(R, g):
    """Physical R (n,) and moments g (n, 12) -> raw condition (n, 13)."""
    return np.concatenate([np.log(R)[:, None], np.asarray(g)], axis=1)


def encode_aniso(R, cos_n, mu, phi, s):
    """Physical outputs -> raw targets (n, 4) (not standardised)."""
    return np.stack([_logit(0.5 * (1.0 + cos_n)), _logit(mu),
                     _logit(phi / np.pi),
                     np.log(np.maximum(s / R - 1.0, EPS))], axis=1)


def decode_aniso(R, y):
    """Raw targets -> (cos_n, mu, phi, s).  The inverse of encode_aniso."""
    return (2.0 * _sigmoid(y[:, 0]) - 1.0, _sigmoid(y[:, 1]),
            np.pi * _sigmoid(y[:, 2]), R * (1.0 + np.exp(y[:, 3])))


def prepare_aniso(parts, n_train, r_min, r_max, hashes):
    """Anisotropic walks -> the standardised arrays every family trains on.

    parts   list of (raw, n_train_rows): each raw is a make_aniso() dict
            (R, cos_n, mu, phi, s, k, law, g_pool), its first n_train_rows
            rows go to training and the rest to validation.  Several parts
            let isotropic rows (g = 0) join the anisotropic ones.
    Built in float32 a slice at a time: 41M rows x 13 conditions is 2 GB.
    """
    ctr, ytr, cva, yva = [], [], [], []
    for raw, nt in parts:
        c = np.empty((raw["R"].size, C_DIM_ANISO), np.float32)
        c[:, 0] = np.log(raw["R"])
        c[:, 1:] = raw["g_pool"].astype(np.float32)[raw["law"]]
        y = np.empty((raw["R"].size, X_DIM_ANISO), np.float32)
        for lo in range(0, raw["R"].size, 4_000_000):
            sl = slice(lo, lo + 4_000_000)
            y[sl] = encode_aniso(raw["R"][sl], raw["cos_n"][sl],
                                 raw["mu"][sl], raw["phi"][sl], raw["s"][sl])
        ctr.append(c[:nt]); ytr.append(y[:nt])
        cva.append(c[nt:]); yva.append(y[nt:])
    c_tr, y_tr = np.concatenate(ctr), np.concatenate(ytr)
    c_va, y_va = np.concatenate(cva), np.concatenate(yva)
    del ctr, ytr, cva, yva
    norms = []
    for tr, va in ((c_tr, c_va), (y_tr, y_va)):
        mean = tr.mean(axis=0, dtype=np.float64)
        std = tr.std(axis=0, dtype=np.float64)
        for a in (tr, va):
            for lo in range(0, a.shape[0], 4_000_000):
                a[lo:lo + 4_000_000] -= mean.astype(np.float32)
                a[lo:lo + 4_000_000] /= std.astype(np.float32)
        norms.append(Norm(mean, std))
    k = np.concatenate([raw["k"] for raw, _ in parts])
    h = hashlib.sha256("".join(hashes).encode()).hexdigest()[:16]
    return {"c_train": c_tr, "y_train": y_tr, "c_val": c_va, "y_val": y_va,
            "cnorm": norms[0], "ynorm": norms[1],
            "mean_scatters": float(k.mean()), "r_min": r_min, "r_max": r_max,
            "sha256": h, "parts_sha256": list(hashes)}
