"""Score a backend one ball at a time against the exact walk (SRS S1-VAL-4).

Used by: run.py (`python run.py ballcheck`), checks/validate.py.
Uses: ball/walk.py (the truth), any backend.

A full problem mixes many balls of many sizes, so a small error in the
backend can hide.  Here each radius is tested on its own: N exact walks
against N backend draws, at the same R.

What is compared, and why each matters to the answer of a real problem:

    mean mu            exit cosine: decides where the particle goes next
    mean s/R           path inside: the flux deposited is proportional to it
                       when absorption is weak
    KS mu, KS s        largest gap between the two cumulative distributions
                       (0 = identical; with N draws each, gaps up to about
                       1.36 sqrt(2/N) are noise at 95% confidence)
    weight factor      E[exp(-kappa s)] for kappa = sigma_a / sigma_s of
                       0.001, 0.01, 0.1: the share of the weight that
                       survives the ball.  Errors here go straight into
                       leakage and absorption.  The rare long walks matter
                       most at large kappa (cf. gmc2d's absorber tail).

Each difference is also given as z = difference / combined standard error.
|z| under about 3 means the backend cannot be told apart from the truth at
this N.
"""

import numpy as np

from ball import walk
from core import rng

RADII = (1.5, 3.0, 5.0, 10.0, 20.0, 29.0)
KAPPAS = (0.001, 0.01, 0.1)


def _ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _mean_cmp(x, y):
    mx, my = x.mean(), y.mean()
    se = np.sqrt(x.var() / x.size + y.var() / y.size)
    return {"exact": float(mx), "backend": float(my),
            "rel_err": float(my / mx - 1.0), "z": float((my - mx) / se)}


def score(backend, radii=RADII, n=200_000, seed=777):
    """One dict per radius with the comparisons above."""
    out = []
    for j, R in enumerate(radii):
        if not (backend.r_min <= R <= backend.r_max):
            continue
        mu0, s0, _ = walk.sample(R, n, seed=seed, first=2 * j * n)
        state = rng.streams(np.uint64(seed), (2 * j + 1) * n, n)
        mu1, s1 = backend.sample(np.full(n, float(R)), state, np.arange(n))
        row = {"R": R, "n": n,
               "mu": _mean_cmp(mu0, mu1),
               "s_over_R": _mean_cmp(s0 / R, s1 / R),
               "ks_mu": _ks(mu0, mu1), "ks_s": _ks(s0, s1),
               "ks_noise_95": 1.36 * np.sqrt(2.0 / n)}
        for k in KAPPAS:
            row[f"weight_k{k:g}"] = _mean_cmp(np.exp(-k * s0),
                                              np.exp(-k * s1))
        out.append(row)
    return out


def report(rows, title):
    lines = [title,
             "      R   mean mu err (z)    mean s err (z)    KS mu   KS s   "
             "noise  | weight err (z) at kappa " +
             "  ".join(f"{k:g}" for k in KAPPAS)]
    for r in rows:
        w = "  ".join(f"{100 * r[f'weight_k{k:g}']['rel_err']:+.3f}% "
                      f"({r[f'weight_k{k:g}']['z']:+.1f})" for k in KAPPAS)
        lines.append(
            f"  {r['R']:5.1f}   {100 * r['mu']['rel_err']:+.3f}% "
            f"({r['mu']['z']:+5.1f})   {100 * r['s_over_R']['rel_err']:+.3f}%"
            f" ({r['s_over_R']['z']:+5.1f})   {r['ks_mu']:.4f}  "
            f"{r['ks_s']:.4f}  {r['ks_noise_95']:.4f} | {w}")
    return "\n".join(lines)
