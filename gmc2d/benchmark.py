"""The one benchmark every model is scored on, and its frozen reference.

Every model in the comparison runs the same problem, at the same scales,
with the same particle counts and the same seeds, and is scored against the
same reference arrays.  Those arrays are computed once by make_reference.py
and committed under reference/, which is what makes the baseline FIXED: a
model scored today and one scored in six months, on any machine, are
compared against identical numbers.

Before this existed, evaluate.py re-ran Monte Carlo for every evaluation, so
each model got its own noisy reference -- and because timed_mc() returns the
field from its LAST timing repetition, which repetition that is depended on
how fast the machine was.  A model could look better or worse purely by the
luck of that draw.

The problem itself is defined in exactly one place, mc.lattice(); this file
only fixes the protocol around it.  openmc_lattice.py deliberately keeps its
own copy of the numbers (it imports nothing, so it stays an independent
check), and check_benchmark.py asserts that the two copies agree.
"""
import hashlib
import json
import pathlib

import numpy as np

import mc

# ---- the protocol ------------------------------------------------------
SCALES = [1, 4, 10, 20]      # cross-section multipliers; 1 = the published problem
N = 20000                    # particles per solve when scoring accuracy
SPEED_N = 4000               # particles per solve in the speed sweep
SEED = 1
GMC_SEEDS = (SEED + 7, SEED + 77)   # the two GMC realisations evaluate.py used

REFERENCE = pathlib.Path("reference")


def problem(scale=1):
    return mc.lattice(scale)


def definition(scale=1):
    """The parameters that identify the problem, as plain numbers.  Saved in
    every reference file and compared on load."""
    p = problem(scale)
    return {"scale": float(scale), "L": p["L"], "mesh": p["n"],
            "pitch": p["pitch"], "source": list(p["source"]),
            "source_cell": list(p["source_cell"]),
            "absorbers": [list(a) for a in mc.ABSORBERS],
            "sig_s_sha256": sha256(p["sig_s"]), "sig_a_sha256": sha256(p["sig_a"])}


def sha256(arr_or_path):
    if isinstance(arr_or_path, (str, pathlib.Path)):
        return hashlib.sha256(pathlib.Path(arr_or_path).read_bytes()).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(arr_or_path).tobytes()).hexdigest()


def reference_path(scale):
    return REFERENCE / f"lattice_x{scale:g}.npz"


def load_reference(scale):
    """The frozen reference at one scale, checked against the definition."""
    path = reference_path(scale)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run make_reference.py")
    d = dict(np.load(path, allow_pickle=False))
    stored = json.loads(str(d.pop("definition")))
    if stored != definition(scale):
        raise ValueError(f"{path} was built from a different problem "
                         f"definition than mc.lattice({scale}) -- rebuild it")
    return d


def coarsen(phi, per_cm):
    """Average a fine flux field onto the macro grid (same as solve.coarsen;
    repeated so this file does not import torch through solve.py)."""
    ny, nx = phi.shape[0] // per_cm, phi.shape[1] // per_cm
    return phi[:ny * per_cm, :nx * per_cm].reshape(
        ny, per_cm, nx, per_cm).mean(axis=(1, 3))


def agreement(a, se_a, b, se_b, min_rel=None):
    """Do two independent estimates of one field agree to within statistics?

    Per cell z = (a - b) / sqrt(se_a^2 + se_b^2).  If both estimate the same
    thing, z is roughly standard normal: chi^2 per cell ~ 1 and |z| > 3 in
    ~0.3% of cells.  min_rel restricts to cells both resolve better than
    that relative error (deep-shadow cells are dominated by their own
    noise and say nothing)."""
    se = np.sqrt(se_a ** 2 + se_b ** 2)
    ok = (se > 0) & (a > 0) & (b > 0)
    if min_rel is not None:
        ok &= (se_a < min_rel * a) & (se_b < min_rel * b)
    z = (a - b)[ok] / se[ok]
    rel = np.abs(a - b)[ok] / b[ok]
    return {"cells": int(ok.sum()), "mean_z": float(z.mean()),
            "chi2_per_cell": float((z ** 2).mean()),
            "frac_abs_z_gt_3": float((np.abs(z) > 3).mean()),
            "max_abs_z": float(np.abs(z).max()),
            "median_rel_diff": float(np.median(rel)),
            "max_rel_diff": float(rel.max())}
