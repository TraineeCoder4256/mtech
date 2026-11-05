"""The model interface: what a learned ball backend takes in and gives out.

Owned by the Innovation side (core, walk, data, this file).  The model side
(ball/models/) reads the sizes and ranges from here and decides everything
about HOW inputs become outputs: encodings, layers, families, training.
Agreed 5 Nov 2026 (see CHANGELOG v1.6.2).

Any change to the lists below is an interface change: INTERFACE_VERSION goes
up by one, in its own commit tagged "interface vN", the training data is
regenerated, and the model side is told.  A trained model records the
interface version it was built for; the acceptance gate refuses a model
whose version differs from INTERFACE_VERSION.

Interface v1 (5 Nov 2026), one energy group, isotropic scattering:

    input   R    ball radius, in mean free paths, 1 <= R <= 30
    output  mu   exit cosine to the outward normal, 0 < mu < 1
            s    total path length in the ball, in mean free paths, s > R
                 (the walk scattered at least once, so s beats the radius)

All values are physical units; no encoding lives here.  The reference that
every model imitates is walk.walk_many (the exact walk); training rows come
from data.make.
"""

import numpy as np

INTERFACE_VERSION = 1

# name: (low, high, low_inclusive, high_inclusive, meaning)
INPUTS = {
    "R": (1.0, 30.0, True, True, "ball radius, mean free paths"),
}
OUTPUTS = {
    "mu": (0.0, 1.0, False, False, "exit cosine to the outward normal"),
    "s": ("R", np.inf, False, False, "path length in the ball, mfp"),
}

N_IN, N_OUT = len(INPUTS), len(OUTPUTS)
R_MIN, R_MAX = INPUTS["R"][0], INPUTS["R"][1]


def check_outputs(R, mu, s):
    """Count draws outside the interface ranges.  Returns a dict of counts
    (all zero for a valid batch); used by tests and the acceptance gate."""
    R, mu, s = (np.asarray(a, np.float64) for a in (R, mu, s))
    return {
        "R_out_of_range": int(np.sum((R < R_MIN) | (R > R_MAX))),
        "mu_out_of_range": int(np.sum(~((mu > 0.0) & (mu < 1.0)))),
        "s_not_above_R": int(np.sum(~(s > R))),
        "non_finite": int(np.sum(~(np.isfinite(mu) & np.isfinite(s)))),
    }
