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

Interface v2 (10 Nov 2026, agreed with Development), ANISOTROPIC balls.
It ADDS a second spec (ANISO_*) and leaves v1 as it is: balls in
isotropic materials keep calling a v1 model (sample), balls in
anisotropic ones call a v2 model (sample_aniso; core/transport.py).  So
INTERFACE_VERSION, the version a v1 model's manifest records, stays 1;
ANISO_INTERFACE_VERSION = 2 is what a v2 model records.  The walk starts
at the centre along a known direction u0 (just sampled from the law),
first flight along u0 conditioned to end inside, at least one scatter:

    input   R          ball radius, mfp, 1 <= R <= 30
            g1 .. g12  Legendre moments E[P_l(mu)] of the scattering law
                       as OpenMC samples it (ball/laws.moments); every
                       |g_l| <= 1, mean cosine g1 trained in [-0.5, 0.95];
                       12, not 8: peaked laws need them in small balls
                       (runs/2026-11-10/interface-v2/README.md); a model
                       may use fewer
    output  cos_n      cosine between the exit normal n and u0, -1..1
            mu         exit cosine to n, 0 < mu < 1 (as v1)
            phi        exit azimuth about n from the half-plane of
                       (n, u0) on u0's side, 0 <= phi <= pi (the driver
                       draws its sign)
            s          path length in the ball, mfp, s > R

The reference is walk.walk_many_aniso; training rows come from
data.make_aniso (one law per row from ball/laws.family; the test
problems' laws, laws.held_out, are kept out for judging).  An isotropic
law gives v1's (mu, s), but cos_n is not uniform (the first flight runs
along u0).

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


# ------------------------------------------------------------ interface v2
ANISO_INTERFACE_VERSION = 2
N_MOMENTS = 12
ANISO_INPUTS = {
    "R": (1.0, 30.0, True, True, "ball radius, mean free paths"),
    **{f"g{l}": (-1.0, 1.0, True, True, f"Legendre moment E[P_{l}(mu)] "
                 f"of the scattering law") for l in range(1, N_MOMENTS + 1)},
}
ANISO_OUTPUTS = {
    "cos_n": (-1.0, 1.0, True, True, "exit normal . first direction"),
    "mu": (0.0, 1.0, False, False, "exit cosine to the outward normal"),
    "phi": (0.0, np.pi, True, True, "exit azimuth about the normal"),
    "s": ("R", np.inf, False, False, "path length in the ball, mfp"),
}
ANISO_N_IN, ANISO_N_OUT = len(ANISO_INPUTS), len(ANISO_OUTPUTS)
G1_MIN, G1_MAX = -0.5, 0.95          # the trained range of the mean cosine


def check_aniso_outputs(R, cos_n, mu, phi, s):
    """Count draws outside the interface v2 ranges (all zero if valid)."""
    R, cos_n, mu, phi, s = (np.asarray(a, np.float64)
                            for a in (R, cos_n, mu, phi, s))
    return {
        "R_out_of_range": int(np.sum((R < R_MIN) | (R > R_MAX))),
        "cos_n_out_of_range": int(np.sum(~((cos_n >= -1) & (cos_n <= 1)))),
        "mu_out_of_range": int(np.sum(~((mu > 0.0) & (mu < 1.0)))),
        "phi_out_of_range": int(np.sum(~((phi >= 0) & (phi <= np.pi)))),
        "s_not_above_R": int(np.sum(~(s > R))),
        "non_finite": int(np.sum(~(np.isfinite(cos_n) & np.isfinite(mu)
                                   & np.isfinite(phi) & np.isfinite(s)))),
    }
