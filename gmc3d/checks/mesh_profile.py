"""Does the 'profile' mesh rule put a ball's flux where exact walks do?

Used by: run.py (`python run.py meshcheck`).

The rule (ball/profile.py) is exact on average only if, over many balls,
the flux it scores at its drawn points has the same radial distribution as
the flux the real walks lay down.  This check measures that directly, one
ball at a time, with no geometry and no OpenMC:

    exact     fresh exact walks lay their decayed track exp(-kappa l) dl
              into N_SHELL radial shells, exactly (segment by segment)
    profile   other fresh exact walks (independent streams) give a path s;
              the rule draws its point from the table with that s, and the
              walk's whole flux F = (1 - exp(-kappa s)) / kappa goes into
              the shell of that point -- exactly what transport.py does

Each side's share per shell, with an error bar from G independent groups
of walks; shells with under MIN_SHARE of the flux on both sides are
skipped; z = (profile - exact) / combined error shell by shell, summed up
as chi2/dof (about 1 if they agree), max |z|, and the largest relative gap
among shells holding more than 1% of the flux.

The test points sit between the table's grid radii (so the blend of
neighbours is tested), span pure scatterers to strong absorbers
(kappa = 19 is the lattice3d absorber), and include R and kappa pairs of
the six test problems.
"""

import numpy as np
from numba import njit, prange

from ball import profile as BP
from core import rng

N_SHELL = 32
MIN_SHARE = 1.0e-4
POINTS = [  # (R in mfp, kappa = sigma_a / sigma_s)
    (3.0, 0.0), (3.0, 0.1),
    (5.5, 0.0), (5.5, 0.01), (5.5, 0.25),
    (12.0, 0.0), (12.0, 0.0333), (12.0, 2.0),
    (25.0, 0.0), (25.0, 0.0083), (25.0, 0.1),
    (1.5, 19.0), (8.0, 19.0),
]


@njit(cache=True, parallel=True)
def _groups(R, kappa, nb, per, groups, st_exact, st_prof, log_r, log1p_ell,
            f_unsc, cdf):
    ex = np.zeros((groups, nb))
    pr = np.zeros((groups, nb))
    for g in prange(groups):
        dummy = np.zeros(nb)
        for k in range(per):
            i = g * per + k
            BP.exact_walk(ex[g], kappa, R, nb, st_exact, i)
            s = BP.exact_walk(dummy, kappa, R, nb, st_prof, i)
            if kappa * s < 1.0e-12:
                F = s
            else:
                F = -np.expm1(-kappa * s) / kappa
            r = BP.draw_r(log_r, log1p_ell, f_unsc, cdf, R, kappa, s,
                          st_prof, i)
            pr[g, min(int(r / R * nb), nb - 1)] += F
    return ex, pr


def check_point(prof, R, kappa, n_walks=400_000, groups=80, seed=9001):
    per = n_walks // groups
    n = per * groups
    st_e = rng.streams(np.uint64(seed), 0, n)
    st_p = rng.streams(np.uint64(seed + 1), 0, n)
    ex, pr = _groups(float(R), float(kappa), N_SHELL, per, groups, st_e,
                     st_p, *prof.arrays())

    def share(a):
        tot = a.sum()
        g_share = a / a.sum(axis=1, keepdims=True)
        return a.sum(axis=0) / tot, g_share.std(axis=0, ddof=1) / \
            np.sqrt(groups)

    e, e_se = share(ex)
    p, p_se = share(pr)
    se = np.hypot(e_se, p_se)
    # shells holding less than MIN_SHARE of the flux on both sides are
    # skipped: in a strong absorber the outer shells get ~1e-9 of it, which
    # a finite sample of draws cannot resolve (zero hits, zero error bar)
    # and which no tally can see either
    ok = (se > 0) & (np.maximum(e, p) > MIN_SHARE)
    z = np.where(ok, (p - e) / np.where(ok, se, 1.0), 0.0)[ok]
    big = e > 0.01
    rel = float(np.max(np.abs(p[big] / e[big] - 1))) if big.any() else 0.0
    return {"R": R, "kappa": kappa, "shells": int(ok.sum()),
            "chi2_dof": float(np.mean(z ** 2)),
            "max_abs_z": float(np.max(np.abs(z))),
            "max_rel_gap": rel, "exact": e, "exact_se": e_se,
            "profile": p, "profile_se": p_se}


def run(prof=None, log=print):
    prof = prof or BP.get(log)
    rows = []
    log(f"profile rule vs fresh exact walks: {N_SHELL} radial shells, "
        f"400,000 walks a side in 80 groups (table: R {prof.r_min:g}-"
        f"{prof.r_max:g}, {prof.log1p_ell.shape[1]} path lengths, "
        f"{prof.cdf.shape[2] - 1} shells)")
    log("      R    kappa  shells  chi2/dof  max|z|  max rel gap "
        "(shells >1% of flux)")
    for R, k in POINTS:
        r = check_point(prof, R, k)
        rows.append(r)
        log(f"  {R:6.2f}  {k:7.4f}  {r['shells']:6d}  {r['chi2_dof']:8.3f}  "
            f"{r['max_abs_z']:6.2f}  {100 * r['max_rel_gap']:6.2f}%")
    return rows
