"""Point detectors: the flux at a point, by the next-event estimator (v1.6).

Used by: transport.py (scores them), openmc_import.py (reads and checks
them), problems/common.point_detector (how a problem asks for one).

THE ESTIMATOR.  A particle of weight w that leaves a point x isotropically
(born from an isotropic source, or just scattered: scattering is
isotropic in Stage 1) reaches a point P at distance d without colliding
with probability exp(-tau), tau the optical depth (sigma_t times length,
summed over the cells the straight line crosses).  Its contribution to
the scalar flux at P is

    w exp(-tau) / (4 pi d^2).

Summing that over every emission (source sites and scatters) gives the
flux at P exactly on average: the collided flux at P is the sum, over all
emissions, of the flux they send there uncollided.  A line that meets a
vacuum boundary first contributes 0 (a particle leaving there is gone).

INSIDE A BALL the scatters are not seen, only the ball's total flux F
(weighted track length).  But collisions happen at rate sigma_s per unit
of track, so the scatters inside a ball are distributed like sigma_s
times its flux.  The ball's flux positions are exactly what the 'profile'
mesh rule already draws (ball/profile.py): points X_k whose average,
weighted by the flux they carry, is right for EVERY region of the ball,
and so for every smooth function of position.  So a ball scores

    sigma_s (F / K) sum over k of exp(-tau(X_k -> P)) / (4 pi |X_k - P|^2)

with K independent profile points.  It loses the correlation between
where the walk scattered and where it left the ball, which changes only
the noise.

ONE CORRECTION to that.  "Scatters = sigma_s x flux" holds for a walk
left to itself, but a ball's walk is CONDITIONED on scattering at least
once inside (transport.py: if the first flight would leave the ball, the
particle just takes it, and no ball is used).  The straight flights it
excludes carry track but no scatters, so the conditioned walks hold more
scatters than sigma_s x their own flux: over both cases together,

    P_ball E[scatters | ball] = sigma_s (P_ball E[flux | ball]
                                         + P_straight F_line),

with P_straight = exp(-sigma_s d), P_ball = 1 - P_straight and F_line the
weighted track of a straight flight across the ball's radius d,
w (1 - exp(-sigma_a d)) / sigma_a.  So each ball also scores

    sigma_s (P_straight / P_ball) F_line x the kernel at a point of such a
    flight (distance t from the centre drawn with density exp(-sigma_a t)
    on [0, d], direction isotropic).

Measured without it (v1.6 detcheck, exact balls R* = 1, 2M histories):
every detector 2-6% low against plain Monte Carlo (z -4.4, -3.6, -2.3,
-1.1).  The term matters most for small balls: P_straight / P_ball is
0.58 at R = 1 and 0.05 at R = 3.

THE VARIANCE.  A scatter at distance d < some small rho from P scores
1 / d^2; with collisions spread over volume (density ~ d^2 dd) the mean
is finite but the variance is not: a known property of the estimator.
Two ways out, as in MCNP:
    * put the detector where nothing scatters near it (a void or a thin
      region): then d stays large and the variance is fine
    * give it an averaging radius r0 > 0: a contribution from closer than
      r0 scores the AVERAGE uncollided flux of a point source spread
      uniformly over a ball of radius r0 in P's own material,
          3 (1 - exp(-sigma_t r0)) / (4 pi sigma_t r0^3)
      (3 / (4 pi r0^2) in a void).  That is bounded, so the variance is
      finite, at a bias of order r0 (exact in the limit r0 -> 0).
"""

import numpy as np
from numba import njit

from . import geometry as G

TINY = 1.0e-8            # cm, as transport.TINY
MAX_CROSSINGS = 100_000  # per line, a guard
FOUR_PI = 4.0 * np.pi


@njit(cache=True)
def transmission(g, sig_t, x, y, z, px, py, pz, path):
    """(exp(-tau), d, sigma_t at P) along the straight line from (x, y, z)
    to P; exp(-tau) = 0 if the line meets a vacuum boundary first or
    leaves the geometry."""
    dx, dy, dz = px - x, py - y, pz - z
    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    if d <= 0.0:
        cell, mat = G.locate(g, px, py, pz)
        return 1.0, 0.0, sig_t[mat] if mat >= 0 else 0.0
    u, v, w = dx / d, dy / d, dz / d
    tau, left = 0.0, d
    cx, cy, cz = x, y, z
    st = 0.0
    for _ in range(MAX_CROSSINGS):
        cell, mat, _, t, surf, lvl, _ = G._descend(
            g, cx, cy, cz, u, v, w, False, True, path)
        if mat == G.LOST:
            return 0.0, d, 0.0
        st = sig_t[mat] if mat >= 0 else 0.0
        if t >= left:
            tau += st * left
            return np.exp(-tau), d, st
        tau += st * t
        if surf >= 0 and lvl == 0 and g.surf_bc[surf] == G.BC_VACUUM:
            return 0.0, d, st
        cx, cy, cz = cx + (t + TINY) * u, cy + (t + TINY) * v, \
            cz + (t + TINY) * w
        left -= t + TINY
        if left <= 0.0:
            return np.exp(-tau), d, st
    return 0.0, d, st


@njit(cache=True)
def score(row, D, g, sig_t, wgt, x, y, z, path):
    """Add the next-event contribution of an isotropic emission of weight
    wgt at (x, y, z) to every detector (entries D.base + j of row)."""
    for j in range(D.pos.shape[0]):
        tr, d, st = transmission(g, sig_t, x, y, z, D.pos[j, 0],
                                 D.pos[j, 1], D.pos[j, 2], path)
        if tr <= 0.0:
            continue
        r0 = D.r0[j]
        if d < r0:
            if st * r0 > 1e-9:
                k = 3.0 * (-np.expm1(-st * r0)) / (FOUR_PI * st * r0 ** 3)
            else:
                k = 3.0 / (FOUR_PI * r0 * r0)
            row[D.base + j] += wgt * k
        elif d > 0.0:
            row[D.base + j] += wgt * tr / (FOUR_PI * d * d)
