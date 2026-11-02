"""Where inside a ball its flux lies: the radial profile for mesh tallies.

Used by: core/transport.py (the 'profile' mesh rule draws from it),
checks/mesh_profile.py (checks it against fresh exact walks).
Uses: the same walk as walk.py, but one that records WHERE it went.

THE PROBLEM.  A ball step reports only the total path s of the walk inside,
so cell, material and total tallies get it exactly, but a mesh voxel does
not: the walk's track is spread through the ball, and a ball can cover many
voxels.  Putting all of it in the centre voxel smears the flux map
(gmc3d's old 'centre' rule, chi2/dof 538 on the slab and 73,816 on the
sphere against OpenMC); shrinking the ball to one voxel ('cap') is exact
but throws most of the ball's saving away.

THE FIX: score the ball's whole flux at ONE random point of its walk.
Along the walk the weight decays as exp(-kappa l), kappa = sigma_a /
sigma_s (the walk is in mean free paths of scattering), so the flux the
ball deposits is

    F = integral from 0 to s of exp(-kappa l) dl = (1 - exp(-kappa s)) / kappa.

Pick a path length T in [0, s] with density exp(-kappa T) / F -- that is,
the point of the walk weighted by how much flux it laid down there -- and
put all of F in the voxel holding the walk's position X(T).  Averaged over
T, that puts exp(-kappa l) dl of flux at X(l) for every l: exactly the
walk's own deposit.  So every voxel is right on average.

We do not have the walk, only (mu, s).  But we do not need ITS position
X(T): a draw from the distribution of positions of ALL walks that are still
inside the ball at path length T will do, because the average over walks
then gives

    E[flux in V] = integral exp(-kappa l) P(still inside at l) P(X(l) in V |
                   inside at l) dl,

which is the true expected flux in V.  (The proof: the density of T given s
is kappa exp(-kappa T) / (1 - exp(-kappa s)) on [0, s]; multiply by F and
it becomes exp(-kappa T); integrate over s > T and P(inside at T) appears.)

So the table holds ONE thing, with no absorption in it at all:

    Q(r | l, R)   where a walk in a ball of radius R that is still inside
                  at path length l is, as a distribution of radius r

and absorption enters only through T, drawn exactly from the formula
above with the ball's own s.  A first version tabulated the flux profile
on a grid of kappa instead and blended neighbouring kappas; it was wrong in
absorbers (chi2/dof up to 900 against exact walks at kappa = 19), because
the profile changes fast with kappa.  This version has no kappa grid.

What is not exact, and is measured by checks/mesh_profile.py:

    * l and R between grid points: the draw blends the two neighbouring
      grid points (chosen at random in proportion to closeness);
    * inside a radial shell the radius is drawn uniformly;
    * the table is a finite sample of walks (N_PER per radius).

One part is exact by construction.  Until its first scatter a walk goes
straight out from the centre, so at path length l it sits at radius l
exactly.  The table stores the fraction of walks still unscattered at each
l, and those are placed at r = l itself; only the scattered part uses the
shells.  And since a walk that has gone a path l is within l of the centre,
the shells at path length l span [0, min(l, R)], not [0, R], and a draw is
scaled to min(T, R) with the actual T.  (Both matter in absorbers, where T
is short and most of the flux is laid down near the centre.)

GRID.  N_R radii, even in log R from R_MIN to R_MAX.  For each, N_L path
lengths, even in log(1 + l) from 0 to L_MAX(R) = 2 R^2 + 5 R; beyond that
the shape of Q no longer changes (a diffusing walk forgets its start after
about R^2 / pi^2 mean free paths, so by 2 R^2 the slowest correction is
down by exp(-2 pi^2)), and the last grid point is used.  N_RHO shells,
even in r / min(l, R).

Range: transport.py shrinks balls to R_MAX when this rule is on, and
refuses an R* below R_MIN.

Cost: building takes about two minutes on four cores (N_PER = one million
walks per radius, so the table's own sampling noise stays well below any
run's); a draw is a few random numbers and three binary searches.
"""

from pathlib import Path

import numpy as np
from numba import njit, prange

from core import rng

N_R, N_L, N_RHO, N_PER = 32, 96, 128, 1000000
R_MIN, R_MAX = 1.0, 30.0
VERSION = 3                     # bump when the stored layout changes


def l_max(R):
    return 2.0 * R * R + 5.0 * R


# ------------------------------------------------------------ the walk
@njit(cache=True)
def _first_flight(R, state, i):
    """The first flight of a walk conditioned to scatter inside (walk.py)."""
    xi = rng.rand(state, i)
    return -np.log(1.0 - xi * (-np.expm1(-R)))


@njit(cache=True)
def record_positions(R, ell, nb, state, i, hist, unsc, inside):
    """One collided walk (walk.py's rules).  For every grid path length
    ell[j] the walk is still inside at, count it in inside[j] and either
    unsc[j] (not scattered yet) or hist[j, shell of its radius].  The
    shells span [0, min(l, R)]: a walk that has gone a path l is within l
    of the centre, so short paths get shells as fine as long ones."""
    nl = ell.size
    l1 = _first_flight(R, state, i)
    u, v, w = rng.isotropic(state, i)
    j = 0
    while j < nl and ell[j] < l1:            # straight out from the centre
        inside[j] += 1
        unsc[j] += 1
        j += 1
    x, y, z = l1 * u, l1 * v, l1 * w
    s = l1
    while True:
        u, v, w = rng.isotropic(state, i)
        b = x * u + y * v + z * w
        c = x * x + y * y + z * z - R * R
        t_exit = -b + np.sqrt(max(b * b - c, 0.0))
        l = -np.log(1.0 - rng.rand(state, i))
        step = min(l, t_exit)
        while j < nl and ell[j] < s + step:
            t = ell[j] - s
            r = np.sqrt((x + t * u) ** 2 + (y + t * v) ** 2
                        + (z + t * w) ** 2)
            hist[j, min(int(r / min(ell[j], R) * nb), nb - 1)] += 1
            inside[j] += 1
            j += 1
        x, y, z = x + step * u, y + step * v, z + step * w
        s += step
        if l >= t_exit:
            return s


@njit(cache=True, parallel=True)
def _record_many(R, ell, nb, n, state, nchunk):
    nl = ell.size
    hist = np.zeros((nchunk, nl, nb))
    unsc = np.zeros((nchunk, nl))
    inside = np.zeros((nchunk, nl))
    per = (n + nchunk - 1) // nchunk
    for c in prange(nchunk):
        for i in range(c * per, min(n, (c + 1) * per)):
            record_positions(R, ell, nb, state, i, hist[c], unsc[c],
                             inside[c])
    return hist.sum(axis=0), unsc.sum(axis=0), inside.sum(axis=0)


# ------------------------------------------------------------ drawing
@njit(cache=True, inline="always")
def _bracket(grid, q):
    """(k0, frac) with grid[k0] <= q <= grid[k0 + 1], clamped to the ends."""
    n = grid.size
    lo, hi = 0, n - 1
    if q <= grid[0]:
        return 0, 0.0
    if q >= grid[n - 1]:
        return n - 2, 1.0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= q:
            lo = mid
        else:
            hi = mid
    return lo, (q - grid[lo]) / (grid[lo + 1] - grid[lo])


@njit(cache=True)
def draw_r(log_r, log1p_ell, f_unsc, cdf, R, kappa, s, state, i):
    """A radius (in mfp, from the ball's centre) at which to score the
    whole flux of a ball of radius R whose walk had path s (mfp), in a
    material with sigma_a / sigma_s = kappa.  See the docstring."""
    return draw_rT(log_r, log1p_ell, f_unsc, cdf, R, kappa, s, state, i)[0]


@njit(cache=True)
def draw_rT(log_r, log1p_ell, f_unsc, cdf, R, kappa, s, state, i):
    """draw_r, also returning the path length T (mfp) at which the point
    lies (core/ext.py: the time of a point in a ball)."""
    # 1. the path length T in [0, s], density proportional to exp(-kappa T)
    xi = rng.rand(state, i)
    if kappa * s < 1.0e-9:
        T = xi * s
    else:
        T = -np.log1p(-xi * (-np.expm1(-kappa * s))) / kappa
    T = min(T, s)

    # 2. the grid radius: one of the two neighbours of R, by closeness
    k0, fr = _bracket(log_r, np.log(R))
    kr = k0 + 1 if rng.rand(state, i) < fr else k0

    # 3. the grid path length, the same way (beyond the grid: the last)
    j0, fl = _bracket(log1p_ell[kr], np.log1p(T))
    jl = j0 + 1 if rng.rand(state, i) < fl else j0

    # 4. not scattered yet: exactly at radius T; otherwise from the shells
    if rng.rand(state, i) < f_unsc[kr, jl]:
        return min(T, R), T
    c = cdf[kr, jl]
    nb = c.size - 1
    xi = rng.rand(state, i)
    lo, hi = 0, nb
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if c[mid] <= xi:
            lo = mid
        else:
            hi = mid
    width = c[lo + 1] - c[lo]
    frac = (xi - c[lo]) / width if width > 0.0 else 0.5
    return (lo + min(max(frac, 0.0), 1.0)) / nb * min(T, R), T


class Profile:
    name = "profile"

    def __init__(self, log_r, log1p_ell, f_unsc, cdf):
        self.log_r = np.ascontiguousarray(log_r, np.float64)
        self.log1p_ell = np.ascontiguousarray(log1p_ell, np.float64)
        self.f_unsc = np.ascontiguousarray(f_unsc, np.float64)
        self.cdf = np.ascontiguousarray(cdf, np.float64)
        self.r_min = float(np.exp(log_r[0]))
        self.r_max = float(np.exp(log_r[-1]))

    def arrays(self):
        """What transport.py hands to draw_r."""
        return self.log_r, self.log1p_ell, self.f_unsc, self.cdf

    @classmethod
    def build(cls, r_min=R_MIN, r_max=R_MAX, n_r=N_R, n_l=N_L, nb=N_RHO,
              n_per=N_PER, seed=202, log=None):
        log_r = np.linspace(np.log(r_min), np.log(r_max), n_r)
        log1p_ell = np.zeros((n_r, n_l))
        f_unsc = np.zeros((n_r, n_l))
        cdf = np.zeros((n_r, n_l, nb + 1))
        for a, lr in enumerate(log_r):
            R = float(np.exp(lr))
            log1p_ell[a] = np.linspace(0.0, np.log1p(l_max(R)), n_l)
            ell = np.expm1(log1p_ell[a])
            state = rng.streams(np.uint64(seed), a * n_per, n_per)
            hist, unsc, inside = _record_many(R, ell, nb, n_per, state, 32)
            f_unsc[a] = np.where(inside > 0, unsc / np.maximum(inside, 1),
                                 1.0)
            tot = hist.sum(axis=1, keepdims=True)
            c = np.cumsum(hist, axis=1) / np.where(tot > 0, tot, 1.0)
            cdf[a, :, 1:] = np.where(tot > 0, c,
                                     np.arange(1, nb + 1) / nb)
            if log:
                log(f"  profile R = {R:6.2f}: walks still inside at the "
                    f"last grid length {int(inside[-1])} of {n_per}")
        cdf[:, :, -1] = 1.0
        return cls(log_r, log1p_ell, f_unsc, cdf)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, version=VERSION, log_r=self.log_r,
                 log1p_ell=self.log1p_ell, f_unsc=self.f_unsc, cdf=self.cdf)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        if int(d.get("version", 1)) != VERSION:
            raise ValueError("stale profile file")
        return cls(d["log_r"], d["log1p_ell"], d["f_unsc"], d["cdf"])


FILE = Path(__file__).resolve().parent.parent / "data" / "ball" / \
    "profile.npz"


def get(log=print):
    """The stored profile, built (and saved) on first use or when the file
    is from an older layout."""
    if FILE.exists():
        try:
            return Profile.load(FILE)
        except (ValueError, KeyError):
            pass
    log(f"building the mesh profile ({FILE}) ...")
    p = Profile.build(log=log)
    p.save(FILE)
    return p


# --------------------------------------------- the exact answer, for checks
@njit(cache=True, inline="always")
def _deposit(acc, kappa, l0, ta, tb, j):
    """Decayed track exp(-kappa l) dl from l0+ta to l0+tb, into shell j."""
    dt = tb - ta
    if dt <= 0.0:
        return
    if kappa * dt < 1.0e-12:
        acc[j] += np.exp(-kappa * (l0 + ta)) * dt
    else:
        acc[j] += np.exp(-kappa * (l0 + ta)) * (-np.expm1(-kappa * dt)) \
            / kappa


@njit(cache=True)
def _segment(acc, kappa, R, nb, x, y, z, u, v, w, length, l0):
    """Spread one straight segment over the radial shells it crosses.
    r(t)^2 = c0 + 2 b t + t^2 falls until t = -b and rises after, so the
    segment is cut there into two monotone pieces, and each piece is cut
    where it crosses a shell radius."""
    c0 = x * x + y * y + z * z
    b = x * u + y * v + z * w
    dr = R / nb
    tm = min(max(-b, 0.0), length)
    for piece in range(2):
        ta, tb = (0.0, tm) if piece == 0 else (tm, length)
        if tb <= ta:
            continue
        t = ta
        r_a = np.sqrt(max(c0 + 2.0 * b * ta + ta * ta, 0.0))
        r_b = np.sqrt(max(c0 + 2.0 * b * tb + tb * tb, 0.0))
        j = min(int(r_a / dr), nb - 1)
        jb = min(int(r_b / dr), nb - 1)
        if piece == 1:
            while j < jb:
                re = (j + 1) * dr
                tc = -b + np.sqrt(max(b * b - c0 + re * re, 0.0))
                tc = min(max(tc, t), tb)
                _deposit(acc, kappa, l0, t, tc, j)
                t = tc
                j += 1
        else:
            while j > jb:
                re = j * dr
                tc = -b - np.sqrt(max(b * b - c0 + re * re, 0.0))
                tc = min(max(tc, t), tb)
                _deposit(acc, kappa, l0, t, tc, j)
                t = tc
                j -= 1
        _deposit(acc, kappa, l0, t, tb, j)


@njit(cache=True)
def exact_walk(acc, kappa, R, nb, state, i):
    """One collided walk (walk.py's rules) laying its decayed track into
    radial shells acc[shell], exactly.  Returns its path s."""
    l1 = _first_flight(R, state, i)
    u, v, w = rng.isotropic(state, i)
    _segment(acc, kappa, R, nb, 0.0, 0.0, 0.0, u, v, w, l1, 0.0)
    x, y, z = l1 * u, l1 * v, l1 * w
    s = l1
    while True:
        u, v, w = rng.isotropic(state, i)
        b = x * u + y * v + z * w
        c = x * x + y * y + z * z - R * R
        t_exit = -b + np.sqrt(max(b * b - c, 0.0))
        l = -np.log(1.0 - rng.rand(state, i))
        step = min(l, t_exit)
        _segment(acc, kappa, R, nb, x, y, z, u, v, w, step, s)
        x, y, z = x + step * u, y + step * v, z + step * w
        s += step
        if l >= t_exit:
            return s
