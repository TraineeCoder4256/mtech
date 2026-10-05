"""The solver: ordinary Monte Carlo, with ball steps where they fit.

Used by: run.py and checks/ (call `run`).
Uses: geometry.py (where am I, how far to the next surface or the nearest
one), tallies.py (scoring), rng.py (random numbers), and a ball backend
from ball/ (the exact walk, the lookup table or a trained network).

One energy, isotropic scattering, fixed source (SRS Stage 1).

THE PHYSICS, as in gmc2d's mc.py.  A particle flies a distance drawn from
the SCATTERING cross section, -ln(xi)/sigma_s.  Absorption never kills it;
instead its weight decays continuously, w -> w exp(-sigma_a l) over a path
l ("implicit capture").  So the walk itself is a pure scatterer's walk, and
absorption is bookkeeping on top.  When the weight drops below `w_cut`,
Russian roulette either kills it or restores it to `w_surv` (keeping the
expected weight unchanged).

THE BALL STEP.  Right after a scatter, the new direction is isotropic and
the particle has no memory of where it came from.  If the nearest surface
is d away, the ball of radius d around it holds one material, and the walk
until the particle leaves that ball is the walk of a pure scatterer in a
ball of R = d sigma_s mean free paths, started at the centre.  By symmetry:

    * the exit point is uniform on the sphere,
    * the exit direction, measured from the outward normal there, has a
      cosine mu and a uniformly random azimuth,
    * mu and the total path s inside depend on R only.

So a backend only has to draw (mu, s) for a given R; this file does the
rest: picks the exit point, turns mu into a direction, deposits the flux
w (1 - exp(-sigma_a s)) / sigma_a, and decays the weight by exp(-sigma_a s).

Two details keep it exact and cheap:

    * A ball is used only if R >= r_star (small balls cost more than the
      flights they save) and is shrunk to r_max if bigger (the backend's
      range).
    * The particle first draws its ordinary flight distance.  If that is
      longer than the ball's radius, it would leave the ball without
      scattering -- so it just takes that flight; no ball is needed.  Only
      when it would scatter inside is the backend asked for a walk, and the
      backend's walk is conditioned on "scatters at least once".  This is
      exactly the right split (probability exp(-R) / 1 - exp(-R)), and the
      backend never has to learn the spike of straight-through particles.

THE LOOP.  Particles run in parallel chunks.  Each runs until it dies or
asks for a ball step; then every waiting particle's R goes to the backend
in ONE call, the answers are written back, and the chunks resume.  So a
network sees big batches, and the oracle and table use the particle's own
random stream (bit-for-bit reproducible runs).
"""

from collections import namedtuple
import time

import numpy as np
from numba import njit, prange

from . import geometry as G
from . import rng
from . import surfaces as S
from . import tallies as T

DEAD, MOVING, WAITING, READY = 0, 1, 2, 3
TINY = 1.0e-8            # cm: how far past a surface a crossing is pushed
GUARD = 1.0e-9           # cm: a ball stops this far short of a surface
MAX_EVENTS = 10_000_000  # per history, a guard against a stuck particle
INF = G.INF

# work counters, one row per chunk (cnt[chunk, k])
COUNTERS = ("histories", "flights", "scatters", "crossings", "balls",
            "ball_skipped", "roulette_killed", "lost", "rescued")
(C_HIST, C_FLIGHT, C_SCATTER, C_CROSS, C_BALL, C_SKIP, C_RKILL, C_LOST,
 C_RESCUE) = range(len(COUNTERS))

Particles = namedtuple("Particles", [
    "x", "y", "z", "u", "v", "w", "wgt", "seed", "status", "fresh",
    "cell", "mat",                    # where a waiting particle is
    "ball_R", "ball_d",               # its ball, in mfp and in cm
    "mu", "s",                        # the backend's answer (s in mfp)
])

Physics = namedtuple("Physics", [
    "sig_s", "sig_a",                 # per material, 1/cm
    "bc_surfs",                       # surfaces with a boundary condition
    "w_cut", "w_surv",                # Russian roulette
    "ball_on", "r_star", "r_max",     # ball step on/off, R range (mfp)
    "mesh_cap",                       # 'cap' rule for mesh tallies
])

SourceArrays = namedtuple("SourceArrays", [
    "kind", "lo", "hi", "mono", "direction", "cdf"])


# ------------------------------------------------------------- directions
@njit(cache=True, inline="always")
def rotate(u, v, w, mu, phi):
    """The direction at polar cosine mu and azimuth phi from (u, v, w)
    (OpenMC's rotate_angle)."""
    a = np.sqrt(max(0.0, 1.0 - mu * mu))
    b = np.sqrt(max(0.0, 1.0 - w * w))
    cp, sp = np.cos(phi), np.sin(phi)
    if b > 1.0e-10:
        return (mu * u + a * (u * w * cp - v * sp) / b,
                mu * v + a * (v * w * cp + u * sp) / b,
                mu * w - a * b * cp)
    b = np.sqrt(max(0.0, 1.0 - v * v))
    return (mu * u + a * (u * v * cp + w * sp) / b,
            mu * v - a * b * cp,
            mu * w + a * (v * w * cp - u * sp) / b)


@njit(cache=True, inline="always")
def _reflect(g, k, x, y, z, u, v, w):
    """Mirror (u, v, w) in surface k at (x, y, z)."""
    nx, ny, nz = S.normal(g.surf_kind[k], g.surf_coef[k], x, y, z)
    nn = nx * nx + ny * ny + nz * nz
    dot = (u * nx + v * ny + w * nz) / nn
    u, v, w = u - 2.0 * dot * nx, v - 2.0 * dot * ny, w - 2.0 * dot * nz
    r = np.sqrt(u * u + v * v + w * w)
    return u / r, v / r, w / r


# ----------------------------------------------------------------- source
@njit(cache=True)
def _birth_one(g, src, P, i):
    """Sample particle i from the source.  Returns False if no valid
    position was found (a source outside the geometry)."""
    xi = rng.rand(P.seed, i)
    j = 0
    while j < src.cdf.size - 1 and xi > src.cdf[j]:
        j += 1
    found = False
    for _ in range(1000):
        if src.kind[j] == 0:
            x, y, z = src.lo[j, 0], src.lo[j, 1], src.lo[j, 2]
        else:
            x = src.lo[j, 0] + (src.hi[j, 0] - src.lo[j, 0]) * \
                rng.rand(P.seed, i)
            y = src.lo[j, 1] + (src.hi[j, 1] - src.lo[j, 1]) * \
                rng.rand(P.seed, i)
            z = src.lo[j, 2] + (src.hi[j, 2] - src.lo[j, 2]) * \
                rng.rand(P.seed, i)
        _, mat = G.locate(g, x, y, z)
        if mat != G.LOST:
            found = True
            break
    if not found:
        return False
    if src.mono[j]:
        u, v, w = src.direction[j, 0], src.direction[j, 1], \
            src.direction[j, 2]
    else:
        u, v, w = rng.isotropic(P.seed, i)
    P.x[i], P.y[i], P.z[i] = x, y, z
    P.u[i], P.v[i], P.w[i] = u, v, w
    P.wgt[i] = 1.0
    P.fresh[i] = not src.mono[j]       # isotropic birth = like a scatter
    P.status[i] = MOVING
    return True


@njit(cache=True, parallel=True)
def _birth(g, src, P, cnt, nchunk):
    n = P.x.size
    per = (n + nchunk - 1) // nchunk
    for c in prange(nchunk):
        for i in range(c * per, min(n, (c + 1) * per)):
            cnt[c, C_HIST] += 1
            if not _birth_one(g, src, P, i):
                P.status[i] = DEAD
                cnt[c, C_LOST] += 1


# ------------------------------------------------------------ one history
@njit(cache=True)
def _track(i, g, ph, L, P, row, cnt):
    """Run particle i until it dies or asks for a ball step."""
    x, y, z = P.x[i], P.y[i], P.z[i]
    u, v, w = P.u[i], P.v[i], P.w[i]
    wgt = P.wgt[i]
    fresh = P.fresh[i]

    if P.status[i] == READY:
        # ---- finish a ball step: the backend has filled mu and s
        cell, mat = P.cell[i], P.mat[i]
        sa = ph.sig_a[mat]
        d = P.ball_d[i]
        s_cm = P.s[i] * d / P.ball_R[i]          # mfp -> cm
        flux = T._decayed(wgt, sa, 0.0, s_cm)
        T.score_ball(row, L, cell, mat, sa, flux, x, y, z)
        wgt *= np.exp(-sa * s_cm)
        nx, ny, nz = rng.isotropic(P.seed, i)    # exit point
        x, y, z = x + d * nx, y + d * ny, z + d * nz
        u, v, w = rotate(nx, ny, nz, P.mu[i],
                         2.0 * np.pi * rng.rand(P.seed, i))
        fresh = False
        cnt[C_BALL] += 1
        if wgt < ph.w_cut:
            if rng.rand(P.seed, i) < wgt / ph.w_surv:
                wgt = ph.w_surv
            else:
                cnt[C_RKILL] += 1
                P.status[i] = DEAD
                return

    rescues = 0
    for _ in range(MAX_EVENTS):
        want_near = fresh and ph.ball_on
        cell, mat, near, t_ray, surf, at_root = G._descend(
            g, x, y, z, u, v, w, want_near, True)

        if mat == G.LOST:
            # Just past a boundary surface that a lattice wall or an inner
            # surface beat by a rounding error: apply its condition here.
            k = -1
            best = 10.0 * TINY
            for j in range(ph.bc_surfs.size):
                s = ph.bc_surfs[j]
                dd = S.nearest(g.surf_kind[s], g.surf_coef[s], x, y, z)
                if dd < best:
                    k, best = s, dd
            if k < 0 or rescues > 3:
                cnt[C_LOST] += 1
                P.status[i] = DEAD
                return
            rescues += 1
            cnt[C_RESCUE] += 1
            if g.surf_bc[k] == G.BC_VACUUM:
                row[T.LEAKAGE] += wgt
                P.status[i] = DEAD
                return
            u, v, w = _reflect(g, k, x, y, z, u, v, w)
            x, y, z = x + 2.0 * TINY * u, y + 2.0 * TINY * v, \
                z + 2.0 * TINY * w
            continue
        rescues = 0

        ss = ph.sig_s[mat] if mat >= 0 else 0.0
        sa = ph.sig_a[mat] if mat >= 0 else 0.0
        d_sc = -np.log(1.0 - rng.rand(P.seed, i)) / ss if ss > 0.0 else INF

        # ---- a ball step, if one fits and the particle would scatter in it
        if want_near and ss > 0.0:
            room = near - GUARD
            if ph.mesh_cap:
                room = min(room, T.mesh_room(L, x, y, z) - GUARD)
            R = room * ss
            if R >= ph.r_star:
                if R > ph.r_max:
                    R = ph.r_max
                    room = R / ss
                if d_sc < room:
                    P.x[i], P.y[i], P.z[i] = x, y, z
                    P.wgt[i] = wgt
                    P.cell[i], P.mat[i] = cell, mat
                    P.ball_R[i], P.ball_d[i] = R, room
                    P.status[i] = WAITING
                    return
                cnt[C_SKIP] += 1          # flies straight out: no ball
        fresh = False

        # ---- an ordinary flight
        scatter = d_sc < t_ray
        step = d_sc if scatter else t_ray
        if step >= INF:
            cnt[C_LOST] += 1              # flying off to infinity in a void
            P.status[i] = DEAD
            return
        T.score_segment(row, L, cell, mat, sa, wgt, x, y, z, u, v, w, step)
        wgt *= np.exp(-sa * step)
        x, y, z = x + step * u, y + step * v, z + step * w
        cnt[C_FLIGHT] += 1

        if scatter:
            u, v, w = rng.isotropic(P.seed, i)
            fresh = True
            cnt[C_SCATTER] += 1
        else:
            cnt[C_CROSS] += 1
            if at_root and g.surf_bc[surf] == G.BC_VACUUM:
                row[T.LEAKAGE] += wgt
                P.status[i] = DEAD
                return
            if at_root and g.surf_bc[surf] == G.BC_REFLECTIVE:
                u, v, w = _reflect(g, surf, x, y, z, u, v, w)
            x, y, z = x + TINY * u, y + TINY * v, z + TINY * w

        if wgt < ph.w_cut:
            if rng.rand(P.seed, i) < wgt / ph.w_surv:
                wgt = ph.w_surv
            else:
                cnt[C_RKILL] += 1
                P.status[i] = DEAD
                return
    cnt[C_LOST] += 1
    P.status[i] = DEAD


@njit(cache=True, parallel=True)
def _advance(g, ph, L, P, acc, cnt, nchunk):
    """Run every live particle until it dies or waits for a ball."""
    n = P.x.size
    per = (n + nchunk - 1) // nchunk
    for c in prange(nchunk):
        for i in range(c * per, min(n, (c + 1) * per)):
            st = P.status[i]
            if st == MOVING or st == READY:
                _track(i, g, ph, L, P, acc[c], cnt[c])


# ------------------------------------------------------------- the driver
def _particles(n):
    f = lambda: np.zeros(n)
    return Particles(f(), f(), f(), f(), f(), f(), f(),
                     np.zeros(n, np.uint64), np.zeros(n, np.int64),
                     np.zeros(n, np.bool_), np.zeros(n, np.int64),
                     np.zeros(n, np.int64), f(), f(), f(), f())


def run(problem, particles=100_000, batches=20, seed=1, backend=None,
        r_star=3.0, r_max=None, mesh_rule="cap", w_cut=0.25, w_surv=1.0,
        nchunk=32, verbose=False):
    """Solve a Problem (from openmc_import.load).

    backend      None for plain Monte Carlo, or a ball backend from ball/
    r_star       smallest ball used, in mean free paths
    r_max        biggest ball used (default: the backend's own limit)
    mesh_rule    'cap' or 'centre' (see tallies.py)

    Returns tallies.Results: per-source-particle means and standard errors.
    """
    if mesh_rule not in ("cap", "centre"):
        raise ValueError("mesh_rule must be 'cap' or 'centre'")
    g, m = problem.geom, problem.mats
    if backend is not None:
        r_max = backend.r_max if r_max is None else min(r_max,
                                                        backend.r_max)
        if r_star < backend.r_min:
            raise ValueError(f"r_star {r_star} is below the backend's "
                             f"smallest ball {backend.r_min}")
    ph = Physics(m.sig_s.astype(float), m.sig_a.astype(float),
                 np.flatnonzero(g.surf_bc != G.BC_TRANSMISSION),
                 float(w_cut), float(w_surv), backend is not None,
                 float(r_star), float(r_max or INF), mesh_rule == "cap")
    s = problem.source
    src = SourceArrays(s.kind, s.lo, s.hi, s.mono.astype(np.bool_),
                       s.direction, s.cdf)
    L = T.compile_layout(problem.tallies, len(problem.cell_ids),
                         len(m.ids))
    accum = T.Accumulator(L)
    cnt = np.zeros((nchunk, len(COUNTERS)), np.int64)
    t_track = t_backend = 0.0
    calls = 0

    for b in range(batches):
        P = _particles(particles)
        P.seed[:] = rng.streams(np.uint64(seed), b * particles, particles)
        acc = np.zeros((nchunk, L.n_entries))
        t0 = time.perf_counter()
        _birth(g, src, P, cnt, nchunk)
        while True:
            _advance(g, ph, L, P, acc, cnt, nchunk)
            t1 = time.perf_counter()
            t_track += t1 - t0
            idx = np.flatnonzero(P.status == WAITING)
            if idx.size == 0:
                break
            mu, s_mfp = backend.sample(P.ball_R[idx], P.seed, idx)
            P.mu[idx], P.s[idx] = mu, s_mfp
            P.status[idx] = READY
            calls += 1
            t0 = time.perf_counter()
            t_backend += t0 - t1
        accum.add_batch(acc, particles)
        if verbose:
            mean, se = accum.mean_se()
            print(f"batch {b + 1}/{batches}  leakage {mean[0]:.5f} "
                  f"+- {se[0]:.5f}", flush=True)

    counters = dict(zip(COUNTERS, cnt.sum(axis=0).tolist()))
    counters["backend_calls"] = calls
    timing = {"transport_s": t_track, "backend_s": t_backend}
    settings = dict(particles=particles, batches=batches, seed=seed,
                    backend=getattr(backend, "name", None), r_star=r_star,
                    r_max=r_max, mesh_rule=mesh_rule, w_cut=w_cut,
                    w_surv=w_surv, nchunk=nchunk)
    return accum.results(problem.tallies, counters, timing, settings)
