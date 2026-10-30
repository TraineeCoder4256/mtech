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
expected weight unchanged).  Both are relative to the particle's weight at
birth, which is 1 unless the source gives weights (sources.py).

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

Where in a mesh that flux goes is the mesh rule (tallies.py): under
'profile', the default, a point of the walk is drawn (ball/profile.py:
a path length T weighted by the flux laid down there, then where walks of
that length sit) and its voxel gets the deposit -- exact on average for
every voxel.  Under 'cap' the ball is shrunk to one voxel.

One point per ball is already exact on average, but a big ball then drops
its whole flux in one voxel among hundreds, which is very noisy -- and
worst at the edge of the geometry, where a voxel holding a thin sliver of
material is hit rarely but hard (measured: the sphere's edge voxels gave
chi2/dof 1.86 against OpenMC with one point, the error bar itself being
unreliable for such rare, large scores).  So the flux is split equally over

    K = min(K_MAX, ceil(2 (d / h)^2))

independent points, d the ball's radius and h the smallest voxel width: a
ball spans about (2 d / h)^3 voxels, a flight across it about 2 d / h, so K
of that ratio gives each voxel about as many scores as a flight would.  K
changes the noise and the cost, never the mean.  Measured 27 Oct 2026
(exact balls, R* = 1, 2M histories, mesh chi2/dof against OpenMC; plain MC
gets 1.04 on the sphere):

    K_MAX    sphere chi2/dof  transport s     slab chi2/dof  transport s
      1           1.945           1.9              1.346          6.8
      4           1.192           2.5              1.116          8.5
     16           1.074           5.2              0.870         12.7
     64           1.079          15.0              0.847         22.7

Each point costs about 0.1 us; past 16 they buy nothing measurable, hence
K_MAX = 16.

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
from ball import profile as ball_profile
from ball.profile import draw_r

DEAD, MOVING, WAITING, READY = 0, 1, 2, 3
TINY = 1.0e-8            # cm: how far past a surface a crossing is pushed
GUARD = 1.0e-9           # cm: a ball stops this far short of a surface
MAX_EVENTS = 10_000_000  # per history, a guard against a stuck particle
K_MAX = 16               # most mesh points a ball's flux is split over
INF = G.INF

# work counters, one row per chunk (cnt[chunk, k])
COUNTERS = ("histories", "flights", "scatters", "crossings", "balls",
            "ball_skipped", "roulette_killed", "lost", "rescued",
            "mesh_points")
(C_HIST, C_FLIGHT, C_SCATTER, C_CROSS, C_BALL, C_SKIP, C_RKILL, C_LOST,
 C_RESCUE, C_POINTS) = range(len(COUNTERS))

Particles = namedtuple("Particles", [
    "x", "y", "z", "u", "v", "w", "wgt", "seed", "status", "fresh",
    "cell", "mat",                    # where a waiting particle is
    "ball_R", "ball_d",               # its ball, in mfp and in cm
    "mu", "s",                        # the backend's answer (s in mfp)
    "w0",                             # weight at birth (roulette is
                                      # relative to it)
    "path",                           # (n, MAX_LEVELS): a waiting
                                      # particle's cell at every level
])

Physics = namedtuple("Physics", [
    "sig_s", "sig_a",                 # per material, 1/cm
    "bc_surfs",                       # surfaces with a boundary condition
    "w_cut", "w_surv",                # Russian roulette
    "ball_on", "r_star", "r_max",     # ball step on/off, R range (mfp)
    "mesh_cap",                       # 'cap' rule for mesh tallies
    "mesh_profile",                   # 'profile' rule (and a mesh exists)
    "mesh_h", "k_max",                # smallest voxel width; most points
    "prof_log_r", "prof_log1p_ell", "prof_f_unsc", "prof_cdf",
                                      # ball/profile.py's table
])

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


@njit(cache=True)
def _boundary(g, k, P, i, x, y, z, u, v, w):
    """A particle at (x, y, z) on root surface k, heading out through it:
    where it carries on from, and in which direction, under k's boundary
    condition (not vacuum, which ends the history).

      reflective   mirrored direction
      white        a new direction drawn from the cosine law about the
                   inward normal (mu = sqrt(xi)), OpenMC's definition
      periodic     moved to the partner plane: translational (parallel
                   planes) by g.surf_shift[k], same direction; rotational
                   (planes through the z axis) by turning position and
                   direction through g.surf_angle[k] about z, as OpenMC
    """
    bc = g.surf_bc[k]
    if bc == G.BC_REFLECTIVE:
        u, v, w = _reflect(g, k, x, y, z, u, v, w)
    elif bc == G.BC_WHITE:
        nx, ny, nz = S.normal(g.surf_kind[k], g.surf_coef[k], x, y, z)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        sg = -1.0 if u * nx + v * ny + w * nz > 0.0 else 1.0   # inward
        mu = np.sqrt(rng.rand(P.seed, i))
        u, v, w = rotate(sg * nx / nn, sg * ny / nn, sg * nz / nn, mu,
                         2.0 * np.pi * rng.rand(P.seed, i))
    elif bc == G.BC_PERIODIC:
        a = g.surf_angle[k]
        if a != 0.0:
            ca, sa = np.cos(a), np.sin(a)
            x, y = ca * x - sa * y, sa * x + ca * y
            u, v = ca * u - sa * v, sa * u + ca * v
        else:
            x, y, z = (x + g.surf_shift[k, 0], y + g.surf_shift[k, 1],
                       z + g.surf_shift[k, 2])
    return x, y, z, u, v, w


@njit(cache=True, inline="always")
def _crossing_sign(g, k, x, y, z, u, v, w):
    """+1 if (u, v, w) at (x, y, z) heads to surface k's positive side."""
    nx, ny, nz = S.normal(g.surf_kind[k], g.surf_coef[k], x, y, z)
    return 1.0 if u * nx + v * ny + w * nz > 0.0 else -1.0


# ----------------------------------------------------------------- source
@njit(cache=True, parallel=True)
def _birth(xyz, uvw, wgt, iso, P, cnt, nchunk):
    """Start every particle from its source site (sources.py drew them)."""
    n = P.x.size
    per = (n + nchunk - 1) // nchunk
    for c in prange(nchunk):
        for i in range(c * per, min(n, (c + 1) * per)):
            cnt[c, C_HIST] += 1
            _start(P, i, xyz, uvw, wgt, iso)   # a call, not inlined: a
            # parallel loop that writes a namedtuple's arrays directly loses
            # the writes (seen with numba here, 29 Oct 2026)


@njit(cache=True)
def _start(P, i, xyz, uvw, wgt, iso):
    P.x[i], P.y[i], P.z[i] = xyz[i, 0], xyz[i, 1], xyz[i, 2]
    P.u[i], P.v[i], P.w[i] = uvw[i, 0], uvw[i, 1], uvw[i, 2]
    P.wgt[i] = wgt[i]
    P.w0[i] = wgt[i]
    P.fresh[i] = iso[i]        # isotropic birth = like a scatter
    P.status[i] = MOVING


def source_rng(seed, batch):
    """The source's random numbers for one batch (numpy, independent of
    the particles' own streams)."""
    return np.random.default_rng([int(seed), int(batch), 0x5EED])


# ------------------------------------------------------------ one history
@njit(cache=True)
def _track(i, g, ph, L, X, P, row, cnt, pth, last, to, univ, scr):
    """Run particle i until it dies or asks for a ball step.

    pth, last, to, univ (MAX_LEVELS each) and scr (tallies.scratch) are
    the calling thread's work space: the cell at every level of the
    particle's path now, at the last successful descent (for a surface
    tally's cell-from), and on the far side of a crossing."""
    x, y, z = P.x[i], P.y[i], P.z[i]
    u, v, w = P.u[i], P.v[i], P.w[i]
    wgt = P.wgt[i]
    fresh = P.fresh[i]

    if P.status[i] == READY:
        # ---- finish a ball step: the backend has filled mu and s
        cell, mat = P.cell[i], P.mat[i]
        sa, ss = ph.sig_a[mat], ph.sig_s[mat]
        d = P.ball_d[i]
        s_cm = P.s[i] * d / P.ball_R[i]          # mfp -> cm
        flux = T._decayed(wgt, sa, 0.0, s_cm)
        path = P.path[i]
        T.score_ball(row, L, cell, path, mat, sa, ss, flux, scr)
        if ph.mesh_profile:
            # K points of the walk share the flux (tallies.py, profile.py)
            k_pts = min(ph.k_max, max(1, int(np.ceil(
                2.0 * (d / ph.mesh_h) ** 2))))
            for _ in range(k_pts):
                r = draw_r(ph.prof_log_r, ph.prof_log1p_ell, ph.prof_f_unsc,
                           ph.prof_cdf, P.ball_R[i], sa / ss,
                           P.s[i], P.seed, i) * d / P.ball_R[i]  # in cm
                px, py, pz = rng.isotropic(P.seed, i)
                T.score_ball_mesh(row, L, X, cell, path, mat, sa, ss,
                                  flux / k_pts, x + r * px, y + r * py,
                                  z + r * pz, scr)
            cnt[C_POINTS] += k_pts
        elif L.has_mesh:
            T.score_ball_mesh(row, L, X, cell, path, mat, sa, ss, flux,
                              x, y, z, scr)
        wgt *= np.exp(-sa * s_cm)
        nx, ny, nz = rng.isotropic(P.seed, i)    # exit point
        x, y, z = x + d * nx, y + d * ny, z + d * nz
        u, v, w = rotate(nx, ny, nz, P.mu[i],
                         2.0 * np.pi * rng.rand(P.seed, i))
        fresh = False
        cnt[C_BALL] += 1
        if wgt < ph.w_cut * P.w0[i]:
            if rng.rand(P.seed, i) < wgt / (ph.w_surv * P.w0[i]):
                wgt = ph.w_surv * P.w0[i]
            else:
                cnt[C_RKILL] += 1
                P.status[i] = DEAD
                return

    rescues = 0
    last_mat = G.LOST
    last[0] = -1
    for _ in range(MAX_EVENTS):
        want_near = fresh and ph.ball_on
        cell, mat, near, t_ray, surf, lvl, sg = G._descend(
            g, x, y, z, u, v, w, want_near, True, pth, L.has_surface)
        at_root = lvl == 0

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
                if L.has_surface:
                    T.score_surface(row, L, k,
                                    _crossing_sign(g, k, x, y, z, u, v, w)
                                    * wgt, last, last, last_mat, scr)
                row[T.LEAKAGE] += wgt
                P.status[i] = DEAD
                return
            x, y, z, u, v, w = _boundary(g, k, P, i, x, y, z, u, v, w)
            if g.surf_bc[k] == G.BC_PERIODIC and L.has_surface:
                k2 = int(g.surf_shift[k, 3])         # as at a crossing
                to[:] = -1
                m_to = G._path(g, x + TINY * u, y + TINY * v, z + TINY * w,
                               to, univ)
                T.score_surface(row, L, k2, _crossing_sign(
                    g, k2, x, y, z, u, v, w) * wgt, last, to, m_to, scr)
            x, y, z = x + 2.0 * TINY * u, y + 2.0 * TINY * v, \
                z + 2.0 * TINY * w
            continue
        rescues = 0
        if L.has_surface:
            for lv in range(pth.size):
                last[lv] = pth[lv]
                if pth[lv] < 0:
                    break
            last_mat = mat

        ss = ph.sig_s[mat] if mat >= 0 else 0.0
        sa = ph.sig_a[mat] if mat >= 0 else 0.0
        d_sc = -np.log(1.0 - rng.rand(P.seed, i)) / ss if ss > 0.0 else INF

        # ---- a ball step, if one fits and the particle would scatter in it
        if want_near and ss > 0.0:
            room = near - GUARD
            if ph.mesh_cap:
                room = min(room, T.mesh_room(L, X, x, y, z) - GUARD)
            elif L.has_meshsurf:      # a ball must not cross a mesh face
                room = min(room,
                           T.mesh_room(L, X, x, y, z, True) - GUARD)
            R = room * ss
            if R >= ph.r_star:
                if R > ph.r_max:
                    R = ph.r_max
                    room = R / ss
                if d_sc < room:
                    P.x[i], P.y[i], P.z[i] = x, y, z
                    P.wgt[i] = wgt
                    P.cell[i], P.mat[i] = cell, mat
                    for lv in range(pth.size):
                        P.path[i, lv] = pth[lv]
                        if pth[lv] < 0:
                            break
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
        T.score_segment(row, L, cell, pth, mat, sa, ss, wgt, x, y, z, u, v,
                        w, step, scr)
        if L.has_curved:
            T.score_segment_curved(row, L, X, cell, pth, mat, sa, ss,
                                   wgt, x, y, z, u, v, w, step, scr)
        if L.has_meshsurf:
            # Mesh faces crossed, as OpenMC sees them: a flight that ends
            # by passing through a surface (or out through vacuum) is seen
            # up to TINY beyond it, where the next flight starts, so a
            # mesh face lying ON that surface is counted; one that ends at
            # a reflective, white or periodic wall stops TINY short of it,
            # so a face on such a wall is not (OpenMC moves the particle
            # by TINY_BIT the same ways).
            reach = step
            if not scatter:
                bc = g.surf_bc[surf] if at_root else G.BC_TRANSMISSION
                reach = step + TINY if bc == G.BC_VACUUM or \
                    bc == G.BC_TRANSMISSION else step - TINY
            T.score_mesh_faces(row, L, X, sa, wgt, x, y, z, u, v, w,
                               reach)
        wgt *= np.exp(-sa * step)
        x, y, z = x + step * u, y + step * v, z + step * w
        cnt[C_FLIGHT] += 1

        if scatter:
            u, v, w = rng.isotropic(P.seed, i)
            fresh = True
            cnt[C_SCATTER] += 1
        else:
            cnt[C_CROSS] += 1
            bc = g.surf_bc[surf] if at_root else G.BC_TRANSMISSION
            if surf >= 0 and L.has_surface:
                # Current, as OpenMC 0.16 scores it (measured, v1.3 and
                # v1.5): signed by the side the flight crosses to, in the
                # surface's own frame (sg), at any universe level.
                #   vacuum      the cell left is also the 'cell entered'
                #   reflective  OpenMC scores +w then -w: exactly 0, so
                #   / white     nothing is scored
                #   periodic    scored after the jump, below
                #   otherwise   the cells on the far side (cell and
                #               material filters see the cell entered).
                #               A crossing that leaves the cell at the
                #               surface's level (and above) unchanged is
                #               the surface's extension beyond the cell's
                #               real boundary (the plane of a box face,
                #               outside the face): OpenMC does not count
                #               it, so neither does this.
                if bc == G.BC_VACUUM:
                    T.score_surface(row, L, surf, sg * wgt, pth, pth, mat,
                                    scr)
                elif bc == G.BC_TRANSMISSION:
                    to[:] = -1
                    m_to = G._path(g, x + TINY * u, y + TINY * v,
                                   z + TINY * w, to, univ)
                    moved = False
                    for lv in range(lvl + 1):
                        if to[lv] != pth[lv]:
                            moved = True
                    if moved:
                        T.score_surface(row, L, surf, sg * wgt, pth, to,
                                        m_to, scr)
            if bc == G.BC_VACUUM:
                row[T.LEAKAGE] += wgt
                P.status[i] = DEAD
                return
            if bc != G.BC_TRANSMISSION:
                x, y, z, u, v, w = _boundary(g, surf, P, i, x, y, z, u, v, w)
                if bc == G.BC_PERIODIC and L.has_surface:
                    # OpenMC scores a periodic crossing once, at the
                    # PARTNER plane, signed by the direction after the
                    # jump; cell-from is the cell left, cell the one
                    # arrived in (measured, v1.5)
                    k2 = int(g.surf_shift[surf, 3])
                    to[:] = -1
                    m_to = G._path(g, x + TINY * u, y + TINY * v,
                                   z + TINY * w, to, univ)
                    T.score_surface(row, L, k2, _crossing_sign(
                        g, k2, x, y, z, u, v, w) * wgt, pth, to, m_to, scr)
            x, y, z = x + TINY * u, y + TINY * v, z + TINY * w

        if wgt < ph.w_cut * P.w0[i]:
            if rng.rand(P.seed, i) < wgt / (ph.w_surv * P.w0[i]):
                wgt = ph.w_surv * P.w0[i]
            else:
                cnt[C_RKILL] += 1
                P.status[i] = DEAD
                return
    cnt[C_LOST] += 1
    P.status[i] = DEAD


@njit(cache=True, parallel=True)
def _advance(g, ph, L, X, P, acc, cnt, nchunk):
    """Run every live particle until it dies or waits for a ball.

    Chunk c takes particles [c*per, min(n, (c+1)*per)).  Each chunk is a
    disjoint set of histories with independent random streams, which is why
    tallies.py can use the chunks as the groups its error bar comes from;
    tallies.chunk_sizes must keep matching the split below.
    """
    n = P.x.size
    per = (n + nchunk - 1) // nchunk
    for c in prange(nchunk):
        pth = np.full(G.MAX_LEVELS, -1, np.int64)
        last = np.full(G.MAX_LEVELS, -1, np.int64)
        to = np.full(G.MAX_LEVELS, -1, np.int64)
        univ = np.full(G.MAX_LEVELS, -1, np.int64)
        scr = T.scratch(G.MAX_LEVELS)
        for i in range(c * per, min(n, (c + 1) * per)):
            st = P.status[i]
            if st == MOVING or st == READY:
                _track(i, g, ph, L, X, P, acc[c], cnt[c], pth, last, to,
                       univ, scr)


# ------------------------------------------------------------- the driver
def physics(problem, backend=None, r_star=3.0, r_max=None,
            mesh_rule="profile", w_cut=0.25, w_surv=1.0, k_max=None):
    """The Physics and tally Layout a run uses (also used by
    checks/loop_breakdown.py, which times run() phase by phase).
    Returns (ph, L, r_max actually used, X): X is the tallies.MeshExtra,
    passed to the kernels as its own argument (Numba's parallel loop
    cannot take it inside Physics)."""
    if mesh_rule not in ("profile", "cap"):
        raise ValueError("mesh_rule must be 'profile' or 'cap'")
    k_max = K_MAX if k_max is None else k_max
    g, m = problem.geom, problem.mats
    L, X = T.compile_layouts(problem.tallies, len(problem.cell_ids),
                             len(m.ids), len(g.surf_kind),
                             leaf=g.cell_fill_type <= G.FILL_VOID)
    use_profile = backend is not None and mesh_rule == "profile" and \
        L.has_mesh
    if backend is not None:
        r_max = backend.r_max if r_max is None else min(r_max,
                                                        backend.r_max)
        if r_star < backend.r_min:
            raise ValueError(f"r_star {r_star} is below the backend's "
                             f"smallest ball {backend.r_min}")
    mesh_h = float(X.mesh_h[L.mesh_on & ((L.kind == T.VOLUME) |
                                         (L.kind == T.CURVED))].min()) \
        if L.has_mesh else 1.0
    if use_profile:
        prof = ball_profile.get()
        r_max = min(r_max, prof.r_max)       # the profile's range
        if r_star < prof.r_min:
            raise ValueError(f"r_star {r_star} is below the mesh profile's "
                             f"smallest ball {prof.r_min}")
        tab = prof.arrays()
    else:
        tab = (np.zeros(2), np.zeros((1, 2)), np.zeros((1, 2)),
               np.zeros((1, 2, 2)))
    ph = Physics(m.sig_s.astype(float), m.sig_a.astype(float),
                 np.flatnonzero(g.surf_bc != G.BC_TRANSMISSION),
                 float(w_cut), float(w_surv), backend is not None,
                 float(r_star), float(r_max or INF), mesh_rule == "cap",
                 use_profile, mesh_h, int(k_max), *tab)
    return ph, L, r_max, X


def _particles(n):
    f = lambda: np.zeros(n)
    return Particles(f(), f(), f(), f(), f(), f(), f(),
                     np.zeros(n, np.uint64), np.zeros(n, np.int64),
                     np.zeros(n, np.bool_), np.zeros(n, np.int64),
                     np.zeros(n, np.int64), f(), f(), f(), f(), f(),
                     np.full((n, G.MAX_LEVELS), -1, np.int64))


def run(problem, particles=100_000, batches=20, seed=1, backend=None,
        r_star=3.0, r_max=None, mesh_rule="profile", w_cut=0.25, w_surv=1.0,
        nchunk=32, verbose=False, k_max=None):
    """Solve a Problem (from openmc_import.load).

    backend      None for plain Monte Carlo, or a ball backend from ball/
    r_star       smallest ball used, in mean free paths
    r_max        biggest ball used (default: the backend's own limit)
    mesh_rule    'profile' or 'cap' (see tallies.py)
    k_max        most mesh points per ball under 'profile' (default K_MAX;
                 1 gives the one-point rule, for comparisons)

    Returns tallies.Results: per-source-particle means and standard errors.
    """
    g = problem.geom
    ph, L, r_max, X = physics(problem, backend, r_star, r_max, mesh_rule,
                           w_cut, w_surv, k_max)
    accum = T.Accumulator(L)
    cnt = np.zeros((nchunk, len(COUNTERS)), np.int64)
    t_track = t_backend = 0.0
    calls = 0
    redrawn = 0

    for b in range(batches):
        P = _particles(particles)
        P.seed[:] = rng.streams(np.uint64(seed), b * particles, particles)
        acc = np.zeros((nchunk, L.n_entries))
        t0 = time.perf_counter()
        sites = problem.source.sample(particles, source_rng(seed, b))
        redrawn += sites.redrawn
        _birth(sites.xyz, sites.uvw, sites.wgt, sites.iso, P, cnt, nchunk)
        while True:
            _advance(g, ph, L, X, P, acc, cnt, nchunk)
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
        # OpenMC's leakage is per unit of source weight, its tallies per
        # source particle (sources.py); the same here
        acc[:, T.LEAKAGE] *= particles / sites.wgt.sum()
        accum.add_batch(acc, particles)
        if verbose:
            mean, se = accum.mean_se()
            print(f"batch {b + 1}/{batches}  leakage {mean[0]:.5f} "
                  f"+- {se[0]:.5f}", flush=True)

    counters = dict(zip(COUNTERS, cnt.sum(axis=0).tolist()))
    counters["backend_calls"] = calls
    counters["source_redrawn"] = redrawn
    timing = {"transport_s": t_track, "backend_s": t_backend}
    settings = dict(particles=particles, batches=batches, seed=seed,
                    backend=getattr(backend, "name", None), r_star=r_star,
                    r_max=r_max, mesh_rule=mesh_rule,
                    k_max=ph.k_max if ph.mesh_profile else None, w_cut=w_cut,
                    w_surv=w_surv, nchunk=nchunk)
    return accum.results(problem.tallies, counters, timing, settings)
