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

from . import detectors as DT
from . import ext as EX
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

# point detectors (detectors.py), passed as their own argument like
# tallies.MeshExtra so that runs without them pay nothing
Detectors = namedtuple("Detectors", [
    "on",                             # any detector at all
    "pos", "r0",                      # (n, 3) points; averaging radii, cm
    "sig_t",                          # per material, 1/cm
    "base",                           # entry of acc where they start
    "k",                              # profile points per ball
    "img", "img_det",                 # (m, 3) each detector and its wall
                                      # images; the detector of each
    "img_cut", "walls",               # images used within this distance
                                      # (cm); any wall (v1.6.1)
])
# what the general tally scorer (ext.py) follows of each particle, passed
# as its own argument too; arrays of length 1 when no ext tally exists
ExtState = namedtuple("ExtState", [
    "time",                           # s
    "ncoll",                          # scatters so far
    "mu_last",                        # cosine of the last scatter
    "prev_mat",                       # material before the last crossing
                                      # (-1 after a scatter: OpenMC's
                                      # material_last)
    "w_analog",                       # the weight an analog game carries
                                      # (the birth weight; WeightFilter)
    "born",                           # (n, k) birth bins (CellBorn, MeshBorn)
    "prev",                           # (n, MAX_LEVELS) cells before the
                                      # last crossing (OpenMC's cell_last)
    "v_mu",                           # Mu-filter runs only (else empty):
                                      # the last scatter cosine of the
                                      # analog history this walk stands
                                      # for, once it has been absorbed
                                      # (NaN while it lives; _advance)
])
K_DET = 4                # profile points per ball for detectors
BIRTH_SLICE = 4096       # source sites scored at detectors per call


def _ext_state(n, n_born, on, mu_on=False):
    nv = n if on and mu_on else 0
    n = n if on else 1
    return ExtState(np.zeros(n), np.zeros(n, np.int64), np.zeros(n),
                    np.full(n, -1, np.int64), np.ones(n),
                    np.full((n, max(n_born, 1)), -1, np.int64),
                    np.full((n, G.MAX_LEVELS), -1, np.int64),
                    np.full(nv, np.nan))


@njit(cache=True, inline="always")
def _iv(E, mat):
    """Inverse velocity in material mat (the last entry: void)."""
    return E.iv[mat] if mat >= 0 else E.iv[E.iv.size - 1]


@njit(cache=True)
def _ext_cross(row, E, L, X, g, P, XS, i, k, sw, frm, to, m_to, m_from, x,
               y, z, u, v, w, ws):
    """A crossing of surface k for the ext surface tallies (ext.cross),
    with MuSurface's cosine |u . n| (OpenMC flips the normal to the side
    crossed to, so it is never negative)."""
    nx, ny, nz = S.normal(g.surf_kind[k], g.surf_coef[k], x, y, z)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    mus = min(abs(u * nx + v * ny + w * nz) / nn, 1.0) if nn > 0.0 else 1.0
    EX.cross(row, E, L, X, g, P, XS, i, k, sw, frm, to, m_to, m_from, x, y,
             z, u, v, w, mus, ws)


@njit(cache=True)
def _ext_crossed(XS, i, cells, mat):
    """After any crossing: the cells and material left (CellFrom and
    MaterialFrom on later flights)."""
    for lv in range(cells.size):
        XS.prev[i, lv] = cells[lv]
    XS.prev_mat[i] = mat


def _no_detectors(n_entries, n_mats):
    return Detectors(False, np.zeros((0, 3)), np.zeros(0),
                     np.zeros(max(n_mats, 1)), n_entries, 1,
                     np.zeros((0, 3)), np.zeros(0, np.int64), INF, False)


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
def _track(i, g, ph, L, X, D, P, row, cnt, pth, last, to, univ, scr,
           dpth, E, XS, ws):
    """Run particle i until it dies or asks for a ball step.

    pth, last, to, univ (MAX_LEVELS each) and scr (tallies.scratch) are
    the calling thread's work space: the cell at every level of the
    particle's path now, at the last successful descent (for a surface
    tally's cell-from), and on the far side of a crossing; dpth is the
    point detectors' (their lines descend the geometry too), ws the
    general scorer's (E, XS: ext.py; E.n = 0 when there is none)."""
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
        if D.on:
            # the ball's unseen scatters (detectors.py): sigma_s x its
            # flux at D.k profile points of the walk, plus the scatters a
            # walk conditioned on scattering has beyond that, carried by
            # the straight flight it excludes (weight `excess` each)
            p_out = np.exp(-ss * d)
            excess = ss * p_out / (1.0 - p_out) * \
                T._decayed(wgt, sa, 0.0, d) / D.k
            for _ in range(D.k):
                r = draw_r(ph.prof_log_r, ph.prof_log1p_ell, ph.prof_f_unsc,
                           ph.prof_cdf, P.ball_R[i], sa / ss,
                           P.s[i], P.seed, i) * d / P.ball_R[i]
                px, py, pz = rng.isotropic(P.seed, i)
                DT.score(row, D, g, D.sig_t, ss * flux / D.k, x + r * px,
                         y + r * py, z + r * pz, dpth)
                xi = rng.rand(P.seed, i)
                tl = xi * d if sa * d < 1e-9 else \
                    -np.log1p(xi * np.expm1(-sa * d)) / sa
                px, py, pz = rng.isotropic(P.seed, i)
                DT.score(row, D, g, D.sig_t, excess, x + tl * px,
                         y + tl * py, z + tl * pz, dpth)
        if E.n:
            ivm = _iv(E, mat)
            EX.ball(row, E, L, X, g, P, XS, i, path, XS.prev[i], mat, sa,
                    ss, flux, x, y, z, d, s_cm, ivm, ph, ws)
            XS.time[i] += s_cm * ivm
            XS.prev_mat[i] = -1
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
            g, x, y, z, u, v, w, want_near, True, pth,
            L.has_surface or E.n > 0)
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
                if E.n:
                    _ext_cross(row, E, L, X, g, P, XS, i, k, _crossing_sign(
                        g, k, x, y, z, u, v, w) * wgt, last, last, last_mat,
                        XS.prev_mat[i], x, y, z, u, v, w, ws)
                row[T.LEAKAGE] += wgt
                P.status[i] = DEAD
                return
            turn = g.surf_bc[k] == G.BC_REFLECTIVE or \
                g.surf_bc[k] == G.BC_WHITE
            if E.n and turn:
                _ext_cross(row, E, L, X, g, P, XS, i, k, _crossing_sign(
                    g, k, x, y, z, u, v, w) * wgt, last, last, last_mat,
                    XS.prev_mat[i], x, y, z, u, v, w, ws)
            x, y, z, u, v, w = _boundary(g, k, P, i, x, y, z, u, v, w)
            if E.n and turn:
                _ext_cross(row, E, L, X, g, P, XS, i, k, _crossing_sign(
                    g, k, x, y, z, u, v, w) * wgt, last, last, last_mat,
                    last_mat, x, y, z, u, v, w, ws)
            w_in = wgt                  # currents use it (OpenMC: wgt_last)
            if g.surf_shift[k, 4] != 1.0:             # albedo
                row[T.LEAKAGE] += wgt * (1.0 - g.surf_shift[k, 4])
                wgt *= g.surf_shift[k, 4]
            if D.on and g.surf_bc[k] == G.BC_WHITE:
                _white_detectors(g, k, D, row, wgt, x, y, z, dpth)
            if g.surf_bc[k] == G.BC_PERIODIC and (L.has_surface or E.n):
                k2 = int(g.surf_shift[k, 3])         # as at a crossing
                to[:] = -1
                m_to = G._path(g, x + TINY * u, y + TINY * v, z + TINY * w,
                               to, univ)
                if L.has_surface:
                    T.score_surface(row, L, k2, _crossing_sign(
                        g, k2, x, y, z, u, v, w) * w_in, last, to, m_to,
                        scr)
                if E.n:
                    _ext_cross(row, E, L, X, g, P, XS, i, k2, _crossing_sign(
                        g, k2, x, y, z, u, v, w) * w_in, last, to, m_to,
                        last_mat, x, y, z, u, v, w, ws)
            if E.n:
                _ext_crossed(XS, i, last, last_mat)
            x, y, z = x + 2.0 * TINY * u, y + 2.0 * TINY * v, \
                z + 2.0 * TINY * w
            continue
        rescues = 0
        if L.has_surface or E.n:
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
            if E.n and (ph.mesh_cap or E.any_faces):   # ext.py's meshes
                room = min(room, EX.room(E, L, X, x, y, z, ph.mesh_cap)
                           - GUARD)
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
        if L.has_meshsurf or E.n:
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
            if L.has_meshsurf:
                T.score_mesh_faces(row, L, X, sa, wgt, x, y, z, u, v, w,
                                   reach)
            if E.n:
                ivm = _iv(E, mat)
                EX.track(row, E, L, X, g, P, XS, i, pth, XS.prev[i], mat,
                         sa, ss, wgt, x, y, z, u, v, w, step, reach, ivm, ws)
                XS.time[i] += step * ivm
            if XS.v_mu.size and np.isnan(XS.v_mu[i]) and sa > 0.0 and \
                    rng.rand(P.seed, i) < -np.expm1(-sa * step):
                XS.v_mu[i] = XS.mu_last[i]       # the analog one is absorbed
        wgt *= np.exp(-sa * step)
        x, y, z = x + step * u, y + step * v, z + step * w
        cnt[C_FLIGHT] += 1

        if scatter:
            if D.on:
                DT.score(row, D, g, D.sig_t, wgt, x, y, z, dpth)
            nu, nv, nw = rng.isotropic(P.seed, i)
            if E.n:
                XS.mu_last[i] = u * nu + v * nv + w * nw
                XS.ncoll[i] += 1
                XS.prev_mat[i] = -1
            u, v, w = nu, nv, nw
            fresh = True
            cnt[C_SCATTER] += 1
        else:
            cnt[C_CROSS] += 1
            bc = g.surf_bc[surf] if at_root else G.BC_TRANSMISSION
            if surf >= 0 and (L.has_surface or E.n):
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
                    if L.has_surface:
                        T.score_surface(row, L, surf, sg * wgt, pth, pth,
                                        mat, scr)
                    if E.n:
                        # MaterialFrom: OpenMC's material_last, which no
                        # cell search has updated (the one before this
                        # cell, or none after a collision)
                        _ext_cross(row, E, L, X, g, P, XS, i, surf, sg * wgt,
                                   pth, pth, mat, XS.prev_mat[i], x, y, z, u,
                                   v, w, ws)
                elif bc == G.BC_TRANSMISSION:
                    to[:] = -1
                    m_to = G._path(g, x + TINY * u, y + TINY * v,
                                   z + TINY * w, to, univ)
                    moved = False
                    for lv in range(lvl + 1):
                        if to[lv] != pth[lv]:
                            moved = True
                    if moved:
                        if L.has_surface:
                            T.score_surface(row, L, surf, sg * wgt, pth, to,
                                            m_to, scr)
                        if E.n:
                            _ext_cross(row, E, L, X, g, P, XS, i, surf,
                                       sg * wgt, pth, to, m_to, mat, x, y,
                                       z, u, v, w, ws)
                elif E.n:
                    # reflective / white: OpenMC scores the crossing
                    # twice, before the turn and after it (+w then -w,
                    # which cancel unless a filter tells the two
                    # directions apart); here the first, below the second
                    _ext_cross(row, E, L, X, g, P, XS, i, surf, sg * wgt,
                               pth, pth, mat, XS.prev_mat[i], x, y, z, u, v,
                               w, ws)
            elif surf < 0 and E.n:
                # a lattice wall: OpenMC scores every one for the crossing
                # tallies without a SurfaceFilter (CellFrom, MaterialFrom),
                # with its normal along the step taken, so current is +w;
                # sg is the wall's |cosine| here (geometry._descend)
                to[:] = -1
                m_to = G._path(g, x + TINY * u, y + TINY * v, z + TINY * w,
                               to, univ)
                EX.cross(row, E, L, X, g, P, XS, i, -1, wgt, pth, to, m_to,
                         mat, x, y, z, u, v, w, sg, ws)
            if bc == G.BC_VACUUM:
                row[T.LEAKAGE] += wgt
                P.status[i] = DEAD
                return
            if bc != G.BC_TRANSMISSION:
                x, y, z, u, v, w = _boundary(g, surf, P, i, x, y, z, u, v, w)
                if E.n and surf >= 0 and (bc == G.BC_REFLECTIVE or
                                          bc == G.BC_WHITE):
                    _ext_cross(row, E, L, X, g, P, XS, i, surf,
                               _crossing_sign(g, surf, x, y, z, u, v, w) *
                               wgt, pth, pth, mat, mat, x, y, z, u, v, w, ws)
                w_in = wgt
                if g.surf_shift[surf, 4] != 1.0:
                    # albedo: the weight not sent back counts as leakage,
                    # as in OpenMC (openmc_import._periodic_shifts); the
                    # periodic current below is scored with the weight
                    # that arrived (OpenMC's wgt_last, measured v1.6.1)
                    row[T.LEAKAGE] += wgt * (1.0 - g.surf_shift[surf, 4])
                    wgt *= g.surf_shift[surf, 4]
                if D.on and bc == G.BC_WHITE:
                    _white_detectors(g, surf, D, row, wgt, x, y, z, dpth)
                if bc == G.BC_PERIODIC and (L.has_surface or E.n):
                    # OpenMC scores a periodic crossing once, at the
                    # PARTNER plane, signed by the direction after the
                    # jump; cell-from is the cell left, cell the one
                    # arrived in (measured, v1.5)
                    k2 = int(g.surf_shift[surf, 3])
                    to[:] = -1
                    m_to = G._path(g, x + TINY * u, y + TINY * v,
                                   z + TINY * w, to, univ)
                    if L.has_surface:
                        T.score_surface(row, L, k2, _crossing_sign(
                            g, k2, x, y, z, u, v, w) * w_in, pth, to, m_to,
                            scr)
                    if E.n:
                        _ext_cross(row, E, L, X, g, P, XS, i, k2,
                                   _crossing_sign(g, k2, x, y, z, u, v, w) *
                                   w_in, pth, to, m_to, mat, x, y, z, u, v,
                                   w, ws)
            if E.n:
                _ext_crossed(XS, i, pth, mat)
            x, y, z = x + TINY * u, y + TINY * v, z + TINY * w

        if wgt < ph.w_cut * P.w0[i] and not (XS.v_mu.size and
                                             np.isnan(XS.v_mu[i])):
            # (roulette waits while the analog history a Mu filter needs
            # is still alive: without roulette the weights stay unbiased)
            if rng.rand(P.seed, i) < wgt / (ph.w_surv * P.w0[i]):
                wgt = ph.w_surv * P.w0[i]
            else:
                cnt[C_RKILL] += 1
                P.status[i] = DEAD
                return
    cnt[C_LOST] += 1
    P.status[i] = DEAD


@njit(cache=True)
def _white_detectors(g, k, D, row, wgt, x, y, z, dpth):
    """A white wall's re-emission seen by the point detectors
    (detectors.score_white), from just inside the wall."""
    nx, ny, nz = S.normal(g.surf_kind[k], g.surf_coef[k], x, y, z)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    px, py, pz = x + TINY * nx, y + TINY * ny, z + TINY * nz
    if G.locate(g, px, py, pz)[1] == G.LOST:
        nx, ny, nz = -nx, -ny, -nz                  # inward is the - side
        px, py, pz = x + TINY * nx, y + TINY * ny, z + TINY * nz
    DT.score_white(row, D, g, D.sig_t, wgt, px, py, pz, nx, ny, nz, dpth)


@njit(cache=True, parallel=True)
def _birth_detectors(g, D, xyz, wimg, lo, acc, n, nchunk):
    """Every source site lo .. lo + len(xyz) - 1 scored at the point
    detectors (detectors.score_birth), into its own chunk's row."""
    per = (n + nchunk - 1) // nchunk
    m = xyz.shape[0]
    for c in prange(nchunk):
        dpth = np.full(G.MAX_LEVELS, -1, np.int64)
        a = max(c * per, lo) - lo
        b = min(min(n, (c + 1) * per), lo + m) - lo
        for q in range(a, b):
            DT.score_birth(acc[c], D, g, D.sig_t, wimg[q], xyz[q, 0],
                           xyz[q, 1], xyz[q, 2], dpth)


@njit(cache=True)
def _set_mu(mu_last, i, carry):
    # (a store made in _advance's own prange body did not reach _track:
    # measured v1.6.1, numba's parfor lowering; a call keeps it)
    mu_last[i] = carry


@njit(cache=True, parallel=True)
def _advance(g, ph, L, X, D, P, acc, cnt, nchunk, E, XS):
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
        dpth = np.full(G.MAX_LEVELS, -1, np.int64)
        ws = EX.workspace()
        carry, have = 0.0, False
        for i in range(c * per, min(n, (c + 1) * per)):
            st = P.status[i]
            if st == MOVING or st == READY:
                if E.n and st == MOVING and have:
                    # OpenMC keeps a thread's last scatter cosine from one
                    # history to the next (it is never reset): a Mu filter
                    # sees the previous history's on unscattered flights.
                    # The same here, chunk by chunk (a run with a Mu filter
                    # has no balls, so each history runs to its end in one
                    # go; ext.py)
                    _set_mu(XS.mu_last, i, carry)
                _track(i, g, ph, L, X, D, P, acc[c], cnt[c], pth, last, to,
                       univ, scr, dpth, E, XS, ws)
                if E.n and P.status[i] == DEAD:
                    # the cosine OpenMC's (analog) history ends with: the
                    # one it was absorbed with, or the last if it leaked
                    carry, have = XS.mu_last[i], True
                    if XS.v_mu.size and not np.isnan(XS.v_mu[i]):
                        carry = XS.v_mu[i]


# ------------------------------------------------------------- the driver
def physics(problem, backend=None, r_star=3.0, r_max=None,
            mesh_rule="profile", w_cut=0.25, w_surv=1.0, k_max=None,
            need_profile=False):
    """The Physics and tally Layout a run uses (also used by
    checks/loop_breakdown.py, which times run() phase by phase).
    Returns (ph, L, r_max actually used, X): X is the tallies.MeshExtra,
    passed to the kernels as its own argument (Numba's parallel loop
    cannot take it inside Physics).  need_profile loads ball/profile.py's
    table even with no mesh (point detectors use it)."""
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
    if use_profile or (need_profile and backend is not None):
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


def _triggers_met(problem, accum):
    """Is every tally trigger met (openmc_import._triggers)?"""
    mean, se = accum.mean_se()
    for k, kind, thr, cols, ignore0 in problem.triggers:
        tl = problem.tallies[k]
        a = accum.layout.offset[k]
        nb = tl.nbins
        m = mean[a:a + T.NS * nb].reshape(nb, T.NS)
        e = se[a:a + T.NS * nb].reshape(nb, T.NS)
        for c in cols:
            if c >= T.NS:          # a derived column (tallies.ALL_SCORES)
                c = 2 if c == T.NS else None
                if c is None:
                    continue
            mu, sd = m[:, c], e[:, c]
            if kind == "variance":
                val = sd * sd
            elif kind == "std_dev":
                val = sd
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    val = np.where(mu != 0.0, sd / np.abs(mu), np.inf)
            if ignore0:
                val = val[mu != 0.0]
            if val.size and np.max(val) > thr:
                return False
    return True


def _particles(n):
    f = lambda: np.zeros(n)
    return Particles(f(), f(), f(), f(), f(), f(), f(),
                     np.zeros(n, np.uint64), np.zeros(n, np.int64),
                     np.zeros(n, np.bool_), np.zeros(n, np.int64),
                     np.zeros(n, np.int64), f(), f(), f(), f(), f(),
                     np.full((n, G.MAX_LEVELS), -1, np.int64))


def run(problem, particles=100_000, batches=20, seed=1, backend=None,
        r_star=3.0, r_max=None, mesh_rule="profile", w_cut=0.25, w_surv=1.0,
        nchunk=32, verbose=False, k_max=None, k_det=K_DET, triggers=True):
    """Solve a Problem (from openmc_import.load).

    backend      None for plain Monte Carlo, or a ball backend from ball/
    r_star       smallest ball used, in mean free paths
    r_max        biggest ball used (default: the backend's own limit)
    mesh_rule    'profile' or 'cap' (see tallies.py)
    k_max        most mesh points per ball under 'profile' (default K_MAX;
                 1 gives the one-point rule, for comparisons)
    k_det        profile points per ball for point detectors
    triggers     honour the model's tally triggers (when it turns them
                 on): run more batches, up to trigger_max_batches, until
                 every trigger is met, as OpenMC does; `batches` is then
                 the least.  settings['batches'] says how many ran.

    Returns tallies.Results: per-source-particle means and standard errors.
    """
    g = problem.geom
    dets = getattr(problem, "detectors", []) or []
    ext_t = [tl for tl in problem.tallies if getattr(tl, "ext", False)]
    if ext_t and EX.no_balls(ext_t):
        backend = None          # plain Monte Carlo (openmc_import warned)
    ph, L, r_max, X = physics(problem, backend, r_star, r_max, mesh_rule,
                              w_cut, w_surv, k_max,
                              need_profile=bool(dets) or bool(ext_t))
    m = problem.mats
    if ext_t:
        info = problem.ext_info
        at = [t for t, tl in enumerate(problem.tallies)
              if getattr(tl, "ext", False)]
        E, (cols_mesh, cols_cell, n_born) = EX.build(
            ext_t, L.offset[at], info["cell_univ"], len(problem.cell_ids),
            len(m.ids), info["iv"], info["occ"])
    else:
        E, (cols_mesh, cols_cell, n_born) = EX.empty()
    mu_carry = bool(E.n) and bool(np.any(E.f_type == EX.X_MU))
    if dets:
        img, img_det, cut = problem.det_images
        D = Detectors(True,
                      np.array([d[1] for d in dets], float).reshape(-1, 3),
                      np.array([d[2] for d in dets], float),
                      (m.sig_s + m.sig_a).astype(float), L.n_entries,
                      int(k_det), np.asarray(img, float),
                      np.asarray(img_det, np.int64), float(cut),
                      len(img) > len(dets) or
                      bool(np.any(g.surf_bc == G.BC_WHITE)))
    else:
        D = _no_detectors(L.n_entries, len(m.ids))
    accum = T.Accumulator(L, len(dets))
    cnt = np.zeros((nchunk, len(COUNTERS)), np.int64)
    t_track = t_backend = 0.0
    calls = 0
    redrawn = 0

    trig = getattr(problem, "trigger_settings", None) if triggers else None
    b, n_batches, trig_met = 0, batches, None
    while b < n_batches:
        P = _particles(particles)
        P.seed[:] = rng.streams(np.uint64(seed), b * particles, particles)
        acc = np.zeros((nchunk, L.n_entries + len(dets)))
        t0 = time.perf_counter()
        sites = problem.source.sample(particles, source_rng(seed, b))
        redrawn += sites.redrawn
        _birth(sites.xyz, sites.uvw, sites.wgt, sites.iso, P, cnt, nchunk)
        XS = _ext_state(particles, n_born, E.n > 0, mu_carry)
        if E.n:
            # the first history of each chunk: a scatter cosine drawn
            # uniformly (the rest carry the previous history's, _advance)
            XS.time[:] = sites.t
            XS.w_analog[:] = sites.wgt
            XS.mu_last[:] = np.random.default_rng(
                [int(seed), int(b), 0xE77]).uniform(-1.0, 1.0, particles)
            cells, _, _ = G.paths_many(g, sites.xyz)
            XS.prev[:] = cells
            if n_born:
                deep = cells[np.arange(particles),
                             np.maximum((cells >= 0).sum(axis=1) - 1, 0)]
                EX.born_bins(E, L, X, sites.xyz, deep, cols_mesh, cols_cell,
                             XS.born)
        if dets:
            for lo in range(0, particles, BIRTH_SLICE):
                hi = min(particles, lo + BIRTH_SLICE)
                wimg = problem.source.detector_weights(sites, D.img, lo, hi)
                _birth_detectors(g, D, sites.xyz[lo:hi], wimg, lo, acc,
                                 particles, nchunk)
        while True:
            _advance(g, ph, L, X, D, P, acc, cnt, nchunk, E, XS)
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
            print(f"batch {b + 1}/{n_batches}  leakage {mean[0]:.5f} "
                  f"+- {se[0]:.5f}", flush=True)
        b += 1
        if trig is not None and b >= batches and \
                (b - batches) % trig["interval"] == 0:
            trig_met = _triggers_met(problem, accum)
            n_batches = b if trig_met else min(
                trig["max_batches"], b + trig["interval"])

    counters = dict(zip(COUNTERS, cnt.sum(axis=0).tolist()))
    counters["backend_calls"] = calls
    counters["source_redrawn"] = redrawn
    timing = {"transport_s": t_track, "backend_s": t_backend}
    settings = dict(particles=particles, batches=b, seed=seed,
                    backend=getattr(backend, "name", None), r_star=r_star,
                    r_max=r_max, mesh_rule=mesh_rule,
                    k_max=ph.k_max if ph.mesh_profile else None, w_cut=w_cut,
                    w_surv=w_surv, nchunk=nchunk,
                    k_det=int(k_det) if dets else None,
                    triggers_met=trig_met)
    return accum.results(problem.tallies, counters, timing, settings, dets)
