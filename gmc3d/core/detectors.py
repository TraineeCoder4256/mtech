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

WALLS (v1.6.1).  A reflective, white or periodic boundary sends some of
the uncollided flux to P along paths that are not straight lines.
    * reflective (planes) and periodic walls: such a path, unfolded
      through the walls, is a straight line to an IMAGE of P -- P mirrored
      in the walls, or moved by the periodic jumps, as often as the path
      meets them.  `images` lists the images within IMAGE_MFP mean free
      paths (of the thinnest material) of the geometry; each emission
      follows the line to every image near enough, reflecting, jumping
      and turning at the walls as a particle would (with each wall's
      albedo), and scores it if the line really ends at P.  Each path is
      counted once, by the one image it unfolds to.  Truncation leaves out
      paths longer than IMAGE_MFP mean free paths: a relative error below
      exp(-IMAGE_MFP) where the paths cross material (void stretches are
      not attenuated, so a void-filled box between reflecting walls would
      be truncated harder; the import warns).
    * white walls: a particle reaching one is sent back in a new
      direction drawn from the cosine law, so the wall is a new emission
      point with angular density cos(theta) / pi about its inward normal
      (transport.py scores it there, `score_white`); a line that meets a
      white wall carries nothing past it.
A curved reflective wall (a mirror sphere or cylinder) has no such images
and is refused.

SOURCES THAT ARE NOT ISOTROPIC (v1.6.1).  A birth's direction is drawn
from its source's angular law, of density p(Omega) per steradian, so the
birth scores w 4 pi p(Omega_P) exp(-tau) / (4 pi d^2), Omega_P the
direction to P (sources.SourceSampler.detector_weights).  A monodirectional
beam, or a file's sites with their fixed directions, has density 0 towards
almost every point: its uncollided flux is a line, and the point sees only
what scatters out of it, which is right.

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
from . import surfaces as S

TINY = 1.0e-8            # cm, as transport.TINY
MAX_CROSSINGS = 100_000  # per line, a guard
FOUR_PI = 4.0 * np.pi
IMAGE_MFP = 25.0         # images further than this (in mfp of the
                         # thinnest material) from the geometry are dropped
MAX_IMAGES = 20_000      # per detector


@njit(cache=True)
def transmission(g, sig_t, x, y, z, px, py, pz, path, walls, tx, ty, tz):
    """(exp(-tau), d, sigma_t at the end) along the straight line from
    (x, y, z) to the point (px, py, pz) -- the detector or, with walls,
    one of its images -- following reflective and periodic walls on the
    way (times their albedo) and requiring the line to end at the
    detector (tx, ty, tz).  exp(-tau) = 0 if the line meets a vacuum or
    white boundary, leaves the geometry, or (walls) ends elsewhere."""
    dx, dy, dz = px - x, py - y, pz - z
    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    if d <= 0.0:
        cell, mat = G.locate(g, px, py, pz)
        return 1.0, 0.0, sig_t[mat] if mat >= 0 else 0.0
    u, v, w = dx / d, dy / d, dz / d
    tau, left, fac = 0.0, d, 1.0
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
            cx, cy, cz = cx + left * u, cy + left * v, cz + left * w
            break
        tau += st * t
        left -= t
        bc = g.surf_bc[surf] if surf >= 0 and lvl == 0 else \
            G.BC_TRANSMISSION
        if bc == G.BC_VACUUM or bc == G.BC_WHITE:
            return 0.0, d, st
        cx, cy, cz = cx + t * u, cy + t * v, cz + t * w
        if bc == G.BC_REFLECTIVE:
            nx, ny, nz = S.normal(g.surf_kind[surf], g.surf_coef[surf],
                                  cx, cy, cz)
            nn = nx * nx + ny * ny + nz * nz
            dot = (u * nx + v * ny + w * nz) / nn
            u, v, w = u - 2.0 * dot * nx, v - 2.0 * dot * ny, \
                w - 2.0 * dot * nz
            fac *= g.surf_shift[surf, 4]
        elif bc == G.BC_PERIODIC:
            a = g.surf_angle[surf]
            if a != 0.0:
                ca, sa = np.cos(a), np.sin(a)
                cx, cy = ca * cx - sa * cy, sa * cx + ca * cy
                u, v = ca * u - sa * v, sa * u + ca * v
            else:
                cx, cy, cz = (cx + g.surf_shift[surf, 0],
                              cy + g.surf_shift[surf, 1],
                              cz + g.surf_shift[surf, 2])
            fac *= g.surf_shift[surf, 4]
        cx, cy, cz = cx + TINY * u, cy + TINY * v, cz + TINY * w
        left -= TINY
        if left <= 0.0:
            break
    else:
        return 0.0, d, st
    if walls:
        ex, ey, ez = cx - tx, cy - ty, cz - tz
        if ex * ex + ey * ey + ez * ez > (1e-6 * (1.0 + d)) ** 2:
            return 0.0, d, st          # this line unfolds to another image
    return fac * np.exp(-tau), d, st


@njit(cache=True, inline="always")
def _kernel(D, j, tr, d, st):
    """exp(-tau) / (4 pi d^2), or the r0 average closer than r0."""
    r0 = D.r0[j]
    if d < r0:
        if st * r0 > 1e-9:
            return tr * 3.0 * (-np.expm1(-st * r0)) / (FOUR_PI * st *
                                                       r0 ** 3)
        return tr * 3.0 / (FOUR_PI * r0 * r0)
    if d > 0.0:
        return tr / (FOUR_PI * d * d)
    return 0.0


@njit(cache=True)
def score(row, D, g, sig_t, wgt, x, y, z, path):
    """Add the next-event contribution of an isotropic emission of weight
    wgt at (x, y, z) to every detector (entries D.base + j of row)."""
    for m in range(D.img.shape[0]):
        j = D.img_det[m]
        px, py, pz = D.img[m, 0], D.img[m, 1], D.img[m, 2]
        if D.walls:
            ex, ey, ez = px - x, py - y, pz - z
            if ex * ex + ey * ey + ez * ez > D.img_cut * D.img_cut:
                continue
        tr, d, st = transmission(g, sig_t, x, y, z, px, py, pz, path,
                                 D.walls, D.pos[j, 0], D.pos[j, 1],
                                 D.pos[j, 2])
        if tr > 0.0:
            row[D.base + j] += wgt * _kernel(D, j, tr, d, st)


@njit(cache=True)
def score_birth(row, D, g, sig_t, wimg, x, y, z, path):
    """`score` for a source site, whose weight towards image m is wimg[m]
    (sources.SourceSampler.detector_weights: its weight times 4 pi times
    its source's angular density in the direction the line to that image
    starts in)."""
    for m in range(D.img.shape[0]):
        if wimg[m] == 0.0:
            continue
        j = D.img_det[m]
        px, py, pz = D.img[m, 0], D.img[m, 1], D.img[m, 2]
        if D.walls:
            ex, ey, ez = px - x, py - y, pz - z
            if ex * ex + ey * ey + ez * ez > D.img_cut * D.img_cut:
                continue
        tr, d, st = transmission(g, sig_t, x, y, z, px, py, pz, path,
                                 D.walls, D.pos[j, 0], D.pos[j, 1],
                                 D.pos[j, 2])
        if tr > 0.0:
            row[D.base + j] += wimg[m] * _kernel(D, j, tr, d, st)


@njit(cache=True)
def score_white(row, D, g, sig_t, wgt, x, y, z, nx, ny, nz, path):
    """A particle of weight wgt sent back by a white wall at (x, y, z)
    with unit inward normal (nx, ny, nz): a new emission with density
    cos(theta) / pi, i.e. weight 4 wgt cos(theta) in `score`'s form."""
    for m in range(D.img.shape[0]):
        j = D.img_det[m]
        px, py, pz = D.img[m, 0], D.img[m, 1], D.img[m, 2]
        ex, ey, ez = px - x, py - y, pz - z
        dd = np.sqrt(ex * ex + ey * ey + ez * ez)
        if dd <= 0.0 or (D.walls and dd > D.img_cut):
            continue
        c = (ex * nx + ey * ny + ez * nz) / dd
        if c <= 0.0:
            continue
        tr, d, st = transmission(g, sig_t, x, y, z, px, py, pz, path,
                                 D.walls, D.pos[j, 0], D.pos[j, 1],
                                 D.pos[j, 2])
        if tr > 0.0:
            row[D.base + j] += 4.0 * wgt * c * _kernel(D, j, tr, d, st)


# ------------------------------------------------------------- images
def wall_maps(model, surfs, geom):
    """The affine maps p -> A p + b by which a straight line, unfolded
    through one wall, continues: one per reflective plane (its mirror)
    and per periodic surface (its jump or turn).  Raises ValueError for a
    curved reflective wall."""
    maps = []
    for i, s in enumerate(surfs):
        bc = s.boundary_type
        if bc == "reflective":
            if geom.surf_kind[i] not in (S.X_PLANE, S.Y_PLANE, S.Z_PLANE,
                                         S.PLANE):
                raise ValueError(
                    f"surface {s.id} is a curved reflective wall: a point "
                    f"detector's uncollided flux reflected in a curved "
                    f"mirror has no image points to follow")
            from .openmc_import import _plane
            n, d = _plane(s)
            maps.append((np.eye(3) - 2.0 * np.outer(n, n), 2.0 * d * n))
        elif bc == "periodic":
            a = geom.surf_angle[i]
            if a != 0.0:
                c, sn = np.cos(a), np.sin(a)
                maps.append((np.array([[c, -sn, 0.0], [sn, c, 0.0],
                                       [0.0, 0.0, 1.0]]), np.zeros(3)))
            else:
                maps.append((np.eye(3), np.asarray(geom.surf_shift[i, :3],
                                                   float).copy()))
    return maps


def images(point, maps, box, cut):
    """The point and its images under every composition of the wall maps
    (each used forwards and backwards), kept while within `cut` cm of the
    box (lower, upper) that holds the geometry."""
    lo, hi = box
    inv = [(A.T, -A.T @ b) for A, b in maps]      # A is orthogonal
    gens = maps + inv
    p0 = np.asarray(point, float)
    out = [p0]
    seen = {tuple(np.round(p0, 6))}
    front = [p0]
    while front:
        nxt = []
        for p in front:
            for A, b in gens:
                q = A @ p + b
                key = tuple(np.round(q, 6))
                if key in seen:
                    continue
                gap = np.maximum(np.maximum(lo - q, q - hi), 0.0)
                if np.sqrt(np.sum(gap ** 2)) > cut:
                    continue
                seen.add(key)
                out.append(q)
                nxt.append(q)
                if len(out) > MAX_IMAGES:
                    raise ValueError(
                        f"a point detector has more than {MAX_IMAGES:,} "
                        f"images within {IMAGE_MFP:g} mean free paths "
                        f"of the geometry (reflecting walls of a large, "
                        f"thin problem): too many lines to follow")
        front = nxt
    return np.array(out)
