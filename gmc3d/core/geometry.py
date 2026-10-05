"""Where is a particle, and how far is it from the nearest boundary?

Used by: transport.py (every flight and every ball step asks one of the
three questions below).
Built by: openmc_import.py (fills the `Geometry` arrays from an OpenMC model).
Uses: surfaces.py (the four per-surface kernels).

The geometry is OpenMC's constructive solid geometry, flattened into arrays
so Numba can walk it:

    surfaces   kind, coefficients and boundary condition of every surface
    cells      a region (a formula over half-spaces) and a fill: a material,
               nothing (void), another universe, or a lattice
    universes  a list of cells; the root universe is the whole problem
    lattices   rectangular arrays of universes

A point is located by descending from the root universe: find the cell that
contains it; if that cell holds a universe or lattice, move into that
universe's own coordinates (translation, rotation, or the lattice element's
centre) and repeat, until a cell holds a material.  Translations and
rotations are rigid, so distances measured at any level are true distances.

The three questions, all answered on that same descent:

    locate            which cell and material contain the point
    nearest_boundary  the radius of the biggest ball around the point that
                      stays inside the current cell at EVERY level -- the
                      distance field.  It is the minimum, over levels, of the
                      distance to every surface the cell's region mentions,
                      and to the lattice element's walls.  The region's
                      boundary is part of those surfaces, so this can only
                      undershoot the true distance, never overshoot.
    ray_boundary      how far a flight can go before it reaches any of those
                      same surfaces or walls -- the ordinary Monte Carlo
                      "distance to boundary".  Stopping at a surface that
                      turns out not to bound the region is harmless: the
                      particle is re-located and carries on.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from . import surfaces as S

INF = S.INF
MAX_LEVELS = 16          # deepest universe nesting followed
STACK = 64               # deepest region formula evaluated

# fill types of a cell
FILL_MATERIAL, FILL_VOID, FILL_UNIVERSE, FILL_LATTICE = 0, 1, 2, 3

# boundary conditions of a surface
BC_TRANSMISSION, BC_VACUUM, BC_REFLECTIVE = 0, 1, 2

# region formula tokens: a half-space is +(i+1) or -(i+1) for surface i;
# operators are large codes no surface index reaches
OP_AND, OP_OR, OP_NOT = 1 << 40, (1 << 40) + 1, (1 << 40) + 2

# what locate() returns as the material of a point outside every cell
LOST = -2
VOID = -1

Geometry = namedtuple("Geometry", [
    "surf_kind", "surf_coef", "surf_bc",                    # per surface
    "cell_tok_start", "cell_tok_len", "tokens",             # region formulas
    "cell_surf_start", "cell_surf_len", "cell_surfs",       # surfaces a cell uses
    "cell_fill_type", "cell_fill",                          # what fills it
    "cell_trans", "cell_rot",                               # fill transform
    "univ_cell_start", "univ_cell_len", "univ_cells",       # universes
    "lat_ll", "lat_pitch", "lat_shape", "lat_ndim",         # lattices
    "lat_univ_start", "lat_univs", "lat_outer",
    "root",                                                 # root universe
])


# --------------------------------------------------------------- regions
@njit(cache=True)
def in_cell(g, cell, x, y, z):
    """Is the point inside the cell's region?  Evaluates the region formula,
    stored in postfix order, with a small stack."""
    n = g.cell_tok_len[cell]
    if n == 0:
        return True                       # a cell with no region is everywhere
    stack = np.empty(STACK, np.bool_)
    top = 0
    s0 = g.cell_tok_start[cell]
    for k in range(s0, s0 + n):
        t = g.tokens[k]
        if t == OP_AND:
            top -= 1
            stack[top - 1] = stack[top - 1] and stack[top]
        elif t == OP_OR:
            top -= 1
            stack[top - 1] = stack[top - 1] or stack[top]
        elif t == OP_NOT:
            stack[top - 1] = not stack[top - 1]
        else:
            i = abs(t) - 1
            f = S.value(g.surf_kind[i], g.surf_coef[i], x, y, z)
            stack[top] = f > 0.0 if t > 0 else f < 0.0
            top += 1
    return stack[0]


# ------------------------------------------------------------- the descent
@njit(cache=True)
def _descend(g, x, y, z, u, v, w, want_near, want_ray):
    """Walk from the root to the material cell holding (x, y, z).

    Returns (cell, material, near, t_ray, ray_surface, ray_at_root):
      cell, material  the deepest cell and its material (VOID / LOST codes)
      near            nearest-boundary distance (if want_near), else INF
      t_ray           distance along (u, v, w) to the next surface or
                      lattice wall at any level (if want_ray), else INF
      ray_surface     the surface t_ray stops at (-1 for a lattice wall)
      ray_at_root     True if that surface belongs to the root universe,
                      the only place boundary conditions act
    """
    univ = g.root
    near, t_ray, ray_surf, ray_root = INF, INF, -1, False
    for level in range(MAX_LEVELS):
        # ---- which cell of this universe holds the point
        cell = -1
        c0 = g.univ_cell_start[univ]
        for k in range(c0, c0 + g.univ_cell_len[univ]):
            if in_cell(g, g.univ_cells[k], x, y, z):
                cell = g.univ_cells[k]
                break
        if cell < 0:
            return -1, LOST, 0.0, INF, -1, False

        # ---- distances to this cell's surfaces
        s0 = g.cell_surf_start[cell]
        for k in range(s0, s0 + g.cell_surf_len[cell]):
            i = g.cell_surfs[k]
            if want_near:
                d = S.nearest(g.surf_kind[i], g.surf_coef[i], x, y, z)
                if d < near:
                    near = d
            if want_ray:
                d = S.ray(g.surf_kind[i], g.surf_coef[i], x, y, z, u, v, w)
                if d < t_ray:
                    t_ray, ray_surf, ray_root = d, i, level == 0

        ft = g.cell_fill_type[cell]
        if ft == FILL_MATERIAL:
            return cell, g.cell_fill[cell], near, t_ray, ray_surf, ray_root
        if ft == FILL_VOID:
            return cell, VOID, near, t_ray, ray_surf, ray_root

        # ---- move into the fill's coordinates: p' = R (p - t), d' = R d
        x, y, z = (x - g.cell_trans[cell, 0], y - g.cell_trans[cell, 1],
                   z - g.cell_trans[cell, 2])
        R = g.cell_rot[cell]
        x, y, z = (R[0, 0] * x + R[0, 1] * y + R[0, 2] * z,
                   R[1, 0] * x + R[1, 1] * y + R[1, 2] * z,
                   R[2, 0] * x + R[2, 1] * y + R[2, 2] * z)
        u, v, w = (R[0, 0] * u + R[0, 1] * v + R[0, 2] * w,
                   R[1, 0] * u + R[1, 1] * v + R[1, 2] * w,
                   R[2, 0] * u + R[2, 1] * v + R[2, 2] * w)

        if ft == FILL_UNIVERSE:
            univ = g.cell_fill[cell]
            continue

        # ---- a lattice: find the element, then its centre is the origin
        lat = g.cell_fill[cell]
        ll, p, shp = g.lat_ll[lat], g.lat_pitch[lat], g.lat_shape[lat]
        three_d = g.lat_ndim[lat] == 3
        ix = int(np.floor((x - ll[0]) / p[0]))
        iy = int(np.floor((y - ll[1]) / p[1]))
        iz = int(np.floor((z - ll[2]) / p[2])) if three_d else 0
        valid = (0 <= ix < shp[0]) and (0 <= iy < shp[1]) and \
                (0 <= iz < shp[2])
        lx = x - (ll[0] + (ix + 0.5) * p[0])
        ly = y - (ll[1] + (iy + 0.5) * p[1])
        lz = z - (ll[2] + (iz + 0.5) * p[2]) if three_d else z
        if valid:
            # the element's walls bound the ball and the flight too
            hx, hy, hz = 0.5 * p[0], 0.5 * p[1], 0.5 * p[2]
            if want_near:
                d = min(hx - abs(lx), hy - abs(ly))
                if three_d:
                    d = min(d, hz - abs(lz))
                if d < near:
                    near = d
            if want_ray:
                d = _box_exit(lx, ly, lz, u, v, w, hx, hy, hz, three_d)
                if d < t_ray:
                    t_ray, ray_surf, ray_root = d, -1, False
            univ = g.lat_univs[g.lat_univ_start[lat] +
                               (iz * shp[1] + iy) * shp[0] + ix]
        else:
            if g.lat_outer[lat] < 0:
                return -1, LOST, 0.0, INF, -1, False
            # outside the lattice's array: the outer universe, which ends
            # where the array begins
            if want_near:
                d = _box_outside_distance(x, y, z, ll, p, shp, three_d)
                if d < near:
                    near = d
            if want_ray:
                d = _box_entry(x, y, z, u, v, w, ll, p, shp, three_d)
                if d < t_ray:
                    t_ray, ray_surf, ray_root = d, -1, False
            univ = g.lat_outer[lat]
        x, y, z = lx, ly, lz
    return -1, LOST, 0.0, INF, -1, False


@njit(cache=True, inline="always")
def _box_exit(lx, ly, lz, u, v, w, hx, hy, hz, three_d):
    """Distance from inside a centred box (half-widths h) to its wall."""
    t = INF
    if u > 0.0:
        t = min(t, (hx - lx) / u)
    elif u < 0.0:
        t = min(t, (-hx - lx) / u)
    if v > 0.0:
        t = min(t, (hy - ly) / v)
    elif v < 0.0:
        t = min(t, (-hy - ly) / v)
    if three_d:
        if w > 0.0:
            t = min(t, (hz - lz) / w)
        elif w < 0.0:
            t = min(t, (-hz - lz) / w)
    return max(t, 0.0)


@njit(cache=True)
def _box_outside_distance(x, y, z, ll, p, shp, three_d):
    """Distance from a point outside a lattice's array to the array."""
    dx = max(ll[0] - x, 0.0, x - (ll[0] + shp[0] * p[0]))
    dy = max(ll[1] - y, 0.0, y - (ll[1] + shp[1] * p[1]))
    dz = max(ll[2] - z, 0.0, z - (ll[2] + shp[2] * p[2])) if three_d else 0.0
    return np.sqrt(dx * dx + dy * dy + dz * dz)


@njit(cache=True)
def _box_entry(x, y, z, u, v, w, ll, p, shp, three_d):
    """Distance along a ray to where it enters a lattice's array (slab
    method), or INF if it misses."""
    t0, t1 = 0.0, INF
    for a in range(3 if three_d else 2):
        q = (x, y, z)[a]
        d = (u, v, w)[a]
        lo, hi = ll[a], ll[a] + shp[a] * p[a]
        if d == 0.0:
            if q < lo or q > hi:
                return INF
        else:
            ta, tb = (lo - q) / d, (hi - q) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return INF
    return t0 if t0 > 0.0 else INF


# ------------------------------------------------------- the three questions
@njit(cache=True)
def locate(g, x, y, z):
    """(cell, material) holding the point; material is VOID or LOST if so."""
    cell, mat, _, _, _, _ = _descend(g, x, y, z, 1.0, 0.0, 0.0, False, False)
    return cell, mat


@njit(cache=True)
def nearest_boundary(g, x, y, z):
    """(distance, cell, material): the distance field at the point."""
    cell, mat, near, _, _, _ = _descend(g, x, y, z, 1.0, 0.0, 0.0,
                                        True, False)
    return near, cell, mat


@njit(cache=True)
def ray_boundary(g, x, y, z, u, v, w):
    """(distance, surface, at_root, cell, material) for a flight."""
    cell, mat, _, t, s, root = _descend(g, x, y, z, u, v, w, False, True)
    return t, s, root, cell, mat


# ---------------------------------------------------- Python-side helpers
def locate_many(g, pts):
    """Locate an (n, 3) array of points; returns (cells, materials)."""
    return _locate_many(g, np.ascontiguousarray(pts, np.float64))


@njit(cache=True)
def _locate_many(g, pts):
    n = pts.shape[0]
    cells = np.empty(n, np.int64)
    mats = np.empty(n, np.int64)
    for i in range(n):
        cells[i], mats[i] = locate(g, pts[i, 0], pts[i, 1], pts[i, 2])
    return cells, mats


def nearest_many(g, pts):
    """Distance field at an (n, 3) array of points."""
    return _nearest_many(g, np.ascontiguousarray(pts, np.float64))


@njit(cache=True)
def _nearest_many(g, pts):
    n = pts.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i], _, _ = nearest_boundary(g, pts[i, 0], pts[i, 1], pts[i, 2])
    return out
