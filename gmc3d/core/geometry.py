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
    lattices   rectangular or hexagonal arrays of universes

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
                      and to the lattice element's walls (rectangular or
                      hexagonal).  The region's
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
BC_TRANSMISSION, BC_VACUUM, BC_REFLECTIVE, BC_WHITE, BC_PERIODIC = \
    0, 1, 2, 3, 4

# lattice types: hexagonal lattices come in two orientations, named (as
# in OpenMC) by the axis the flats face: 'y' has flat tops, 'x' flat sides
LAT_RECT, LAT_HEX_Y, LAT_HEX_X = 0, 1, 2
SQ3, SQ75 = np.sqrt(3.0), np.sqrt(0.75)

# region formula tokens: a half-space is +(i+1) or -(i+1) for surface i;
# operators are large codes no surface index reaches
OP_AND, OP_OR, OP_NOT = 1 << 40, (1 << 40) + 1, (1 << 40) + 2

# what locate() returns as the material of a point outside every cell
LOST = -2
VOID = -1

Geometry = namedtuple("Geometry", [
    "surf_kind", "surf_coef", "surf_bc",                    # per surface
    "surf_shift",       # periodic surface: the jump to its partner plane
    "surf_angle",       # ... or, if not 0, the turn about z onto it
    "cell_tok_start", "cell_tok_len", "tokens",             # region formulas
    "cell_surf_start", "cell_surf_len", "cell_surfs",       # surfaces a cell uses
    "cell_fill_type", "cell_fill",                          # what fills it
    "cell_trans", "cell_rot",                               # fill transform
    "cell_dist", "cell_off", "lat_off", "leaf_mat",         # distributed
    "univ_cell_start", "univ_cell_len", "univ_cells",       # universes
    "lat_ll", "lat_pitch", "lat_shape", "lat_ndim",         # lattices
    "lat_type", "lat_rings",
    "lat_univ_start", "lat_univs", "lat_outer",
    "root",                                                 # root universe
])
# A lattice's arrays:
#   rectangular  lat_ll lower-left corner, lat_pitch the element sizes,
#                lat_shape (nx, ny, nz)
#   hexagonal    lat_ll (centre x, centre y, bottom z), lat_pitch
#                (pitch, pitch, axial pitch), lat_rings the number of rings
#                n, lat_shape (2n - 1, 2n - 1, nz): a square array over
#                OpenMC's two skewed indices (shifted by n - 1), of which
#                only the elements within n - 1 rings of the centre exist
# lat_univs holds the universe of every entry (-1 where none exists).
#
# Distributed materials (a cell with one material per instance, OpenMC's
# 'distribmat'): every material or void cell instance in the problem has a
# number, its place in OpenMC's own instance order (Geometry.determine_paths:
# cells in universe order, lattice elements in index order, outer universes
# not counted).  That number is the sum, down the path, of cell_off[cell]
# at every level and lat_off[entry] at every lattice level, so the descent
# keeps it as a running total.  A cell with cell_dist set takes its
# material from leaf_mat[number] (VOID for a void instance).


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
def _descend(g, x, y, z, u, v, w, want_near, want_ray, path):
    """Walk from the root to the material cell holding (x, y, z).

    path (MAX_LEVELS long) receives the cell at each level, root first,
    then -1 after the deepest (what a tally's cell filter looks at: OpenMC
    matches a cell at any level).

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
    inst = 0
    for level in range(MAX_LEVELS):
        # ---- which cell of this universe holds the point
        cell = -1
        c0 = g.univ_cell_start[univ]
        for k in range(c0, c0 + g.univ_cell_len[univ]):
            if in_cell(g, g.univ_cells[k], x, y, z):
                cell = g.univ_cells[k]
                break
        if cell < 0:
            path[level] = -1
            return -1, LOST, 0.0, INF, -1, False
        path[level] = cell
        if level + 1 < MAX_LEVELS:
            path[level + 1] = -1
        inst += g.cell_off[cell]

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
            m = g.leaf_mat[inst] if g.cell_dist[cell] else g.cell_fill[cell]
            return cell, m, near, t_ray, ray_surf, ray_root
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

        # ---- a lattice: find the element, then its centre is the origin.
        # Outside the array OpenMC fills every would-be element with the
        # outer universe in that element's own coordinates, so the walls
        # bound the flight and the ball there too.
        lat = g.cell_fill[cell]
        valid, k, x, y, z = _element(g, lat, x, y, z)
        if want_near:
            d = _wall_near(g, lat, x, y, z)
            if d < near:
                near = d
        if want_ray:
            d = _wall_exit(g, lat, x, y, z, u, v, w)
            if d < t_ray:
                t_ray, ray_surf, ray_root = d, -1, False
        if valid:
            univ = g.lat_univs[g.lat_univ_start[lat] + k]
            inst += g.lat_off[g.lat_univ_start[lat] + k]
        elif g.lat_outer[lat] >= 0:
            univ = g.lat_outer[lat]
        else:
            path[level + 1 if level + 1 < MAX_LEVELS else level] = -1
            return -1, LOST, 0.0, INF, -1, False
    return -1, LOST, 0.0, INF, -1, False


# ------------------------------------------------------------- lattices
@njit(cache=True)
def _element(g, lat, x, y, z):
    """The lattice element holding (x, y, z) (in the lattice's frame).

    Returns (valid, k, lx, ly, lz): whether the element exists, its entry
    in the lattice's universe list, and the point relative to its centre.
    A hexagonal element is the one whose centre is nearest, found among
    four candidates exactly as OpenMC does (HexLattice.find_element)."""
    ll, p, shp = g.lat_ll[lat], g.lat_pitch[lat], g.lat_shape[lat]
    iz, lz = 0, z
    if g.lat_ndim[lat] == 3:
        iz = int(np.floor((z - ll[2]) / p[2]))
        lz = z - (ll[2] + (iz + 0.5) * p[2])
    t = g.lat_type[lat]
    if t == LAT_RECT:
        ix = int(np.floor((x - ll[0]) / p[0]))
        iy = int(np.floor((y - ll[1]) / p[1]))
        lx = x - (ll[0] + (ix + 0.5) * p[0])
        ly = y - (ll[1] + (iy + 0.5) * p[1])
        valid = (0 <= ix < shp[0]) and (0 <= iy < shp[1]) and \
                (0 <= iz < shp[2])
        return valid, (iz * shp[1] + iy) * shp[0] + ix, lx, ly, lz
    pp = p[0]
    X, Y = x - ll[0], y - ll[1]
    if t == LAT_HEX_Y:
        i1 = int(np.floor(X / (SQ75 * pp)))
        i2 = int(np.floor((Y - X / SQ3) / pp))
    else:
        i1 = int(np.floor(-(Y - X * SQ3) / (SQ3 * pp)))
        i2 = int(np.floor(Y / (SQ75 * pp)))
    best, a, b, lx, ly = INF, 0, 0, 0.0, 0.0
    for c in range(4):
        ca, cb = i1 + (c & 1), i2 + (c >> 1)
        if t == LAT_HEX_Y:
            dx, dy = X - SQ75 * pp * ca, Y - (0.5 * ca + cb) * pp
        else:
            dx, dy = X - (ca + 0.5 * cb) * pp, Y - SQ75 * pp * cb
        d = dx * dx + dy * dy
        if d < best:
            best, a, b, lx, ly = d, ca, cb, dx, dy
    n = g.lat_rings[lat]
    ring = max(abs(a), abs(b), abs(a + b))
    valid = ring < n and 0 <= iz < shp[2]
    return valid, (iz * shp[1] + b + n - 1) * shp[0] + a + n - 1, lx, ly, lz


@njit(cache=True, inline="always")
def _wall_near(g, lat, lx, ly, lz):
    """Distance from a point inside an element (relative to its centre) to
    the element's walls: exact, the element being convex."""
    p = g.lat_pitch[lat]
    t = g.lat_type[lat]
    if t == LAT_RECT:
        d = min(0.5 * p[0] - abs(lx), 0.5 * p[1] - abs(ly))
    else:
        h = 0.5 * p[0]
        if t == LAT_HEX_Y:                   # flats facing +-y
            a1, a2, a3 = ly, SQ75 * lx + 0.5 * ly, SQ75 * lx - 0.5 * ly
        else:                                # flats facing +-x
            a1, a2, a3 = lx, 0.5 * lx + SQ75 * ly, 0.5 * lx - SQ75 * ly
        d = h - max(abs(a1), abs(a2), abs(a3))
    if g.lat_ndim[lat] == 3:
        d = min(d, 0.5 * p[2] - abs(lz))
    return max(d, 0.0)


@njit(cache=True, inline="always")
def _slab_exit(q, s, h):
    """Distance along a line (position q, speed s) to |q| = h, from inside."""
    if s > 0.0:
        return (h - q) / s
    if s < 0.0:
        return (-h - q) / s
    return INF


@njit(cache=True, inline="always")
def _wall_exit(g, lat, lx, ly, lz, u, v, w):
    """Distance along (u, v, w) from inside an element to its wall."""
    p = g.lat_pitch[lat]
    t = g.lat_type[lat]
    if t == LAT_RECT:
        d = min(_slab_exit(lx, u, 0.5 * p[0]), _slab_exit(ly, v, 0.5 * p[1]))
    else:
        h = 0.5 * p[0]
        if t == LAT_HEX_Y:
            d = min(_slab_exit(ly, v, h),
                    _slab_exit(SQ75 * lx + 0.5 * ly, SQ75 * u + 0.5 * v, h),
                    _slab_exit(SQ75 * lx - 0.5 * ly, SQ75 * u - 0.5 * v, h))
        else:
            d = min(_slab_exit(lx, u, h),
                    _slab_exit(0.5 * lx + SQ75 * ly, 0.5 * u + SQ75 * v, h),
                    _slab_exit(0.5 * lx - SQ75 * ly, 0.5 * u - SQ75 * v, h))
    if g.lat_ndim[lat] == 3:
        d = min(d, _slab_exit(lz, w, 0.5 * p[2]))
    return max(d, 0.0)


# ------------------------------------------------------- the three questions
@njit(cache=True)
def locate(g, x, y, z):
    """(cell, material) holding the point; material is VOID or LOST if so."""
    cell, mat, _, _, _, _ = _descend(g, x, y, z, 1.0, 0.0, 0.0, False, False,
                                     np.empty(MAX_LEVELS, np.int64))
    return cell, mat


@njit(cache=True)
def nearest_boundary(g, x, y, z):
    """(distance, cell, material): the distance field at the point."""
    cell, mat, near, _, _, _ = _descend(g, x, y, z, 1.0, 0.0, 0.0,
                                        True, False,
                                        np.empty(MAX_LEVELS, np.int64))
    return near, cell, mat


@njit(cache=True)
def ray_boundary(g, x, y, z, u, v, w):
    """(distance, surface, at_root, cell, material) for a flight."""
    cell, mat, _, t, s, root = _descend(g, x, y, z, u, v, w, False, True,
                                        np.empty(MAX_LEVELS, np.int64))
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


def paths_many(g, pts):
    """For an (n, 3) array of points: the cell and the universe at every
    level from the root down (padded with -1), and the deepest material.
    Used to check source constraints, which OpenMC applies to a cell or
    universe at ANY level of the point's path, and to the material at the
    deepest one."""
    return _paths_many(g, np.ascontiguousarray(pts, np.float64))


@njit(cache=True)
def _paths_many(g, pts):
    n = pts.shape[0]
    cells = np.full((n, MAX_LEVELS), -1, np.int64)
    univs = np.full((n, MAX_LEVELS), -1, np.int64)
    mats = np.full(n, LOST, np.int64)
    for i in range(n):
        mats[i] = _path(g, pts[i, 0], pts[i, 1], pts[i, 2], cells[i],
                        univs[i])
    return cells, univs, mats


@njit(cache=True)
def _path(g, x, y, z, cells, univs):
    """The descent of `_descend` for a point only, recording each level's
    universe and cell.  Returns the deepest material (VOID / LOST)."""
    univ = g.root
    inst = 0
    for level in range(MAX_LEVELS):
        univs[level] = univ
        cell = -1
        c0 = g.univ_cell_start[univ]
        for k in range(c0, c0 + g.univ_cell_len[univ]):
            if in_cell(g, g.univ_cells[k], x, y, z):
                cell = g.univ_cells[k]
                break
        if cell < 0:
            return LOST
        cells[level] = cell
        inst += g.cell_off[cell]
        ft = g.cell_fill_type[cell]
        if ft == FILL_MATERIAL:
            return g.leaf_mat[inst] if g.cell_dist[cell] else g.cell_fill[cell]
        if ft == FILL_VOID:
            return VOID
        x, y, z = (x - g.cell_trans[cell, 0], y - g.cell_trans[cell, 1],
                   z - g.cell_trans[cell, 2])
        R = g.cell_rot[cell]
        x, y, z = (R[0, 0] * x + R[0, 1] * y + R[0, 2] * z,
                   R[1, 0] * x + R[1, 1] * y + R[1, 2] * z,
                   R[2, 0] * x + R[2, 1] * y + R[2, 2] * z)
        if ft == FILL_UNIVERSE:
            univ = g.cell_fill[cell]
            continue
        lat = g.cell_fill[cell]
        valid, k, x, y, z = _element(g, lat, x, y, z)
        if valid:
            univ = g.lat_univs[g.lat_univ_start[lat] + k]
            inst += g.lat_off[g.lat_univ_start[lat] + k]
        elif g.lat_outer[lat] >= 0:
            univ = g.lat_outer[lat]
        else:
            return LOST
    return LOST
