"""Scoring: where flux and absorption are recorded, and their error bars.

Used by: transport.py (calls `score_segment` after every flight,
`score_ball` / `score_ball_mesh` after every ball step and
`score_surface` at root-surface crossings).
Built from: the Tally list made by openmc_import.py.
Read by: output.py and checks/ (via `Results`).

Two kinds of event deposit flux:

  a flight     a straight segment.  Flux is its track length, weighted by
               the particle's weight, which decays as exp(-sigma_a * l)
               along the way (absorption is continuous; see transport.py).
               Over a stretch from l_a to l_b that gives
                   w * (exp(-sigma_a l_a) - exp(-sigma_a l_b)) / sigma_a.
  a ball step  a whole random walk inside a ball, summarised by its total
               path s.  Flux is w * (1 - exp(-sigma_a s)) / sigma_a.  The
               ball lies inside ONE cell at every universe level, so cell,
               material and total tallies get this exactly.  A mesh voxel
               is different: the walk's track is spread through the ball,
               and where exactly is not known from (mu, s) alone
               (research item R4).  Two rules are offered, chosen per run:
                 'profile' (default) the deposit is split equally over
                           K points of the ball's walk, each drawn from
                           where walks lay their flux (ball/profile.py),
                           and each point's voxel gets its share: exact
                           on average, full-size balls, some extra noise
                           (K grows with the ball's size over the voxel
                           size; see transport.py)
                 'cap'     the ball is shrunk to fit inside the voxel holding
                           its centre, so the deposit is exact (smaller balls)
               The old 'centre' rule (everything in the centre voxel) was
               removed in v1.1: it smeared every flux map.

A surface crossing scores current: the weight crossing, + if the particle
moves to the surface's positive side and - if to its negative side (OpenMC's
'current' score with a SurfaceFilter; on a vacuum boundary that is the
leakage through that surface, signed).  Only flights cross surfaces, so a
surface tally never depends on the ball backend.

Absorption, scatter and total are scored as sigma_a, sigma_s and sigma_t
times the flux deposit, the same track-length estimator OpenMC uses by
default.  Each is accumulated on its own (not derived from the flux
afterwards), because a bin that spans two materials has no single sigma.

Layout.  Every tally bin holds NS = 4 numbers, flux, absorption, scatter
and total (in that order, whatever the tally asked for), in one flat
array; entry 0 is the leakage (weight leaving through vacuum
boundaries).  Each thread chunk has its own row (`acc[chunk]`), so threads
never write the same memory and sums come out in a fixed order.  Means are
per source particle, as OpenMC's tallies are.

ERROR BARS, and why they are taken per CHUNK rather than per batch.  The
mean of a run uses every history, but an error bar is only as good as the
number of independent groups it is estimated from: a standard deviation
from G groups is itself uncertain by about 1 / sqrt(2 (G - 1)), and a figure
of merit, which goes as one over its square, by about sqrt(2 / G).  With the
40 batches a race uses that is 11% on the error bar and 24% on the figure of
merit, so two honest runs of the same thing can disagree by a factor of two
-- measured, 5 and 6 October 2026, on the slab.

The chunks inside a batch are disjoint sets of histories with independent
random streams, so each is an independent estimate in its own right.  Using
them as the groups turns 40 groups into nchunk x batches (1,280 at the
defaults) at no cost: the error bar's own uncertainty falls to 2% and the
figure of merit's to 4%.  The MEAN is unchanged to the last bit, since it is
still the total over the total number of histories.

With groups of unequal size (the last chunk of a batch can be short) the
per-history variance is estimated as

    sigma^2 = sum_g n_g (x_g - X)^2 / (G - 1),   SE(X) = sqrt(sigma^2 / N)

where x_g is group g's mean, n_g its size, N the total and X the overall
mean.  For equal groups this is exactly the standard error of the group
means, SD(x_g) / sqrt(G).  Only running sums are kept, so memory does not
grow with G.

`mean_se_batches` still gives the old batch-mean estimate, and
`checks/error_bars.py` compares the two against the spread of many
independent runs.

SEVERAL FILTERS (v1.3).  A tally's filters each give a bin, and the
combined bin is their weighted sum, the last filter varying fastest:
OpenMC's order, so our (bins, 4) array lines up with its tally.mean row
for row.  Along one flight the cell, material and energy bins are fixed,
so `_combos` works them out once and the mesh walk adds its voxel on top.
A cell filter looks at the cell at EVERY level of the particle's path (a
cell filled with a universe or lattice scores whatever happens inside
it), as OpenMC does; if it lists cells on two levels of one path, both
score.  A tally with a surface filter scores current at crossings, where
a cell or material filter sees the cell entered (on a vacuum boundary,
the cell left) and a cell-from filter the cell left -- partial currents.
These conventions were measured in OpenMC 0.16, 29 Oct 2026.

Mesh bins are numbered with x fastest, then y, then z -- OpenMC's order.
Every mesh is stored as its bin edges on each axis, so regular and
rectilinear meshes share one grid walk.  A 1D or 2D regular mesh has one
bin from -BIG to +BIG on each missing axis, as OpenMC's are unbounded
there.  Regular axes find a point's bin by division, rectilinear ones by
binary search.

CURVED MESHES (v1.5).  A CylindricalMesh has axes (r, phi, z) and a
SphericalMesh (r, theta, phi), first axis fastest, all relative to the
mesh's origin; phi runs from +x towards +y in [0, 2 pi), theta from +z
(OpenMC's conventions).  Their edges are not planes, so the flight walk is
different: from where the flight is, find the nearest point ahead where it
meets ANY edge surface (a cylinder or sphere for an r edge, a half-plane
through the z axis for a phi edge, a cone -- or the plane z = 0 at
theta = pi/2 -- for a theta edge, a plane for a z edge), deposit the piece
up to there in the bin of the piece's midpoint, and repeat.  A cut that is
not a real bin boundary only splits a piece in two inside one bin, so it
costs time, never accuracy.  The 'cap' rule's room is the distance to the
nearest edge surface (exact for each kind).
"""

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from numba import njit

VOLUME, SURFACE, MESHSURF, CURVED, EXT = 0, 1, 2, 3, 4
# tally kinds: track scores, surface crossings, mesh face crossings, track
# scores on a curved mesh (scored by its own kernel; see MeshExtra), and a
# tally for the general scorer (ext.py), which every kernel here skips.
# An ext tally's meshes get slots of their own after the tallies (kind -1,
# no entries), so ext.py reads them through the same mesh arrays.
F_CELL, F_CELLFROM, F_MAT, F_MESH, F_SURF, F_ENERGY = range(1, 7)
_FKIND = {"cell": F_CELL, "cellfrom": F_CELLFROM, "material": F_MAT,
          "mesh": F_MESH, "meshsurface": F_MESH, "surface": F_SURF,
          "energy": F_ENERGY}
MAX_FILTERS = 6                   # one of each kind at most
# Shortcuts for the common tallies (all of v1.2's), which skip `_combos`:
# no filter but a mesh; one cell filter on material / void cells only (it
# can match only the deepest cell); one material filter.
FAST_NONE, FAST_LEAF, FAST_MAT, GENERAL = 0, 1, 2, 3
MAX_COMBOS = 256                  # combined bins one event can score in:
                                  # a cell and a cell-from filter, each
                                  # matching at most every level (16 x 16)
SCORES = ("flux", "absorption", "scatter", "total")   # columns, in order
NS = len(SCORES)                  # entries per bin
# Scores reported beside SCORES (v1.6.1), as more columns of a result:
# nu-scatter is scatter (one group, no multiplication: openmc_import
# checks), inverse-velocity is scored by ext.py (its own copy of the
# tally's bins, flux x 1/v), and the fission family and decay-rate are 0
# (no material is fissile; OpenMC gives 0 too, measured 30 Oct 2026).
EXTRA_SCORES = ("nu-scatter", "inverse-velocity", "fission", "nu-fission",
                "kappa-fission", "prompt-nu-fission", "delayed-nu-fission",
                "decay-rate")
ALL_SCORES = SCORES + EXTRA_SCORES
LEAKAGE = 0                       # index of the leakage entry
INF = 1.0e300
BIG = 1.0e100                     # edge of a mesh's missing axis
MESH_RECT, MESH_CYL, MESH_SPH = 0, 1, 2
_MESH_TYPE = {"rect": MESH_RECT, "cyl": MESH_CYL, "sph": MESH_SPH}
TWO_PI = 2.0 * np.pi

# The compiled form the Numba kernels read.
Layout = namedtuple("Layout", [
    "kind", "offset",            # per tally: VOLUME / SURFACE, first entry
    "nf", "f_kind", "f_stride",  # per tally x filter: kind, bin stride
    "f_lut",                     # per tally x filter x cell / material /
                                 # surface index: that filter's bin, or -1
                                 # (an energy filter's one bin at [.., 0])
    "fast", "fast_lut",          # per tally: FAST_* shortcut; per tally x
                                 # cell (FAST_LEAF) or material (FAST_MAT):
                                 # the combined bin, or -1
    "mesh_on", "mesh_stride",    # per tally: has a mesh filter, its stride
    "mesh_ll", "mesh_dw", "mesh_dim",   # per tally x axis: first edge,
                                        # SMALLEST bin width, bin count
    "mesh_edges", "mesh_uniform",       # per tally x axis: edges, regular?
    "has_mesh", "has_surface",   # any tally of that kind (skip work if not:
    "has_meshsurf", "has_curved",   # a volume mesh, a surface, a mesh
                                    # surface, a curved volume mesh)
    "n_entries",                 # length of a row of acc
])

# What only the v1.5 kernels need (curved meshes, mesh surfaces, the ball
# rules), kept OUT of Layout on purpose: every array in Layout is paid for
# on every call that receives it, and score_segment makes several such
# calls per flight.  Seven more arrays in Layout made the scoring of an
# ordinary flight 40-65% slower even with no curved mesh (measured on cask
# and hexcore, v1.5; one array alone cost 14% on cask).
MeshExtra = namedtuple("MeshExtra", [
    "mesh_type", "mesh_org",     # per tally: MESH_*, origin (curved only)
    "mesh_box",                  # per tally: (lower, upper) corner of a
                                 # box holding the mesh (flights are
                                 # clipped to it)
    "mesh_cos", "mesh_sin",      # per tally x axis x edge: cos / sin of an
                                 # angular edge (curved meshes only)
    "mesh_h",                    # per tally: a voxel's smallest size in cm
                                 # (the 'profile' rule's point count)
    "mesh_ndim",                 # per tally: a MESHSURF mesh's axes
])


@njit(cache=True)
def scratch(levels):
    """Per-thread work space for the kernels: MAX_COMBOS combined bins,
    then room for one filter's matches (one per geometry level)."""
    return np.empty(MAX_COMBOS + levels, np.int64)


def compile_layout(tallies, n_cells, n_mats, n_surfs=1, leaf=None):
    """Tally list -> Layout (see `compile_layouts` for MeshExtra too)."""
    return compile_layouts(tallies, n_cells, n_mats, n_surfs, leaf)[0]


def compile_layouts(tallies, n_cells, n_mats, n_surfs=1, leaf=None):
    """Tally list -> (Layout, MeshExtra).  Combined bin b of tally t is at
    offset[t] + NS b, b = sum over filters of (filter bin) x stride, the
    last filter's stride 1 (OpenMC's order).  (A surface tally keeps the
    same layout: current, then unused zeros.)"""
    ext_mesh = [f for tl in tallies if getattr(tl, "ext", False)
                for f in tl.filters if f.edges is not None]
    for k, f in enumerate(ext_mesh):
        f.slot = len(tallies) + k
    nt = max(len(tallies) + len(ext_mesh), 1)
    kind = np.full(nt, -1, np.int64)
    offset = np.zeros(nt, np.int64)
    nf = np.zeros(nt, np.int64)
    f_kind = np.zeros((nt, MAX_FILTERS), np.int64)
    f_stride = np.zeros((nt, MAX_FILTERS), np.int64)
    f_lut = np.full((nt, MAX_FILTERS, max(n_cells, n_mats, n_surfs, 1)), -1,
                    np.int64)
    fast = np.full(nt, GENERAL, np.int64)
    fast_lut = np.full((nt, max(n_cells, n_mats, 1)), -1, np.int64)
    mesh_on = np.zeros(nt, np.bool_)
    mesh_stride = np.zeros(nt, np.int64)
    mesh_ll = np.zeros((nt, 3))
    mesh_dw = np.ones((nt, 3))
    mesh_dim = np.ones((nt, 3), np.int64)
    edges = [[np.array([-BIG, BIG])] * 3 for _ in range(nt)]
    mesh_uniform = np.ones((nt, 3), np.bool_)
    mesh_type = np.zeros(nt, np.int64)
    mesh_org = np.zeros((nt, 3))
    mesh_box = np.zeros((nt, 2, 3))
    mesh_h = np.ones(nt)
    mesh_ndim = np.full(nt, 3, np.int64)
    def put_mesh(t, f):
        for a, e in enumerate(f.edges):
            e = np.asarray(e, float)
            w = np.diff(e)
            edges[t][a] = e
            mesh_ll[t, a], mesh_dim[t, a] = e[0], e.size - 1
            mesh_dw[t, a] = w.min()
            mesh_uniform[t, a] = np.allclose(w, w[0], rtol=1e-12)
        mesh_type[t] = _MESH_TYPE[getattr(f, "mesh_type", "rect")]
        mesh_org[t] = getattr(f, "origin", None) if \
            getattr(f, "origin", None) is not None else 0.0
        mesh_box[t], mesh_h[t] = _box_and_size(mesh_type[t], mesh_org[t],
                                               edges[t])
        mesh_ndim[t] = getattr(f, "ndim", 3)

    for f in ext_mesh:
        put_mesh(f.slot, f)
    pos = 1                                       # entry 0 is leakage
    for t, tl in enumerate(tallies):
        fs = tl.filters
        if getattr(tl, "ext", False):
            kind[t], offset[t] = EXT, pos
            pos += NS * tl.nbins * (2 if "inverse-velocity" in tl.scores
                                    else 1)
            continue
        kind[t] = SURFACE if any(f.kind == "surface" for f in fs) \
            else MESHSURF if any(f.kind == "meshsurface" for f in fs) \
            else CURVED if any(f.kind == "mesh" and
                               getattr(f, "mesh_type", "rect") != "rect"
                               for f in fs) \
            else VOLUME
        offset[t] = pos
        nf[t] = len(fs)
        stride = 1
        for j in range(len(fs) - 1, -1, -1):     # last filter fastest
            f = fs[j]
            f_kind[t, j], f_stride[t, j] = _FKIND[f.kind], stride
            if f.kind in ("mesh", "meshsurface"):
                mesh_on[t], mesh_stride[t] = True, stride
                put_mesh(t, f)
            elif f.kind == "energy":
                f_lut[t, j, 0] = f.bins[0]
            else:
                f_lut[t, j, f.bins] = np.arange(f.nbins)
            stride *= f.nbins
        other = [j for j, f in enumerate(fs)
                 if f.kind not in ("mesh", "meshsurface")]
        track = kind[t] in (VOLUME, CURVED)
        if track and not other:
            fast[t] = FAST_NONE
        elif track and len(other) == 1:
            f = fs[other[0]]
            j = other[0]
            if f.kind == "material":
                fast[t] = FAST_MAT
            elif f.kind == "cell" and (leaf is None or
                                       all(leaf[b] for b in f.bins)):
                fast[t] = FAST_LEAF
            if fast[t] != GENERAL:
                fast_lut[t, f.bins] = np.arange(f.nbins) * f_stride[t, j]
        pos += NS * stride
    n_edge = max(e.size for row in edges for e in row)
    mesh_edges = np.full((nt, 3, n_edge), BIG)
    for t in range(nt):
        for a in range(3):
            mesh_edges[t, a, :edges[t][a].size] = edges[t][a]
    L = Layout(kind, offset, nf, f_kind, f_stride, f_lut, fast, fast_lut,
               mesh_on,
               mesh_stride, mesh_ll, mesh_dw, mesh_dim, mesh_edges,
               mesh_uniform,
               bool((mesh_on & ((kind == VOLUME) | (kind == CURVED))).any()),
               bool((kind == SURFACE).any()),
               bool((kind == MESHSURF).any()),
               bool((kind == CURVED).any()), pos)
    X = MeshExtra(mesh_type, mesh_org, mesh_box, np.cos(mesh_edges),
                  np.sin(mesh_edges), mesh_h, mesh_ndim)
    return L, X


def _box_and_size(mtype, org, e):
    """A mesh's bounding box ((2, 3): lower and upper corner) and its
    smallest voxel size in cm.  For a curved mesh the size uses the arc
    of the innermost ring at its mid radius: only a point count depends
    on it ('profile' rule), never an answer."""
    dmin = [np.diff(x).min() for x in e]
    if mtype == MESH_RECT:
        return np.array([[x[0] for x in e], [x[-1] for x in e]]), \
            min(dmin)
    r = e[0]
    rmax, rmid = r[-1], 0.5 * (r[0] + r[1])
    if mtype == MESH_CYL:
        lo = org + np.array([-rmax, -rmax, e[2][0]])
        hi = org + np.array([rmax, rmax, e[2][-1]])
        return np.array([lo, hi]), min(dmin[0], dmin[2], rmid * dmin[1])
    return np.array([org - rmax, org + rmax]), \
        min(dmin[0], rmid * dmin[1], rmid * dmin[2])


# ------------------------------------------------------------ the kernels
@njit(cache=True, inline="always")
def _decayed(w, sa, la, lb):
    """Weighted track length from l_a to l_b with weight w exp(-sa l)."""
    if sa * (lb - la) < 1.0e-12:
        return w * np.exp(-sa * la) * (lb - la)
    return w * np.exp(-sa * la) * (-np.expm1(-sa * (lb - la))) / sa


@njit(cache=True, inline="always")
def _add(row, at, flux, sa, ss):
    row[at] += flux
    row[at + 1] += sa * flux
    row[at + 2] += ss * flux
    row[at + 3] += (sa + ss) * flux


@njit(cache=True, inline="always")
def _bin(L, t, a, q):
    """Bin of coordinate q on axis a of mesh t, or -1 / n if outside."""
    n = L.mesh_dim[t, a]
    if L.mesh_uniform[t, a]:
        return int(np.floor((q - L.mesh_ll[t, a]) / L.mesh_dw[t, a]))
    e = L.mesh_edges[t, a]
    if q < e[0]:
        return -1
    if q >= e[n]:
        return n
    return np.searchsorted(e[:n + 1], q, side="right") - 1


@njit(cache=True, inline="always")
def _axis(L, t, a, q, d):
    """DDA set-up on one axis: (voxel index, its step, distance to the
    next voxel wall) for a ray at q going d."""
    n = L.mesh_dim[t, a]
    i = min(max(_bin(L, t, a, q), 0), n - 1)
    e = L.mesh_edges[t, a]
    if d > 0.0:
        return i, 1, (e[i + 1] - q) / d
    if d < 0.0:
        return i, -1, (e[i] - q) / d
    return i, 0, INF


@njit(cache=True, inline="always")
def _next_wall(L, t, a, i, s, q, d):
    """Distance from q (along d) to the far wall of bin i, stepping s."""
    e = L.mesh_edges[t, a]
    return (e[i + 1] - q) / d if s > 0 else (e[i] - q) / d


@njit(cache=True, inline="always")
def _fast(L, t, f, cell, mat):
    """Combined bin (without the mesh's part) of a FAST tally, or -1."""
    if f == FAST_NONE:
        return 0
    k = cell if f == FAST_LEAF else mat
    return L.fast_lut[t, k] if k >= 0 else -1


@njit(cache=True)
def _combos(L, t, path, frm, mat, surf, scr):
    """The combined bins (without the mesh's part) tally t scores an
    event in: written to scr[:n], n returned (0: the event is not in the
    tally).

    path / frm   the cell at each level, root first, -1 after the deepest:
                 where the event is (for a crossing: the cell entered) and,
                 for a crossing, the cell left
    mat          the material there (< 0 for void)
    surf         the surface crossed (crossings only)

    A cell filter can match at several levels at once (a filled cell and
    a cell inside it, both listed): OpenMC then scores every match, and so
    does this, which is why one event can have several combined bins.
    """
    scr[0] = 0
    n = 1
    m0 = MAX_COMBOS
    for j in range(L.nf[t]):
        k = L.f_kind[t, j]
        st = L.f_stride[t, j]
        if k == F_MESH:
            continue
        if k == F_CELL or k == F_CELLFROM:
            # The loop is written out twice on purpose.  Choosing the array
            # (`path if ... else frm`) or calling a helper made Numba count
            # references on every call: 0.3-0.7 us per call, more than all
            # the rest of the scoring (measured, v1.3).
            m = 0
            if k == F_CELL:
                for lv in range(path.size):
                    c = path[lv]
                    if c < 0:
                        break
                    b = L.f_lut[t, j, c]
                    if b >= 0:
                        scr[m0 + m] = b
                        m += 1
            else:
                for lv in range(frm.size):
                    c = frm[lv]
                    if c < 0:
                        break
                    b = L.f_lut[t, j, c]
                    if b >= 0:
                        scr[m0 + m] = b
                        m += 1
            if m == 0:
                return 0
            for q in range(n):
                base = scr[q]
                for a in range(m - 1, -1, -1):
                    scr[a * n + q] = base + st * scr[m0 + a]
            n *= m
            continue
        if k == F_MAT:
            b = L.f_lut[t, j, mat] if mat >= 0 else -1
        elif k == F_SURF:
            # (guarded: track events pass surf = -1, and a constant
            # negative index here made the whole function ~10x slower)
            b = L.f_lut[t, j, surf] if surf >= 0 else -1
        else:                                   # F_ENERGY
            b = L.f_lut[t, j, 0]
        if b < 0:
            return 0
        for q in range(n):
            scr[q] += st * b
    return n


@njit(cache=True)
def _mesh_segment(row, L, t, base, sa, ss, w, x, y, z, u, v, ww, length):
    """Spread one flight over the voxels it crosses (a 3D DDA: step from
    voxel wall to voxel wall, depositing each piece's track length).
    base is the entry of the tally's mesh bin 0 for this event's other
    filters."""
    dim = L.mesh_dim[t]
    ms = NS * L.mesh_stride[t]
    # clip the segment to the mesh box (slab method)
    t0, t1 = 0.0, length
    for a in range(3):
        q = x if a == 0 else (y if a == 1 else z)
        d = u if a == 0 else (v if a == 1 else ww)
        lo, hi = L.mesh_edges[t, a, 0], L.mesh_edges[t, a, dim[a]]
        if d == 0.0:
            if q < lo or q > hi:
                return
        else:
            ta, tb = (lo - q) / d, (hi - q) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
    if t0 >= t1:
        return
    # set up from the clipped start; wall distances shifted back by t0
    x0, y0, z0 = x + t0 * u, y + t0 * v, z + t0 * ww
    ix, sx, nx_ = _axis(L, t, 0, x0, u)
    iy, sy, ny_ = _axis(L, t, 1, y0, v)
    iz, sz, nz_ = _axis(L, t, 2, z0, ww)
    nx_ += t0
    ny_ += t0
    nz_ += t0
    tc = t0
    while tc < t1:
        te = min(nx_, ny_, nz_, t1)
        if te > tc:
            b = (iz * dim[1] + iy) * dim[0] + ix
            _add(row, base + ms * b, _decayed(w, sa, tc, te), sa, ss)
        tc = te
        if te >= t1:
            return
        if nx_ <= ny_ and nx_ <= nz_:
            ix += sx
            if ix < 0 or ix >= dim[0]:
                return
            nx_ = t0 + _next_wall(L, t, 0, ix, sx, x0, u)
        elif ny_ <= nz_:
            iy += sy
            if iy < 0 or iy >= dim[1]:
                return
            ny_ = t0 + _next_wall(L, t, 1, iy, sy, y0, v)
        else:
            iz += sz
            if iz < 0 or iz >= dim[2]:
                return
            nz_ = t0 + _next_wall(L, t, 2, iz, sz, z0, ww)


# ------------------------------------------------------- curved meshes
@njit(cache=True, inline="always")
def _coords(X, t, x, y, z):
    """A point in mesh t's own coordinates: (x, y, z) for a rectangular
    mesh, (r, phi, z) cylindrical, (r, theta, phi) spherical."""
    k = X.mesh_type[t]
    if k == MESH_RECT:
        return x, y, z
    dx = x - X.mesh_org[t, 0]
    dy = y - X.mesh_org[t, 1]
    dz = z - X.mesh_org[t, 2]
    phi = np.arctan2(dy, dx)
    if phi < 0.0:
        phi += TWO_PI
    rho2 = dx * dx + dy * dy
    if k == MESH_CYL:
        return np.sqrt(rho2), phi, dz
    r = np.sqrt(rho2 + dz * dz)
    c = dz / r if r > 0.0 else 1.0
    return r, np.arccos(min(max(c, -1.0), 1.0)), phi


@njit(cache=True)
def _point_bin(L, X, t, x, y, z):
    """Mesh bin (first axis fastest) of mesh t holding a point, or -1."""
    dim = L.mesh_dim[t]
    q0, q1, q2 = _coords(X, t, x, y, z)
    i0 = _bin(L, t, 0, q0)
    i1 = _bin(L, t, 1, q1)
    i2 = _bin(L, t, 2, q2)
    if 0 <= i0 < dim[0] and 0 <= i1 < dim[1] and 0 <= i2 < dim[2]:
        return (i2 * dim[1] + i1) * dim[0] + i0
    return -1


@njit(cache=True, inline="always")
def _first_root(a, b, c, lo):
    """Smallest t > lo with a t^2 + 2 b t + c = 0 (INF if none)."""
    if abs(a) < 1.0e-14:
        if abs(b) < 1.0e-300:
            return INF
        r = -c / (2.0 * b)
        return r if r > lo else INF
    disc = b * b - a * c
    if disc < 0.0:
        return INF
    sq = np.sqrt(disc)
    r1, r2 = (-b - sq) / a, (-b + sq) / a
    if r1 > r2:
        r1, r2 = r2, r1
    if r1 > lo:
        return r1
    return r2 if r2 > lo else INF


@njit(cache=True)
def _next_cut(L, X, t, px, py, pz, u, v, w, lo):
    """Distance (> lo) along the ray from p (relative to the origin) to
    the first point where it meets any edge surface of curved mesh t."""
    k = X.mesh_type[t]
    dim = L.mesh_dim[t]
    best = INF
    pp, pd = px * px + py * py, px * u + py * v
    dd = u * u + v * v
    if k == MESH_SPH:
        pp += pz * pz
        pd += pz * w
        dd += w * w
    # r edges: cylinders (about z) or spheres
    e = L.mesh_edges[t, 0]
    for i in range(dim[0] + 1):
        if e[i] > 0.0:
            best = min(best, _first_root(dd, pd, pp - e[i] * e[i], lo))
    # phi edges: half-planes through the z axis
    ap = 1 if k == MESH_CYL else 2
    for i in range(dim[ap] + 1):
        c, sn = X.mesh_cos[t, ap, i], X.mesh_sin[t, ap, i]
        den = c * v - sn * u                 # normal (-sin, cos, 0) . d
        if abs(den) < 1.0e-300:
            continue
        r = -(c * py - sn * px) / den
        if lo < r < best and c * (px + r * u) + sn * (py + r * v) >= 0.0:
            best = r
    if k == MESH_CYL:
        # z edges: planes
        if w != 0.0:
            e = L.mesh_edges[t, 2]
            for i in range(dim[2] + 1):
                r = (e[i] - pz) / w
                if lo < r < best:
                    best = r
        return best
    # theta edges: cones about z (the plane z = 0 at theta = pi/2)
    for i in range(dim[1] + 1):
        c = X.mesh_cos[t, 1, i]
        if abs(c) > 1.0 - 1.0e-12:           # theta 0 or pi: the z axis
            continue
        if abs(c) < 1.0e-12:
            if w != 0.0:
                r = -pz / w
                if lo < r < best:
                    best = r
            continue
        c2 = c * c
        a = w * w - c2 * dd
        b = pz * w - c2 * pd
        cc = pz * pz - c2 * pp
        # both nappes solve the quadratic; keep the root on this theta's
        r = _first_root(a, b, cc, lo)
        while r < best and (pz + r * w) * c < 0.0:   # other nappe
            r = _first_root(a, b, cc, r)
        best = min(best, r)
    return best


@njit(cache=True)
def _curved_segment(row, L, X, t, base, sa, ss, wgt, x, y, z, u, v, w,
                    length):
    """`_mesh_segment` for a cylindrical or spherical mesh: cut the flight
    at every edge surface it meets, each piece into the bin of its
    midpoint (see the docstring)."""
    ms = NS * L.mesh_stride[t]
    t0, t1 = 0.0, length
    for a in range(3):
        q = x if a == 0 else (y if a == 1 else z)
        d = u if a == 0 else (v if a == 1 else w)
        lo, hi = X.mesh_box[t, 0, a], X.mesh_box[t, 1, a]
        if d == 0.0:
            if q < lo or q > hi:
                return
        else:
            ta, tb = (lo - q) / d, (hi - q) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
    if t0 >= t1:
        return
    px = x - X.mesh_org[t, 0]
    py = y - X.mesh_org[t, 1]
    pz = z - X.mesh_org[t, 2]
    eps = 1.0e-10 * (1.0 + length)
    tc = t0
    while tc < t1:
        te = min(t1, _next_cut(L, X, t, px, py, pz, u, v, w, tc + eps))
        tm = 0.5 * (tc + te)
        b = _point_bin(L, X, t, x + tm * u, y + tm * v, z + tm * w)
        if b >= 0:
            _add(row, base + ms * b, _decayed(wgt, sa, tc, te), sa, ss)
        tc = te


@njit(cache=True)
def _curved_room(L, X, t, x, y, z):
    """Distance from a point to the nearest edge surface of curved mesh t
    (each kind's exact distance; see `mesh_room`)."""
    k = X.mesh_type[t]
    dim = L.mesh_dim[t]
    px = x - X.mesh_org[t, 0]
    py = y - X.mesh_org[t, 1]
    pz = z - X.mesh_org[t, 2]
    rho = np.sqrt(px * px + py * py)
    rad = rho if k == MESH_CYL else np.sqrt(rho * rho + pz * pz)
    room = INF
    e = L.mesh_edges[t, 0]
    for i in range(dim[0] + 1):
        if e[i] > 0.0:
            room = min(room, abs(rad - e[i]))
    ap = 1 if k == MESH_CYL else 2
    for i in range(dim[ap] + 1):
        c, sn = X.mesh_cos[t, ap, i], X.mesh_sin[t, ap, i]
        along = c * px + sn * py             # towards the half-plane
        room = min(room, abs(c * py - sn * px) if along >= 0.0 else rho)
    if k == MESH_CYL:
        e = L.mesh_edges[t, 2]
        for i in range(dim[2] + 1):
            room = min(room, abs(pz - e[i]))
        return room
    for i in range(dim[1] + 1):
        c, sn = X.mesh_cos[t, 1, i], X.mesh_sin[t, 1, i]
        if abs(c) > 1.0 - 1.0e-12:
            continue
        # the cone's generator in the point's meridian half-plane
        along = rho * sn + pz * c
        room = min(room, abs(rho * c - pz * sn) if along >= 0.0 else rad)
    return room


@njit(cache=True)
def score_segment(row, L, cell, path, mat, sa, ss, w, x, y, z, u, v, ww,
                  length, scr):
    """A flight of `length` from (x,y,z) along (u,v,ww), in cell `cell`
    (the deepest; `path` holds every level) and material `mat`, starting
    with weight w, in a material with absorption sa and scattering ss."""
    flux = _decayed(w, sa, 0.0, length)
    for t in range(L.kind.size):
        if L.kind[t] != VOLUME:
            continue
        f = L.fast[t]
        if f != GENERAL:
            c = _fast(L, t, f, cell, mat)
            n = 1 if c >= 0 else 0
        else:
            n = _combos(L, t, path, path, mat, -1, scr)
        for q in range(n):
            if L.fast[t] == GENERAL:
                c = scr[q]
            base = L.offset[t] + NS * c
            if L.mesh_on[t]:
                _mesh_segment(row, L, t, base, sa, ss, w, x, y, z, u, v, ww,
                              length)
            else:
                _add(row, base, flux, sa, ss)


@njit(cache=True)
def score_segment_curved(row, L, X, cell, path, mat, sa, ss, w, x, y, z, u,
                         v, ww, length, scr):
    """`score_segment` for the curved-mesh tallies (kind CURVED, which it
    skips), called only when there is one."""
    for t in range(L.kind.size):
        if L.kind[t] != CURVED:
            continue
        f = L.fast[t]
        if f != GENERAL:
            c = _fast(L, t, f, cell, mat)
            n = 1 if c >= 0 else 0
        else:
            n = _combos(L, t, path, path, mat, -1, scr)
        for q in range(n):
            if L.fast[t] == GENERAL:
                c = scr[q]
            _curved_segment(row, L, X, t, L.offset[t] + NS * c, sa, ss, w,
                            x, y, z, u, v, ww, length)

@njit(cache=True)
def score_surface(row, L, surf, signed_w, frm, to, mat_to, scr):
    """A flight crossing surface `surf` with weight +-w (sign: towards
    the positive side), leaving the cells `frm` for the cells `to` (in
    material mat_to).  On a vacuum boundary to = frm: OpenMC's cell filter
    sees the cell the particle leaves, as there is no other."""
    for t in range(L.kind.size):
        if L.kind[t] != SURFACE:
            continue
        n = _combos(L, t, to, frm, mat_to, surf, scr)
        for q in range(n):
            row[L.offset[t] + NS * scr[q]] += signed_w


@njit(cache=True)
def score_mesh_faces(row, L, X, sa, wgt, x, y, z, u, v, w, length):
    """Every mesh face a flight from (x, y, z) along (u, v, w) crosses in
    (0, length), in every mesh-surface tally: the weight there, w
    exp(-sa t), into the bin 'out of the element left' and 'in to the
    element entered' (just one where the flight leaves or enters the
    mesh).  OpenMC's MeshSurfaceFilter bins (see openmc_import.py)."""
    for t in range(L.kind.size):
        if L.kind[t] != MESHSURF:
            continue
        dim = L.mesh_dim[t]
        nb = 4 * X.mesh_ndim[t]
        at = L.offset[t]
        # clip to the mesh box, remembering which axis bounds each end
        t0, t1, a0, a1 = 0.0, length, -1, -1
        out = False
        for a in range(3):
            q = x if a == 0 else (y if a == 1 else z)
            d = u if a == 0 else (v if a == 1 else w)
            lo, hi = L.mesh_edges[t, a, 0], L.mesh_edges[t, a, dim[a]]
            if d == 0.0:
                if q < lo or q > hi:
                    out = True
            else:
                ta, tb = (lo - q) / d, (hi - q) / d
                if ta > tb:
                    ta, tb = tb, ta
                if ta > t0:
                    t0, a0 = ta, a
                if tb < t1:
                    t1, a1 = tb, a
        if out or t0 >= t1:
            continue
        x0, y0, z0 = x + t0 * u, y + t0 * v, z + t0 * w
        ix, sx, nx_ = _axis(L, t, 0, x0, u)
        iy, sy, ny_ = _axis(L, t, 1, y0, v)
        iz, sz, nz_ = _axis(L, t, 2, z0, w)
        if a0 >= 0:                              # enters the mesh at t0
            d = u if a0 == 0 else (v if a0 == 1 else w)
            e = (iz * dim[1] + iy) * dim[0] + ix
            f = 4 * a0 + (1 if d > 0.0 else 3)
            row[at + NS * (e * nb + f)] += wgt * np.exp(-sa * t0)
        nx_ += t0
        ny_ += t0
        nz_ += t0
        while True:
            te = min(nx_, ny_, nz_)
            if te >= t1:
                break
            e = (iz * dim[1] + iy) * dim[0] + ix
            if nx_ <= ny_ and nx_ <= nz_:
                a, s = 0, sx
            elif ny_ <= nz_:
                a, s = 1, sy
            else:
                a, s = 2, sz
            wt = wgt * np.exp(-sa * te)
            row[at + NS * (e * nb + 4 * a + (2 if s > 0 else 0))] += wt
            if a == 0:
                ix += sx
                if ix < 0 or ix >= dim[0]:
                    break                        # the far face, by rounding
                nx_ = t0 + _next_wall(L, t, 0, ix, sx, x0, u)
            elif a == 1:
                iy += sy
                if iy < 0 or iy >= dim[1]:
                    break
                ny_ = t0 + _next_wall(L, t, 1, iy, sy, y0, v)
            else:
                iz += sz
                if iz < 0 or iz >= dim[2]:
                    break
                nz_ = t0 + _next_wall(L, t, 2, iz, sz, z0, w)
            e = (iz * dim[1] + iy) * dim[0] + ix
            row[at + NS * (e * nb + 4 * a + (1 if s > 0 else 3))] += wt
        if not (0 <= ix < dim[0] and 0 <= iy < dim[1] and 0 <= iz < dim[2]):
            continue                             # already scored as out
        if a1 >= 0 and t1 < length:              # leaves the mesh at t1
            d = u if a1 == 0 else (v if a1 == 1 else w)
            e = (iz * dim[1] + iy) * dim[0] + ix
            f = 4 * a1 + (2 if d > 0.0 else 0)
            row[at + NS * (e * nb + f)] += wgt * np.exp(-sa * t1)


@njit(cache=True)
def score_ball(row, L, cell, path, mat, sa, ss, flux, scr):
    """A ball step that deposited `flux` in total: every track tally
    without a mesh (exact; the ball is inside one cell at every level).
    Mesh tallies are scored by `score_ball_mesh`, at points transport.py
    chooses."""
    for t in range(L.kind.size):
        if L.kind[t] != VOLUME or L.mesh_on[t]:
            continue
        if L.fast[t] != GENERAL:
            c = _fast(L, t, L.fast[t], cell, mat)
            if c >= 0:
                _add(row, L.offset[t] + NS * c, flux, sa, ss)
            continue
        n = _combos(L, t, path, path, mat, -1, scr)
        for q in range(n):
            _add(row, L.offset[t] + NS * scr[q], flux, sa, ss)


@njit(cache=True)
def score_ball_mesh(row, L, X, cell, path, mat, sa, ss, flux, x, y, z, scr):
    """`flux` of a ball into the voxel holding (x, y, z), in every mesh
    tally (and its other filters' bins for the ball's cell and material):
    the ball's centre under 'cap', a drawn point of its walk under
    'profile' (see transport.py)."""
    for t in range(L.kind.size):
        if not L.mesh_on[t] or (L.kind[t] != VOLUME and
                                L.kind[t] != CURVED):
            continue
        b = _point_bin(L, X, t, x, y, z)
        if b >= 0:
            b *= L.mesh_stride[t]
            if L.fast[t] != GENERAL:
                c = _fast(L, t, L.fast[t], cell, mat)
                if c >= 0:
                    _add(row, L.offset[t] + NS * (c + b), flux, sa, ss)
                continue
            n = _combos(L, t, path, path, mat, -1, scr)
            for q in range(n):
                _add(row, L.offset[t] + NS * (scr[q] + b), flux, sa, ss)


@njit(cache=True)
def mesh_room(L, X, x, y, z, faces_only=False):
    """Radius of the biggest ball around (x,y,z) that stays inside one voxel
    of every mesh tally ('cap' rule).  Outside a mesh, the ball must not
    reach the mesh at all.  (A curved mesh: the distance to its nearest
    edge surface, which covers both cases.)  faces_only: only the meshes of
    mesh-surface tallies, which cap balls under either rule (a ball must
    not carry a particle across a mesh face unseen)."""
    room = INF
    for t in range(L.kind.size):
        if not L.mesh_on[t] or (faces_only and L.kind[t] != MESHSURF):
            continue
        if L.kind[t] == CURVED:
            room = min(room, _curved_room(L, X, t, x, y, z))
            continue
        dim = L.mesh_dim[t]
        p = (x, y, z)
        outside2 = 0.0
        inside = INF
        for a in range(3):
            e = L.mesh_edges[t, a]
            lo, hi = e[0], e[dim[a]]
            if p[a] < lo:
                outside2 += (lo - p[a]) ** 2
            elif p[a] > hi:
                outside2 += (p[a] - hi) ** 2
            else:
                i = min(max(_bin(L, t, a, p[a]), 0), dim[a] - 1)
                inside = min(inside, p[a] - e[i], e[i + 1] - p[a])
        r = np.sqrt(outside2) if outside2 > 0.0 else inside
        room = min(room, r)
    return room


# ------------------------------------------------------------ statistics
@dataclass
class Results:
    """Per-source-particle means and standard errors, OpenMC's convention.

    leakage, leakage_se      weight leaving through vacuum boundaries
    tallies[name]            dict with 'mean' and 'se', arrays shaped
                             (bins, 4): columns flux, absorption, scatter,
                             total (SCORES), whatever the tally asked for;
                             'scores' says which were asked.  A surface
                             tally's current is column 0.
    counters                 work done (flights, balls, ...); see transport
    timing                   seconds spent in transport and in the backend
    leakage_se_batches       the same error bar estimated the old way, from
                             the batch means alone: kept so the two can be
                             compared, never used for an answer.  0.0 for a
                             reference code that reports only its own.
    groups                   how many independent groups the error bar came
                             from (chunks x batches; 0 if not ours)
    """
    leakage: float
    leakage_se: float
    tallies: dict
    counters: dict
    timing: dict
    settings: dict
    leakage_se_batches: float = 0.0
    groups: int = 0


def chunk_sizes(n, nchunk):
    """How many of n histories each of nchunk thread chunks gets.

    This MUST match the split `_advance` and `_birth` use in transport.py
    (`per = ceil(n / nchunk)`, then chunk c takes [c*per, min(n, (c+1)*per))),
    because the error bar treats each chunk as an independent group.  A
    trailing chunk can be short or empty.
    """
    per = (n + nchunk - 1) // nchunk
    lo = np.minimum(np.arange(nchunk) * per, n)
    hi = np.minimum(lo + per, n)
    return hi - lo


class Accumulator:
    """Running sums of the per-source-particle tallies.

    Two error bars are kept: one from the thread chunks (what `mean_se`
    returns, and what every answer uses) and one from the batch means (the
    old estimate, for comparison only).  See this file's docstring.
    """

    def __init__(self, layout, n_detectors=0):
        self.layout = layout
        n = layout.n_entries + n_detectors      # detectors after tallies
        self.s1 = np.zeros(n)             # batch-level, for the comparison
        self.s2 = np.zeros(n)
        self.batches = 0
        self.g_sum = np.zeros(n)          # sum_g n_g x_g  (= all the totals)
        self.g_sq = np.zeros(n)           # sum_g n_g x_g^2
        self.g_n = 0                      # N, histories so far
        self.groups = 0                   # G, non-empty chunks so far

    def add_batch(self, acc, n_particles):
        acc = np.asarray(acc)
        x = acc.sum(axis=0) / n_particles
        self.s1 += x
        self.s2 += x * x
        self.batches += 1

        sizes = chunk_sizes(n_particles, acc.shape[0])
        if sizes.sum() != n_particles:
            raise AssertionError("chunk sizes do not add up to the batch; "
                                 "tallies.chunk_sizes has drifted from "
                                 "transport.py's split")
        live = sizes > 0
        n_g = sizes[live, None]
        totals = acc[live]                         # n_g x_g, per group
        self.g_sum += totals.sum(axis=0)
        self.g_sq += (totals * totals / n_g).sum(axis=0)
        self.g_n += int(n_particles)
        self.groups += int(live.sum())

    def mean_se(self):
        """Mean per source particle, and its standard error from the chunk
        groups.  The mean is the total over the total histories, so it does
        not depend on how the histories were grouped."""
        mean = self.g_sum / self.g_n
        if self.groups < 2:
            return mean, np.full_like(mean, np.inf)
        var = np.maximum(self.g_sq - self.g_n * mean * mean, 0.0) \
            / (self.groups - 1)
        return mean, np.sqrt(var / self.g_n)

    def mean_se_batches(self):
        """The old estimate: the standard error of the batch means.  Kept
        only so `checks/error_bars.py` can compare the two."""
        b = self.batches
        mean = self.s1 / b
        var = np.maximum(self.s2 / b - mean * mean, 0.0) / max(b - 1, 1)
        return mean, np.sqrt(var)

    def results(self, tallies, counters, timing, settings, detectors=()):
        mean, se = self.mean_se()
        out = {}
        nc = len(ALL_SCORES)
        for t, tl in enumerate(tallies):
            a = self.layout.offset[t]
            nb = tl.nbins
            m, e = np.zeros((nb, nc)), np.zeros((nb, nc))
            m[:, :NS] = mean[a:a + NS * nb].reshape(nb, NS)
            e[:, :NS] = se[a:a + NS * nb].reshape(nb, NS)
            m[:, NS], e[:, NS] = m[:, 2], e[:, 2]          # nu-scatter
            if "inverse-velocity" in tl.scores:
                c = a + NS * nb                            # ext.py's copy
                m[:, NS + 1] = mean[c:c + NS * nb].reshape(nb, NS)[:, 0]
                e[:, NS + 1] = se[c:c + NS * nb].reshape(nb, NS)[:, 0]
            out[tl.name or f"tally{tl.id}"] = {
                "id": tl.id, "filter": tl.filter, "bin_ids": tl.bin_ids,
                "mesh": tl.mesh, "scores": tl.scores,
                "mesh_type": getattr(tl, "mesh_type", "rect"),
                "edges": tl.edges,
                "filters": [(f.kind, f.nbins) for f in tl.filters],
                "surface_type": bool(getattr(tl, "surface_type", False)),
                "mean": m, "se": e}
        base = self.layout.n_entries
        for j, (name, xyz, r0) in enumerate(detectors):
            mu, sd = np.zeros((1, len(ALL_SCORES))), \
                np.zeros((1, len(ALL_SCORES)))
            mu[0, 0], sd[0, 0] = mean[base + j], se[base + j]
            out[name] = {"id": None, "filter": "detector", "bin_ids": [],
                         "mesh": None, "scores": ["flux"],
                         "mesh_type": "rect", "edges": None, "filters": [],
                         "point": list(xyz), "r0": r0, "mean": mu,
                         "se": sd}
        _, se_b = self.mean_se_batches()
        return Results(float(mean[LEAKAGE]), float(se[LEAKAGE]), out,
                       counters, timing, settings,
                       float(se_b[LEAKAGE]), int(self.groups))
