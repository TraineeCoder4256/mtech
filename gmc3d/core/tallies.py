"""Scoring: where flux and absorption are recorded, and their error bars.

Used by: transport.py (calls `score_segment` after every flight and
`score_ball` after every ball step).
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
               and where exactly is not known (research item R4).  Two
               rules are offered, chosen per run:
                 'cap'     the ball is shrunk to fit inside the voxel holding
                           its centre, so the deposit is exact (smaller balls)
                 'centre'  all of it goes to the voxel holding the centre
                           (full-size balls; the mesh profile is smeared)

Absorption is scored as sigma_a times the flux deposit, the same
track-length estimator OpenMC uses by default.

Layout.  Every tally bin holds two numbers, flux then absorption, in one
flat array; entry 0 is the leakage (weight leaving through vacuum
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

Mesh bins are numbered with x fastest, then y, then z -- OpenMC's order.
"""

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from numba import njit

TOTAL, CELL, MATERIAL, MESH = 0, 1, 2, 3
_KIND = {"total": TOTAL, "cell": CELL, "material": MATERIAL, "mesh": MESH}
SCORES = ("flux", "absorption")
LEAKAGE = 0                       # index of the leakage entry
INF = 1.0e300

# The compiled form the Numba kernels read.
Layout = namedtuple("Layout", [
    "kind", "offset",            # per tally: filter kind, first entry
    "cell_bin", "mat_bin",       # per tally x cell / material: bin or -1
    "mesh_ll", "mesh_dw", "mesh_dim",   # per tally (zeros if not a mesh)
    "n_entries",                 # length of a row of acc
])


def compile_layout(tallies, n_cells, n_mats):
    """Tally list -> Layout.  Bin b of tally t is at offset[t] + 2 b."""
    nt = max(len(tallies), 1)
    kind = np.zeros(nt, np.int64)
    offset = np.zeros(nt, np.int64)
    cell_bin = np.full((nt, max(n_cells, 1)), -1, np.int64)
    mat_bin = np.full((nt, max(n_mats, 1)), -1, np.int64)
    mesh_ll = np.zeros((nt, 3))
    mesh_dw = np.ones((nt, 3))
    mesh_dim = np.ones((nt, 3), np.int64)
    pos = 1                                       # entry 0 is leakage
    for t, tl in enumerate(tallies):
        kind[t] = _KIND[tl.filter]
        offset[t] = pos
        if tl.filter == "total":
            nb = 1
        elif tl.filter == "cell":
            nb = len(tl.bins)
            cell_bin[t, tl.bins] = np.arange(nb)
        elif tl.filter == "material":
            nb = len(tl.bins)
            mat_bin[t, tl.bins] = np.arange(nb)
        else:
            ll, ur, dim = tl.mesh
            mesh_ll[t], mesh_dim[t] = ll, dim
            mesh_dw[t] = (np.asarray(ur) - ll) / dim
            nb = int(np.prod(dim))
        pos += 2 * nb
    if not tallies:
        kind[0] = -1
    return Layout(kind, offset, cell_bin, mat_bin, mesh_ll, mesh_dw,
                  mesh_dim, pos)


# ------------------------------------------------------------ the kernels
@njit(cache=True, inline="always")
def _decayed(w, sa, la, lb):
    """Weighted track length from l_a to l_b with weight w exp(-sa l)."""
    if sa * (lb - la) < 1.0e-12:
        return w * np.exp(-sa * la) * (lb - la)
    return w * np.exp(-sa * la) * (-np.expm1(-sa * (lb - la))) / sa


@njit(cache=True, inline="always")
def _add(row, at, flux, sa):
    row[at] += flux
    row[at + 1] += sa * flux


@njit(cache=True, inline="always")
def _axis(q, d, lo, dw, n):
    """DDA set-up on one axis: (voxel index, its step, distance to the
    next voxel wall, distance between walls) for a ray at q going d."""
    i = min(max(int(np.floor((q - lo) / dw)), 0), n - 1)
    if d > 0.0:
        return i, 1, (lo + (i + 1) * dw - q) / d, dw / d
    if d < 0.0:
        return i, -1, (lo + i * dw - q) / d, -dw / d
    return i, 0, INF, INF


@njit(cache=True)
def _mesh_segment(row, L, t, sa, w, x, y, z, u, v, ww, length):
    """Spread one flight over the voxels it crosses (a 3D DDA: step from
    voxel wall to voxel wall, depositing each piece's track length)."""
    ll, dw, dim = L.mesh_ll[t], L.mesh_dw[t], L.mesh_dim[t]
    # clip the segment to the mesh box (slab method)
    t0, t1 = 0.0, length
    for a in range(3):
        q = x if a == 0 else (y if a == 1 else z)
        d = u if a == 0 else (v if a == 1 else ww)
        lo, hi = ll[a], ll[a] + dim[a] * dw[a]
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
    ix, sx, nx_, dx = _axis(x + t0 * u, u, ll[0], dw[0], dim[0])
    iy, sy, ny_, dy = _axis(y + t0 * v, v, ll[1], dw[1], dim[1])
    iz, sz, nz_, dz = _axis(z + t0 * ww, ww, ll[2], dw[2], dim[2])
    nx_ += t0
    ny_ += t0
    nz_ += t0
    base = L.offset[t]
    tc = t0
    while tc < t1:
        te = min(nx_, ny_, nz_, t1)
        if te > tc:
            b = (iz * dim[1] + iy) * dim[0] + ix
            _add(row, base + 2 * b, _decayed(w, sa, tc, te), sa)
        tc = te
        if te >= t1:
            return
        if nx_ <= ny_ and nx_ <= nz_:
            ix += sx
            nx_ += dx
            if ix < 0 or ix >= dim[0]:
                return
        elif ny_ <= nz_:
            iy += sy
            ny_ += dy
            if iy < 0 or iy >= dim[1]:
                return
        else:
            iz += sz
            nz_ += dz
            if iz < 0 or iz >= dim[2]:
                return


@njit(cache=True)
def score_segment(row, L, cell, mat, sa, w, x, y, z, u, v, ww, length):
    """A flight of `length` from (x,y,z) along (u,v,ww) in `cell`/`mat`,
    starting with weight w, in a material with absorption sa."""
    flux = _decayed(w, sa, 0.0, length)
    for t in range(L.kind.size):
        k = L.kind[t]
        if k == TOTAL:
            _add(row, L.offset[t], flux, sa)
        elif k == CELL:
            b = L.cell_bin[t, cell]
            if b >= 0:
                _add(row, L.offset[t] + 2 * b, flux, sa)
        elif k == MATERIAL:
            if mat >= 0:
                b = L.mat_bin[t, mat]
                if b >= 0:
                    _add(row, L.offset[t] + 2 * b, flux, sa)
        elif k == MESH:
            _mesh_segment(row, L, t, sa, w, x, y, z, u, v, ww, length)


@njit(cache=True)
def score_ball(row, L, cell, mat, sa, flux, x, y, z):
    """A ball step centred at (x,y,z) that deposited `flux` in total."""
    for t in range(L.kind.size):
        k = L.kind[t]
        if k == TOTAL:
            _add(row, L.offset[t], flux, sa)
        elif k == CELL:
            b = L.cell_bin[t, cell]
            if b >= 0:
                _add(row, L.offset[t] + 2 * b, flux, sa)
        elif k == MATERIAL:
            b = L.mat_bin[t, mat]
            if b >= 0:
                _add(row, L.offset[t] + 2 * b, flux, sa)
        elif k == MESH:
            ll, dw, dim = L.mesh_ll[t], L.mesh_dw[t], L.mesh_dim[t]
            ix = int(np.floor((x - ll[0]) / dw[0]))
            iy = int(np.floor((y - ll[1]) / dw[1]))
            iz = int(np.floor((z - ll[2]) / dw[2]))
            if 0 <= ix < dim[0] and 0 <= iy < dim[1] and 0 <= iz < dim[2]:
                b = (iz * dim[1] + iy) * dim[0] + ix
                _add(row, L.offset[t] + 2 * b, flux, sa)


@njit(cache=True)
def mesh_room(L, x, y, z):
    """Radius of the biggest ball around (x,y,z) that stays inside one voxel
    of every mesh tally ('cap' rule).  Outside a mesh, the ball must not
    reach the mesh at all."""
    room = INF
    for t in range(L.kind.size):
        if L.kind[t] != MESH:
            continue
        ll, dw, dim = L.mesh_ll[t], L.mesh_dw[t], L.mesh_dim[t]
        p = (x, y, z)
        outside2 = 0.0
        inside = INF
        for a in range(3):
            lo, hi = ll[a], ll[a] + dim[a] * dw[a]
            if p[a] < lo:
                outside2 += (lo - p[a]) ** 2
            elif p[a] > hi:
                outside2 += (p[a] - hi) ** 2
            else:
                i = min(int(np.floor((p[a] - lo) / dw[a])), dim[a] - 1)
                inside = min(inside, p[a] - (lo + i * dw[a]),
                             lo + (i + 1) * dw[a] - p[a])
        r = np.sqrt(outside2) if outside2 > 0.0 else inside
        room = min(room, r)
    return room


# ------------------------------------------------------------ statistics
@dataclass
class Results:
    """Per-source-particle means and standard errors, OpenMC's convention.

    leakage, leakage_se      weight leaving through vacuum boundaries
    tallies[name]            dict with 'mean' and 'se', arrays shaped
                             (bins, 2): column 0 flux, column 1 absorption
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

    def __init__(self, layout):
        self.layout = layout
        n = layout.n_entries
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

    def results(self, tallies, counters, timing, settings):
        mean, se = self.mean_se()
        out = {}
        for t, tl in enumerate(tallies):
            a = self.layout.offset[t]
            nb = {"total": 1, "cell": len(tl.bins),
                  "material": len(tl.bins)}.get(
                tl.filter, int(np.prod(tl.mesh[2])) if tl.mesh else 1)
            out[tl.name or f"tally{tl.id}"] = {
                "id": tl.id, "filter": tl.filter, "bin_ids": tl.bin_ids,
                "mesh": tl.mesh, "scores": tl.scores,
                "mean": mean[a:a + 2 * nb].reshape(nb, 2),
                "se": se[a:a + 2 * nb].reshape(nb, 2)}
        _, se_b = self.mean_se_batches()
        return Results(float(mean[LEAKAGE]), float(se[LEAKAGE]), out,
                       counters, timing, settings,
                       float(se_b[LEAKAGE]), int(self.groups))
