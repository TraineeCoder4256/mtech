"""The general tally scorer (v1.6.1): every tally the fast kernels of
tallies.py cannot express.

Used by: openmc_import.py (`needs_ext` decides; `ExtTally` describes),
transport.py (calls `track`, `ball`, `cross` and `room`), tallies.py (the
results).

tallies.py scores the common tallies with kernels tuned for speed: at most
one filter of each kind, one mesh, a fixed set of filters.  Everything
else comes here -- one slower, general path, so that adding a filter never
slows the common tallies (a run with no such tally never calls into this
file):

    filters      Universe, CellInstance, Distribcell, MaterialFrom,
                 CellBorn, MeshBorn, MeshMaterial, Particle, Polar,
                 Azimuthal, SphericalHarmonics, Mu, MuSurface, Collision,
                 Time, SpatialLegendre, Zernike, ZernikeRadial,
                 Energyout and Legendre (scatter moments), Weight, plus the
                 tallies.py kinds (Cell, CellFrom, Material, Mesh, Surface,
                 MeshSurface, Energy) in any number and combination: two
                 meshes, a CellFrom on a track tally, a Surface with a
                 Mesh, a MeshSurface on a curved mesh or with other filters

Every filter's bins and weights follow OpenMC 0.16's `get_all_bins`
(src/tallies/filter_*.cpp), read from its source on 30 Oct 2026.  An event
matches a list of (bin, weight) per filter; the tally scores the product
of the weights in every combination, as OpenMC does.

EVENTS.  Three kinds, as in tallies.py:
    track    a flight, or a ball step.  A flight is cut into pieces at
             every edge of the tally's meshes and every time bin edge;
             each piece's mesh and time bins are its midpoint's, exact
             because a piece lies inside one bin of each.  The polynomial
             filters (SpatialLegendre, Zernike) weight a piece by their
             value at ONE point drawn in it with the flux's own density
             along it (exp(-sigma_a l)): exact on average, OpenMC's
             collision estimator having the same mean.
    cross    a crossing, for a tally OpenMC scores at crossings (a Surface
             filter, or CellFrom / MaterialFrom with flux or current):
             current +-w (signed by the surface's side; +w at a lattice
             wall, which counts too), flux by the surface-crossing
             estimator w / |mu| with OpenMC's grazing rule.  Reflective
             walls are seen twice (before and after the turn), periodic
             ones once at the partner plane.
    faces    the mesh faces a flight crosses (a MeshSurface filter): each
             face found where consecutive pieces lie in different
             elements.

BALLS.  A ball step is one track event with the ball's total flux, unless
the tally needs a position or a time inside it (meshes, the polynomial
filters, Time): then K_EXT points of the walk share the flux, each drawn by
ball/profile.py with the path length T at which it lies (so its time, t0 +
T / v), exact on average.  Filters on the DIRECTION along the walk (Polar,
Azimuthal, SphericalHarmonics, Mu, Collision) cannot be scored inside a
ball -- the walk's directions and collisions are not seen -- so a run with
such a track tally uses no balls (plain Monte Carlo; openmc_import warns).
A MeshSurface tally shrinks balls to one element of its mesh, as in
tallies.py.

PARTICLE STATE kept for these filters (transport.py, only when an ext
tally exists): the time (speed from the material's inverse velocity, or
OpenMC's default for its group, `default_inverse_velocity`), the number of
collisions (scatters: under implicit capture a live particle has had no
other), the cosine of the last scatter, the cells and material before the
last surface crossing (CellFrom / MaterialFrom: OpenMC's cell_last and
material_last), and the birth bins (CellBorn: the deepest cell, MeshBorn:
the mesh element of the birth point), drawn at birth.  OpenMC never resets
the last scatter cosine between histories, so a Mu filter sees the
previous history's on an unscattered flight; transport._advance carries it
the same way, from the analog history the weighted walk stands for.

Curved-mesh faces (MeshSurface): out-crossings are filed as OpenMC files
them; in-crossings by the face actually crossed, which OpenMC 0.16 does
not do (GAPS.md part 2b).
"""

from collections import namedtuple

import numpy as np
from numba import njit

from . import geometry as G
from . import rng
from . import tallies as T

# filter kinds
(X_CELL, X_CELLFROM, X_UNIV, X_MAT, X_MATFROM, X_SURF, X_ENERGY, X_FIXED,
 X_INST, X_BORN, X_MESH, X_MESHMAT, X_POLAR, X_AZI, X_MU, X_MUSURF,
 X_COLL, X_TIME, X_SPH, X_SLEG, X_ZERN, X_ZERNR, X_LEG, X_MESHSURF,
 X_WEIGHT) = range(1, 26)
EV_TRACK, EV_CROSS, EV_FACES = 0, 1, 2
MF = 8                     # most filters on one tally
MAXB = 128                 # most (bin, weight) pairs one filter gives
MAXC = 4096                # most combined bins one event scores
K_EXT = 8                  # points per ball when a position is needed
INF = 1.0e300
# event-state slots (ef: floats, ei: ints), per thread
F_X, F_U, F_T, F_MU, F_MUS, F_P, F_W = 0, 3, 6, 7, 8, 9, 12
NEF, NEI = 13, 6           # F_P: a drawn point; F_W: the analog weight
I_MAT, I_MATFROM, I_SURF, I_COLL, I_LEAF, I_PART = range(6)

GRAZE_CUT, GRAZE_RATIO = 0.001, 0.5    # OpenMC 0.16's surface_grazing_*
MASS_NEUTRON_EV = 939565420.52
C_LIGHT = 2.99792458e10    # cm/s

Ext = namedtuple("Ext", [
    "n",                                   # ext tallies
    "t_off", "t_event", "t_nf", "t_piece", "t_ball_pts", "t_inst",
    "t_faces", "t_nb", "t_iv",             # per tally (t_iv: also an
                                           # inverse-velocity copy, at
                                           # bins t_nb ..)
    "t_cur", "t_sflux",                    # a crossing tally scores
                                           # current (column 0), surface
                                           # flux (column 1)
    "f_type", "f_stride", "f_par", "f_nb",  # per tally x filter
    "lut_start", "lut_len", "lut",         # integer look-ups
    "rp_start", "rp",                      # real parameters (edges, ...)
    "occ_ptr", "occ_len", "occ_start",     # cell instances
    "iv",                                  # inverse velocity per material,
                                           # the last entry for void
    "has_time", "has_state",               # keep time; keep other state
    "any_faces",
])


def default_inverse_velocity(e_lo, e_hi):
    """OpenMC 0.16's inverse velocity (s/cm) for a group without data
    (mgxs_interface.cpp): the average of 1/v over the group in lethargy,
    relativistic, the lower edge floored at 1e-5 eV."""
    e_min, e_max = max(e_lo, 1e-5), e_hi
    alpha = 1.0 / (C_LIGHT * np.log(e_max / e_min))
    k_max = np.sqrt(1 + 2.0 * MASS_NEUTRON_EV / e_max)
    k_min = np.sqrt(1 + 2.0 * MASS_NEUTRON_EV / e_min)
    return alpha * (2.0 * (np.arctanh(1.0 / k_max) - np.arctanh(1.0 / k_min))
                    - (k_max - k_min))


# ---------------------------------------------------------- polynomials
@njit(cache=True)
def _legendre(n, x, out, at):
    out[at] = 1.0
    if n >= 1:
        out[at + 1] = x
    for l in range(1, n):
        out[at + l + 1] = ((2 * l + 1) * x * out[at + l] -
                           l * out[at + l - 1]) / (l + 1)


@njit(cache=True)
def _assoc_legendre(l, m, x):
    """P_l^m(x) with the Condon-Shortley phase, m >= 0."""
    pmm = 1.0
    if m > 0:
        s = np.sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
        f = 1.0
        for _ in range(m):
            pmm *= -f * s
            f += 2.0
    if l == m:
        return pmm
    pmm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmm1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmm1 = pmm1, pll
    return pll


@njit(cache=True)
def _sph(L, u, v, w, out, at):
    """OpenMC's real spherical harmonics R_lm (math_functions.cpp
    calc_rn): sqrt(2 (l-|m|)! / (l+|m|)!) P_l^|m|(w) x sin(|m| phi) for
    m < 0, cos(m phi) for m > 0, P_l(w) for m = 0; phi = 0 when u = 0."""
    phi = 0.0 if u == 0.0 else np.arctan2(v, u)
    k = at
    for l in range(L + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            p = _assoc_legendre(l, am, w)
            if m == 0:
                out[k] = p
            else:
                f = 1.0
                for q in range(l - am + 1, l + am + 1):
                    f *= q
                c = np.sqrt(2.0 / f) * p
                out[k] = c * (np.sin(am * phi) if m < 0 else
                              np.cos(m * phi))
            k += 1


@njit(cache=True)
def _zernike(n, rho, phi, out, at, radial):
    """OpenMC's Zernike polynomials (calc_zn / calc_zn_rad), unnormalised,
    ordered (0,0), (1,-1), (1,1), (2,-2), (2,0), (2,2), ...; radial: only
    the m = 0 ones, R_00, R_20, R_40, ..."""
    zm = np.zeros((n + 1, n + 1))
    for p in range(n + 1):
        zm[p, p] = rho ** p
    for q in range(n - 1):
        zm[q, q + 2] = (q + 2) * zm[q + 2, q + 2] - (q + 1) * zm[q, q]
    for p in range(4, n + 1):
        k2 = 2.0 * p * (p - 1) * (p - 2)
        for q in range(p - 4, -1, -2):
            k1 = ((p + q) * (p - q) * (p - 2)) / 2.0
            k3 = -q * q * (p - 1) - p * (p - 1) * (p - 2)
            k4 = (-p * (p + q - 2) * (p - q - 2)) / 2.0
            zm[q, p] = ((k2 * rho * rho + k3) * zm[q, p - 2] +
                        k4 * zm[q, p - 4]) / k1
    k = at
    if radial:
        for p in range(0, n + 1, 2):
            out[k] = zm[0, p]
            k += 1
        return
    for p in range(n + 1):
        for q in range(-p, p + 1, 2):
            if q < 0:
                out[k] = zm[-q, p] * np.sin(-q * phi)
            elif q == 0:
                out[k] = zm[0, p]
            else:
                out[k] = zm[q, p] * np.cos(q * phi)
            k += 1


@njit(cache=True, inline="always")
def _edge_bin(E, j, x):
    """OpenMC's lower_bound_index on a filter's edges: the bin holding x
    (the last edge included), or -1."""
    a = E.rp_start[j]
    n = int(E.rp[a])                     # number of edges
    lo, hi = E.rp[a + 1], E.rp[a + n]
    if not (x >= lo and x <= hi):
        return -1
    k = 0
    for q in range(1, n):
        if E.rp[a + 1 + q] <= x and q < n - 1:
            k = q
        elif E.rp[a + 1 + q] > x:
            break
    return k


# ------------------------------------------------------------ the filters
@njit(cache=True)
def _filter(E, ML, MX, e, j, ef, ei, path, prev, born, outb, outc):
    """The (bin, weight) pairs filter j of ext tally e gives an event:
    written to outb / outc, their number returned (0: no match)."""
    k = E.f_type[e, j]
    par = E.f_par[e, j]
    n = 0
    if k == X_CELL or k == X_CELLFROM or k == X_UNIV:
        p = prev if k == X_CELLFROM else path
        a = E.lut_start[par]
        for lv in range(p.size):
            c = p[lv]
            if c < 0:
                break
            b = E.lut[a + c]
            if b >= 0:
                outb[n], outc[n] = b, 1.0
                n += 1
        return n
    if k == X_MAT or k == X_MATFROM:
        m = ei[I_MAT] if k == X_MAT else ei[I_MATFROM]
        if m < 0:
            return 0
        b = E.lut[E.lut_start[par] + m]
        if b >= 0:
            outb[0], outc[0] = b, 1.0
            return 1
        return 0
    if k == X_SURF:
        s = ei[I_SURF]
        if s < 0:
            return 0
        b = E.lut[E.lut_start[par] + s]
        if b >= 0:
            outb[0], outc[0] = b, 1.0
            return 1
        return 0
    if k == X_ENERGY or k == X_FIXED:
        b = par                           # the one bin (or -1)
        if b >= 0:
            outb[0], outc[0] = b, 1.0
            return 1
        return 0
    if k == X_INST:
        # a cell at any level, with its instance: the rank of the event's
        # leaf number among the first leaf numbers of the cell's instances
        a = E.lut_start[par]
        leaf = ei[I_LEAF]
        for lv in range(path.size):
            c = path[lv]
            if c < 0:
                break
            q0, nq = E.occ_ptr[c], E.occ_len[c]
            if nq == 0:
                continue
            lo, hi = 0, nq
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if E.occ_start[q0 + mid] <= leaf:
                    lo = mid
                else:
                    hi = mid
            b = E.lut[a + q0 + lo]
            if b >= 0:
                outb[n], outc[n] = b, 1.0
                n += 1
        return n
    if k == X_BORN:
        b = born[par]
        if b >= 0:
            outb[0], outc[0] = b, 1.0
            return 1
        return 0
    if k == X_MESH or k == X_MESHMAT:
        m = par if k == X_MESH else E.lut[E.lut_start[par] - 1]
        b = T._point_bin(ML, MX, m, ef[F_X], ef[F_X + 1], ef[F_X + 2])
        if b < 0:
            return 0
        if k == X_MESHMAT:
            mat = ei[I_MAT]
            if mat < 0:
                return 0
            nm = E.lut_len[par] // max(1, ML.mesh_dim[m, 0] *
                                        ML.mesh_dim[m, 1] *
                                        ML.mesh_dim[m, 2])
            b = E.lut[E.lut_start[par] + b * nm + mat]
            if b < 0:
                return 0
        outb[0], outc[0] = b, 1.0
        return 1
    if k == X_POLAR:
        b = _edge_bin(E, par, np.arccos(min(max(ef[F_U + 2], -1.0), 1.0)))
    elif k == X_AZI:
        b = _edge_bin(E, par, np.arctan2(ef[F_U + 1], ef[F_U]))
    elif k == X_MU:
        b = _edge_bin(E, par, ef[F_MU])
    elif k == X_MUSURF:
        b = _edge_bin(E, par, ef[F_MUS])
    elif k == X_TIME:
        b = _edge_bin(E, par, ef[F_T])
    elif k == X_WEIGHT:
        b = _edge_bin(E, par, ef[F_W])
    elif k == X_COLL:
        c = ei[I_COLL]
        b = E.lut[E.lut_start[par] + c] if c < E.lut_len[par] else -1
    elif k == X_SPH:
        _sph(E.f_nb[e, j], ef[F_U], ef[F_U + 1], ef[F_U + 2], outc, 0)
        L = E.f_nb[e, j]
        for q in range((L + 1) * (L + 1)):
            outb[q] = q
        return (L + 1) * (L + 1)
    elif k == X_SLEG:
        a = E.rp_start[par]
        axis, lo, hi = int(E.rp[a]), E.rp[a + 1], E.rp[a + 2]
        x = ef[F_P + axis]
        if not (x >= lo and x <= hi):
            return 0
        N = E.f_nb[e, j]
        _legendre(N, 2.0 * (x - lo) / (hi - lo) - 1.0, outc, 0)
        for q in range(N + 1):
            outb[q] = q
        return N + 1
    elif k == X_ZERN or k == X_ZERNR:
        a = E.rp_start[par]
        dx, dy = ef[F_P] - E.rp[a], ef[F_P + 1] - E.rp[a + 1]
        rho = np.sqrt(dx * dx + dy * dy) / E.rp[a + 2]
        if rho > 1.0:
            return 0
        N = E.f_nb[e, j]
        _zernike(N, rho, np.arctan2(dy, dx), outc, 0, k == X_ZERNR)
        nb = N // 2 + 1 if k == X_ZERNR else (N + 1) * (N + 2) // 2
        for q in range(nb):
            outb[q] = q
        return nb
    elif k == X_LEG:
        # scatter moments P_n(mu) of an isotropic scatter: P_0 = 1 and
        # every higher moment 0 on average (scored exactly so)
        outb[0], outc[0] = 0, 1.0
        return 1
    else:
        return 0
    if b < 0:
        return 0
    outb[0], outc[0] = b, 1.0
    return 1


@njit(cache=True)
def _combos(E, ML, MX, e, ef, ei, path, prev, born, wb, wc, cb, cc,
            skip):
    """Every combined bin (and its weight) of ext tally e for an event:
    into cb / cc, their number returned.  Filter `skip` (a MeshSurface
    filter, whose bin the caller adds) is left out."""
    cb[0], cc[0] = 0, 1.0
    n = 1
    for j in range(E.t_nf[e]):
        if j == skip:
            continue
        m = _filter(E, ML, MX, e, j, ef, ei, path, prev, born, wb, wc)
        if m == 0:
            return 0
        st = E.f_stride[e, j]
        if n * m > MAXC:
            m = MAXC // n
        for q in range(n - 1, -1, -1):
            b0, c0 = cb[q], cc[q]
            for a in range(m):
                cb[q * m + a] = b0 + st * wb[a]
                cc[q * m + a] = c0 * wc[a]
        n *= m
    return n


@njit(cache=True, inline="always")
def _deposit(row, E, e, n, cb, cc, flux, sa, ss, iv):
    off = E.t_off[e]
    for q in range(n):
        T._add(row, off + T.NS * cb[q], flux * cc[q], sa, ss)
    if E.t_iv[e]:
        off += T.NS * E.t_nb[e]
        for q in range(n):
            row[off + T.NS * cb[q]] += flux * cc[q] * iv


@njit(cache=True)
def _leaf(g, x, y, z):
    """The leaf (material or void cell instance) number of a point, as
    geometry._path counts it; -1 if lost."""
    univ = g.root
    inst = 0
    for level in range(G.MAX_LEVELS):
        cell = -1
        c0 = g.univ_cell_start[univ]
        for k in range(c0, c0 + g.univ_cell_len[univ]):
            if G.in_cell(g, g.univ_cells[k], x, y, z):
                cell = g.univ_cells[k]
                break
        if cell < 0:
            return -1
        inst += g.cell_off[cell]
        ft = g.cell_fill_type[cell]
        if ft == G.FILL_MATERIAL or ft == G.FILL_VOID:
            return inst
        x, y, z = (x - g.cell_trans[cell, 0], y - g.cell_trans[cell, 1],
                   z - g.cell_trans[cell, 2])
        R = g.cell_rot[cell]
        x, y, z = (R[0, 0] * x + R[0, 1] * y + R[0, 2] * z,
                   R[1, 0] * x + R[1, 1] * y + R[1, 2] * z,
                   R[2, 0] * x + R[2, 1] * y + R[2, 2] * z)
        if ft == G.FILL_UNIVERSE:
            univ = g.cell_fill[cell]
            continue
        lat = g.cell_fill[cell]
        valid, k, x, y, z = G._element(g, lat, x, y, z)
        if valid:
            univ = g.lat_univs[g.lat_univ_start[lat] + k]
            inst += g.lat_off[g.lat_univ_start[lat] + k]
        elif g.lat_outer[lat] >= 0:
            univ = g.lat_outer[lat]
        else:
            return -1
    return -1


# ------------------------------------------------------------ mesh cuts
@njit(cache=True)
def _rect_cut(ML, m, x, y, z, u, v, w, lo):
    """Distance (> lo) along the ray to the next edge plane of
    rectangular mesh m."""
    best = INF
    dim = ML.mesh_dim[m]
    for a in range(3):
        q = x if a == 0 else (y if a == 1 else z)
        d = u if a == 0 else (v if a == 1 else w)
        if d == 0.0:
            continue
        e = ML.mesh_edges[m, a]
        n = dim[a]
        qa = q + lo * d
        # first edge strictly beyond qa in direction d
        if d > 0.0:
            k = np.searchsorted(e[:n + 1], qa, side="right")
            if k <= n:
                t = (e[k] - q) / d
                if t > lo and t < best:
                    best = t
        else:
            k = np.searchsorted(e[:n + 1], qa, side="left") - 1
            if k >= 0:
                t = (e[k] - q) / d
                if t > lo and t < best:
                    best = t
    return best


@njit(cache=True)
def _mesh_cut(ML, MX, m, x, y, z, u, v, w, lo):
    if MX.mesh_type[m] == T.MESH_RECT:
        return _rect_cut(ML, m, x, y, z, u, v, w, lo)
    return T._next_cut(ML, MX, m, x - MX.mesh_org[m, 0],
                       y - MX.mesh_org[m, 1], z - MX.mesh_org[m, 2],
                       u, v, w, lo)


@njit(cache=True)
def _next_cut(E, ML, MX, e, x, y, z, u, v, w, t0, iv, lo):
    """Next piece boundary (> lo) for ext tally e's track events: the
    nearest mesh edge of any of its meshes, or time bin edge."""
    best = INF
    eps = 1.0e-12 * (1.0 + lo)
    for j in range(E.t_nf[e]):
        k = E.f_type[e, j]
        if k == X_MESH or k == X_MESHMAT or k == X_MESHSURF:
            m = E.f_par[e, j] if k != X_MESHMAT else \
                E.lut[E.lut_start[E.f_par[e, j]] - 1]
            c = _mesh_cut(ML, MX, m, x, y, z, u, v, w, lo + eps)
            if c < best:
                best = c
        elif k == X_TIME and iv > 0.0:
            a = E.rp_start[E.f_par[e, j]]
            ne = int(E.rp[a])
            for q in range(ne):
                c = (E.rp[a + 1 + q] - t0) / iv
                if c > lo + eps and c < best:
                    best = c
    return best


# ------------------------------------------------------------ the events
@njit(cache=True)
def _set_track_state(E, XS, i, ef, ei, mat, leaf):
    ef[F_MU] = XS.mu_last[i]
    ef[F_W] = XS.w_analog[i]
    ei[I_MAT] = mat
    ei[I_MATFROM] = XS.prev_mat[i]
    ei[I_SURF] = -1
    ei[I_COLL] = XS.ncoll[i]
    ei[I_LEAF] = leaf
    ei[I_PART] = i


@njit(cache=True)
def track(row, E, ML, MX, g, P, XS, i, path, prev, mat, sa, ss, wgt, x, y, z,
          u, v, w, length, reach, iv, ws):
    """A flight: every ext track tally (and faces tally) scores it.  The
    faces of a MeshSurface filter are looked for up to `reach` (transport.py
    says why it can differ from the flight's length by TINY)."""
    ef, ei, wb, wc, cb, cc = ws
    born = XS.born[i]
    t0 = XS.time[i]
    ef[F_U], ef[F_U + 1], ef[F_U + 2] = u, v, w
    for e in range(E.n):
        ev = E.t_event[e]
        if ev == EV_CROSS:
            continue
        leaf = _leaf(g, x + 0.5 * length * u, y + 0.5 * length * v,
                     z + 0.5 * length * w) if E.t_inst[e] else -1
        _set_track_state(E, XS, i, ef, ei, mat, leaf)
        if ev == EV_FACES:
            _faces(row, E, ML, MX, e, P, XS, i, path, prev, born, ef, ei, wb,
                   wc, cb, cc, sa, wgt, x, y, z, u, v, w, reach, t0, iv)
            continue
        if not E.t_piece[e]:
            ef[F_X], ef[F_X + 1], ef[F_X + 2] = x, y, z
            ef[F_T] = t0
            n = _combos(E, ML, MX, e, ef, ei, path, prev, born, wb, wc,
                        cb, cc, -1)
            if n:
                _deposit(row, E, e, n, cb, cc,
                         T._decayed(wgt, sa, 0.0, length), sa, ss, iv)
            continue
        tc = 0.0
        while tc < length:
            te = min(length, _next_cut(E, ML, MX, e, x, y, z, u, v, w, t0,
                                       iv, tc))
            tm = 0.5 * (tc + te)
            ef[F_X], ef[F_X + 1], ef[F_X + 2] = x + tm * u, y + tm * v, \
                z + tm * w
            ef[F_T] = t0 + tm * iv
            # a point drawn in the piece with the flux's density
            xi = rng.rand(P.seed, i)
            if sa * (te - tc) < 1e-9:
                tp = tc + xi * (te - tc)
            else:
                tp = tc - np.log1p(xi * np.expm1(-sa * (te - tc))) / sa
            ef[F_P], ef[F_P + 1], ef[F_P + 2] = x + tp * u, y + tp * v, \
                z + tp * w
            n = _combos(E, ML, MX, e, ef, ei, path, prev, born, wb, wc,
                        cb, cc, -1)
            if n:
                _deposit(row, E, e, n, cb, cc,
                         T._decayed(wgt, sa, tc, te), sa, ss, iv)
            tc = te


@njit(cache=True)
def _elem_axes(ML, MX, m, x, y, z):
    """(element, i0, i1, i2) of mesh m at a point, element -1 outside."""
    q0, q1, q2 = T._coords(MX, m, x, y, z)
    i0 = T._bin(ML, m, 0, q0)
    i1 = T._bin(ML, m, 1, q1)
    i2 = T._bin(ML, m, 2, q2)
    dim = ML.mesh_dim[m]
    if 0 <= i0 < dim[0] and 0 <= i1 < dim[1] and 0 <= i2 < dim[2]:
        return (i2 * dim[1] + i1) * dim[0] + i0, i0, i1, i2
    return -1, i0, i1, i2


@njit(cache=True)
def _phi_up(MX, m, x, y, z, u, v, w, te, eps, a):
    """Whether a flight crossing a phi face at te goes towards larger phi
    (axis a of a curved mesh's coordinates)."""
    p = T._coords(MX, m, x + (te - eps) * u, y + (te - eps) * v,
                  z + (te - eps) * w)
    q = T._coords(MX, m, x + (te + eps) * u, y + (te + eps) * v,
                  z + (te + eps) * w)
    d = q[a] - p[a]
    if d > np.pi:
        d -= 2.0 * np.pi
    elif d < -np.pi:
        d += 2.0 * np.pi
    return d > 0.0


@njit(cache=True)
def _faces(row, E, ML, MX, e, P, XS, i, path, prev, born, ef, ei, wb, wc, cb,
           cc, sa, wgt, x, y, z, u, v, w, length, t0, iv):
    """The mesh faces a flight crosses, for MeshSurface tally e (OpenMC's
    bins: per element, per axis, out through the low face, in through it,
    out through the high face, in through it)."""
    jf = E.t_faces[e]
    m = E.f_par[e, jf]
    nd = E.f_nb[e, jf]
    st = E.f_stride[e, jf]
    nb = 4 * nd
    off = E.t_off[e]
    dim = ML.mesh_dim[m]
    tc = 0.0
    eps = 1.0e-9 * (1.0 + length)
    el, a0, a1, a2 = _elem_axes(ML, MX, m, x + eps * u, y + eps * v,
                                z + eps * w)
    while tc < length:
        te = _mesh_cut(ML, MX, m, x, y, z, u, v, w, tc + 1e-12 * (1 + tc))
        if te >= length:
            break
        q = te + eps
        el2, b0, b1, b2 = _elem_axes(ML, MX, m, x + q * u, y + q * v,
                                     z + q * w)
        if el2 != el or b0 != a0 or b1 != a1 or b2 != a2:
            # which axis moved, and which way.  phi wraps round, so its
            # way is the sign of the change in phi itself across the face
            # (bin numbers cannot tell it with only two phi bins)
            ax, up = 0, True
            if b0 != a0:
                ax, up = 0, b0 > a0
            elif b1 != a1:
                ax, up = 1, b1 > a1
                if MX.mesh_type[m] == T.MESH_CYL:
                    up = _phi_up(MX, m, x, y, z, u, v, w, te, eps, 1)
            else:
                ax, up = 2, b2 > a2
                if MX.mesh_type[m] == T.MESH_SPH:
                    up = _phi_up(MX, m, x, y, z, u, v, w, te, eps, 2)
            ef[F_X], ef[F_X + 1], ef[F_X + 2] = x + te * u, y + te * v, \
                z + te * w
            ef[F_T] = t0 + te * iv
            n = _combos(E, ML, MX, e, ef, ei, path, prev, born, wb, wc,
                        cb, cc, jf)
            wt = wgt * np.exp(-sa * te)
            for k in range(n):
                if el >= 0:
                    row[off + T.NS * (cb[k] + st * (el * nb + 4 * ax +
                                                    (2 if up else 0)))] += \
                        wt * cc[k]
                if el2 >= 0:
                    row[off + T.NS * (cb[k] + st * (el2 * nb + 4 * ax +
                                                    (1 if up else 3)))] += \
                        wt * cc[k]
        el, a0, a1, a2 = el2, b0, b1, b2
        tc = te


@njit(cache=True)
def ball(row, E, ML, MX, g, P, XS, i, path, prev, mat, sa, ss, flux, x, y, z,
         d, s_cm, iv, ph, ws):
    """A ball step of total flux `flux`, centre (x, y, z), radius d (cm),
    walk path s_cm: every ext track tally scores it (at K_EXT profile
    points where it needs a position or a time)."""
    ef, ei, wb, wc, cb, cc = ws
    born = XS.born[i]
    t0 = XS.time[i]
    for e in range(E.n):
        if E.t_event[e] != EV_TRACK:
            continue
        leaf = _leaf(g, x, y, z) if E.t_inst[e] else -1
        _set_track_state(E, XS, i, ef, ei, mat, leaf)
        if not E.t_ball_pts[e]:
            ef[F_X], ef[F_X + 1], ef[F_X + 2] = x, y, z
            ef[F_T] = t0
            n = _combos(E, ML, MX, e, ef, ei, path, prev, born, wb, wc,
                        cb, cc, -1)
            if n:
                _deposit(row, E, e, n, cb, cc, flux, sa, ss, iv)
            continue
        R = P.ball_R[i]
        for _ in range(K_EXT):
            r, Tm = draw_rT(ph.prof_log_r, ph.prof_log1p_ell, ph.prof_f_unsc,
                            ph.prof_cdf, R, sa / ss, P.s[i], P.seed, i)
            r *= d / R
            px, py, pz = rng.isotropic(P.seed, i)
            ef[F_X], ef[F_X + 1], ef[F_X + 2] = x + r * px, y + r * py, \
                z + r * pz
            ef[F_P], ef[F_P + 1], ef[F_P + 2] = ef[F_X], ef[F_X + 1], \
                ef[F_X + 2]
            ef[F_T] = t0 + Tm * d / R * iv
            n = _combos(E, ML, MX, e, ef, ei, path, prev, born, wb, wc,
                        cb, cc, -1)
            if n:
                _deposit(row, E, e, n, cb, cc, flux / K_EXT, sa, ss, iv)


@njit(cache=True)
def cross(row, E, ML, MX, g, P, XS, i, surf, signed_w, frm, to, mat_to,
          mat_from, x, y, z, u, v, w, mus, ws):
    """A surface crossing for every ext surface tally: weight signed_w
    (+ towards the surface's positive side), cells left (frm) and
    entered (to), at (x, y, z) going (u, v, w); mus the MuSurface cosine.
    Cell and material filters see the cell entered, CellFrom and
    MaterialFrom the one left (tallies.py's conventions)."""
    ef, ei, wb, wc, cb, cc = ws
    born = XS.born[i]
    ef[F_X], ef[F_X + 1], ef[F_X + 2] = x, y, z
    ef[F_P], ef[F_P + 1], ef[F_P + 2] = x, y, z
    ef[F_U], ef[F_U + 1], ef[F_U + 2] = u, v, w
    ef[F_T] = XS.time[i]
    ef[F_MU] = XS.mu_last[i]
    ef[F_W] = XS.w_analog[i]
    ef[F_MUS] = mus
    ei[I_MAT] = mat_to
    ei[I_MATFROM] = mat_from
    ei[I_SURF] = surf
    ei[I_COLL] = XS.ncoll[i]
    ei[I_PART] = i
    for e in range(E.n):
        if E.t_event[e] != EV_CROSS:
            continue
        ei[I_LEAF] = _leaf(g, x + 1e-8 * u, y + 1e-8 * v, z + 1e-8 * w) \
            if E.t_inst[e] else -1
        n = _combos(E, ML, MX, e, ef, ei, to, frm, born, wb, wc, cb, cc,
                    -1)
        off = E.t_off[e]
        if E.t_cur[e]:
            for q in range(n):
                row[off + T.NS * cb[q]] += signed_w * cc[q]
        if E.t_sflux[e]:
            # OpenMC's surface-crossing flux estimator, w / |mu|, with its
            # grazing rule (settings.cpp: below |mu| = 0.001 use 0.0005)
            f = abs(signed_w) / (mus if mus >= GRAZE_CUT else
                                 GRAZE_RATIO * GRAZE_CUT)
            for q in range(n):
                row[off + T.NS * cb[q] + 1] += f * cc[q]


@njit(cache=True)
def room(E, ML, MX, x, y, z, cap):
    """Largest ball around the point inside one element of every mesh a
    MeshSurface ext tally uses (and, under 'cap', of every ext mesh)."""
    best = INF
    for e in range(E.n):
        for j in range(E.t_nf[e]):
            k = E.f_type[e, j]
            if not (k == X_MESHSURF or (cap and (k == X_MESH or
                                                 k == X_MESHMAT))):
                continue
            m = E.f_par[e, j] if k != X_MESHMAT else \
                E.lut[E.lut_start[E.f_par[e, j]] - 1]
            if MX.mesh_type[m] != T.MESH_RECT:
                r = T._curved_room(ML, MX, m, x, y, z)
            else:
                dim = ML.mesh_dim[m]
                out2, ins = 0.0, INF
                for a in range(3):
                    q = x if a == 0 else (y if a == 1 else z)
                    ed = ML.mesh_edges[m, a]
                    lo, hi = ed[0], ed[dim[a]]
                    if q < lo:
                        out2 += (lo - q) ** 2
                    elif q > hi:
                        out2 += (q - hi) ** 2
                    else:
                        b = min(max(T._bin(ML, m, a, q), 0), dim[a] - 1)
                        ins = min(ins, q - ed[b], ed[b + 1] - q)
                r = np.sqrt(out2) if out2 > 0.0 else ins
            best = min(best, r)
    return best


@njit(cache=True)
def born_bins(E, ML, MX, xyz, deep, cols_mesh, cols_cell, out):
    """Birth bins of a batch of sites: column c of out is the bin each
    site falls in for birth filter c -- MeshBorn: the element of mesh slot
    cols_mesh[c] holding the site; CellBorn (cols_mesh[c] < 0): its
    deepest cell `deep` through the look-up starting at cols_cell[c]."""
    for i in range(xyz.shape[0]):
        for c in range(out.shape[1]):
            m = cols_mesh[c]
            if m >= 0:
                out[i, c] = T._point_bin(ML, MX, m, xyz[i, 0], xyz[i, 1],
                                         xyz[i, 2])
            else:
                out[i, c] = E.lut[cols_cell[c] + deep[i]] if deep[i] >= 0 \
                    else -1


# ------------------------------------------------------------ the builder
EXT_KINDS = {"cell": X_CELL, "cellfrom": X_CELLFROM, "universe": X_UNIV,
             "material": X_MAT, "materialfrom": X_MATFROM,
             "surface": X_SURF, "energy": X_ENERGY, "energyout": X_ENERGY,
             "particle": X_FIXED, "cellinstance": X_INST,
             "distribcell": X_INST, "cellborn": X_BORN, "meshborn": X_BORN,
             "mesh": X_MESH, "meshmaterial": X_MESHMAT, "polar": X_POLAR,
             "azimuthal": X_AZI, "mu": X_MU, "musurface": X_MUSURF,
             "collision": X_COLL, "time": X_TIME, "sph": X_SPH,
             "sleg": X_SLEG, "zernike": X_ZERN, "zernikeradial": X_ZERNR,
             "legendre": X_LEG, "meshsurface": X_MESHSURF,
             "weight": X_WEIGHT, "zero": X_FIXED}
# filters that need a point or a time inside a flight or a ball
_POINT_KINDS = {X_MESH, X_MESHMAT, X_TIME, X_SLEG, X_ZERN, X_ZERNR}
# filters on what a ball's walk hides: its directions and its collisions
NO_BALL_TRACK = {X_POLAR, X_AZI, X_SPH, X_MU, X_COLL}
NO_BALL_ANY = {X_MU, X_COLL}


def no_balls(ext_tallies):
    """True if some ext tally cannot be scored with ball steps (see the
    docstring: filters on the walk's directions or collision count)."""
    for tl in ext_tallies:
        kinds = {EXT_KINDS[f.kind] for f in tl.filters}
        if kinds & NO_BALL_ANY or (not tl.surface_type and
                                   kinds & NO_BALL_TRACK):
            return True
    return False


@njit(cache=True)
def workspace():
    """Per-thread work space of the kernels here: (ef, ei, wb, wc, cb,
    cc)."""
    return (np.zeros(NEF), np.zeros(NEI, np.int64), np.zeros(MAXB, np.int64),
            np.zeros(MAXB), np.zeros(MAXC, np.int64), np.zeros(MAXC))


def empty():
    """(Ext, birth columns) with no tallies (every call into this file is
    skipped)."""
    return build([], np.zeros(0, np.int64), None, 1, 1, np.ones(1), None)


def build(ext_tallies, offsets, cell_univ, n_cells, n_mats, iv, occ):
    """Ext tallies (openmc_import.Tally objects whose filters carry their
    `slot` in the Layout, for meshes) -> Ext.

    offsets   each tally's first entry in a row (tallies.compile_layouts)
    cell_univ the universe each cell belongs to
    iv        inverse velocity (s/cm) per material, then one for void
    occ       per cell, the first leaf number of each of its instances
              (openmc_import._instances), or None if no tally needs them
    """
    n = len(ext_tallies)
    nt = max(n, 1)
    t_off = np.zeros(nt, np.int64)
    t_event = np.zeros(nt, np.int64)
    t_nf = np.zeros(nt, np.int64)
    t_piece = np.zeros(nt, np.bool_)
    t_ball_pts = np.zeros(nt, np.bool_)
    t_inst = np.zeros(nt, np.bool_)
    t_faces = np.full(nt, -1, np.int64)
    t_nb = np.zeros(nt, np.int64)
    t_iv = np.zeros(nt, np.bool_)
    t_cur = np.zeros(nt, np.bool_)
    t_sflux = np.zeros(nt, np.bool_)
    f_type = np.zeros((nt, MF), np.int64)
    f_stride = np.zeros((nt, MF), np.int64)
    f_par = np.full((nt, MF), -1, np.int64)
    f_nb = np.zeros((nt, MF), np.int64)
    lut_start, lut_len, lut = [], [], []
    rp_start, rp = [], []
    born_cols = []                 # (mesh slot or -1, cell lut start or -1)
    if occ is not None:
        occ_len = np.array([len(o) for o in occ], np.int64)
        occ_ptr = np.concatenate([[0], np.cumsum(occ_len)[:-1]]).astype(
            np.int64)
        occ_start = np.array([q for o in occ for q in o] or [0], np.int64)
    else:
        occ_len = np.zeros(max(n_cells, 1), np.int64)
        occ_ptr = np.zeros(max(n_cells, 1), np.int64)
        occ_start = np.zeros(1, np.int64)

    def new_lut(size, head=None):
        if head is not None:
            lut.append(int(head))
        lut_start.append(len(lut))
        lut_len.append(int(size))
        lut.extend([-1] * int(size))
        return len(lut_start) - 1

    def set_lut(j, idx, bins):
        a = lut_start[j]
        for i, b in zip(idx, bins):
            lut[a + int(i)] = int(b)

    def new_rp(values):
        rp_start.append(len(rp))
        rp.extend(float(v) for v in values)
        return len(rp_start) - 1

    for e, tl in enumerate(ext_tallies):
        fs = tl.filters
        if len(fs) > MF:
            raise ValueError(f"tally {tl.id} has {len(fs)} filters; at most "
                             f"{MF}")
        kinds = [EXT_KINDS[f.kind] for f in fs]
        t_off[e] = offsets[e]
        t_nf[e] = len(fs)
        t_nb[e] = tl.nbins
        t_iv[e] = "inverse-velocity" in tl.scores
        t_event[e] = EV_CROSS if tl.surface_type and X_MESHSURF not in \
            kinds else EV_FACES if X_MESHSURF in kinds else EV_TRACK
        t_cur[e] = "current" in tl.scores
        t_sflux[e] = t_event[e] == EV_CROSS and "flux" in tl.scores
        t_piece[e] = t_event[e] == EV_TRACK and bool(
            set(kinds) & _POINT_KINDS)
        t_ball_pts[e] = t_piece[e]
        t_inst[e] = X_INST in kinds
        stride = 1
        for j in range(len(fs) - 1, -1, -1):           # last fastest
            f, k = fs[j], kinds[j]
            f_type[e, j], f_stride[e, j] = k, stride
            stride *= f.nbins
            x = f.extra or {}
            if k in (X_SPH, X_SLEG, X_ZERN, X_ZERNR) and f.nbins > MAXB:
                raise ValueError(f"tally {tl.id}: an expansion filter of "
                                 f"{f.nbins} terms; at most {MAXB}")
            if k in (X_CELL, X_CELLFROM):
                f_par[e, j] = new_lut(n_cells)
                set_lut(f_par[e, j], f.bins, range(f.nbins))
            elif k == X_UNIV:
                f_par[e, j] = q = new_lut(n_cells)
                where = {int(u): b for b, u in enumerate(f.bins)}
                for c in range(n_cells):
                    if int(cell_univ[c]) in where:
                        lut[lut_start[q] + c] = where[int(cell_univ[c])]
            elif k in (X_MAT, X_MATFROM):
                f_par[e, j] = new_lut(n_mats)
                set_lut(f_par[e, j], f.bins, range(f.nbins))
            elif k == X_SURF:
                f_par[e, j] = new_lut(x["n_surfs"])
                set_lut(f_par[e, j], f.bins, range(f.nbins))
            elif k in (X_ENERGY, X_FIXED):
                f_par[e, j] = int(f.bins[0])
            elif k == X_INST:
                f_par[e, j] = q = new_lut(occ_start.size)
                for b, (c, i) in enumerate(x["pairs"]):
                    if i >= occ_len[c]:
                        raise ValueError(f"tally {tl.id}: cell index {c} "
                                         f"has {occ_len[c]} instances, not "
                                         f"{i + 1}")
                    lut[lut_start[q] + occ_ptr[c] + i] = b
            elif k == X_BORN:
                f_par[e, j] = len(born_cols)
                if f.kind == "meshborn":
                    born_cols.append((f.slot, -1))
                else:
                    q = new_lut(n_cells)
                    set_lut(q, f.bins, range(f.nbins))
                    born_cols.append((-1, lut_start[q]))
            elif k in (X_MESH, X_MESHSURF):
                f_par[e, j] = f.slot
                f_nb[e, j] = f.ndim
                if k == X_MESHSURF:
                    t_faces[e] = j
            elif k == X_MESHMAT:
                nel = x["n_elements"]
                f_par[e, j] = q = new_lut(nel * n_mats, head=f.slot)
                for b, (el, m) in enumerate(x["pairs"]):
                    lut[lut_start[q] + el * n_mats + m] = b
            elif k in (X_POLAR, X_AZI, X_MU, X_MUSURF, X_TIME, X_WEIGHT):
                ed = np.asarray(x["edges"], float)
                f_par[e, j] = new_rp([ed.size] + list(ed))
            elif k == X_COLL:
                cnt = [int(c) for c in x["counts"]]
                f_par[e, j] = new_lut(max(cnt) + 1)
                set_lut(f_par[e, j], cnt, range(len(cnt)))
            elif k == X_SPH:
                f_nb[e, j] = x["order"]
            elif k == X_SLEG:
                f_nb[e, j] = x["order"]
                f_par[e, j] = new_rp([x["axis"], x["lo"], x["hi"]])
            elif k in (X_ZERN, X_ZERNR):
                f_nb[e, j] = x["order"]
                f_par[e, j] = new_rp([x["x"], x["y"], x["r"]])
            elif k == X_LEG:
                f_nb[e, j] = f.nbins
        if stride != tl.nbins:
            raise AssertionError("ext strides do not multiply to the bins")
        if stride > MAXC and t_event[e] != EV_FACES:
            raise ValueError(f"tally {tl.id}: {stride} combined bins; the "
                             f"general scorer handles at most {MAXC}")
    as_i = lambda a: np.asarray(a if len(a) else [0], np.int64)
    E = Ext(n, t_off, t_event, t_nf, t_piece, t_ball_pts, t_inst, t_faces,
            t_nb, t_iv, t_cur, t_sflux, f_type, f_stride, f_par, f_nb, as_i(lut_start),
            as_i(lut_len), as_i(lut), as_i(rp_start),
            np.asarray(rp if rp else [0.0], float), occ_ptr, occ_len,
            occ_start, np.asarray(iv, float),
            bool(any(X_TIME in [EXT_KINDS[f.kind] for f in tl.filters]
                     for tl in ext_tallies)),
            n > 0, bool((t_event[:n] == EV_FACES).any()))
    cols_mesh = np.array([c[0] for c in born_cols] or [0], np.int64)
    cols_cell = np.array([c[1] for c in born_cols] or [0], np.int64)
    return E, (cols_mesh, cols_cell, len(born_cols))


from ball.profile import draw_rT  # noqa: E402  (after the kernels above)
