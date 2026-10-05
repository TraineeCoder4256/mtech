"""Turn an OpenMC model into the arrays the solver runs on.

Used by: run.py and checks/ (every problem enters the solver here).
Builds: geometry.Geometry, plus the materials, source and tallies below.

One Python file describes a problem with OpenMC's own API.  The SAME
`openmc.Model` is handed to OpenMC for the reference run and to `load()` for
ours, so the two codes cannot disagree about what the problem is (SRS
S1-GEO-1).

Anything this stage cannot do is refused here with a message naming the
feature (S1-GEO-6), instead of being run wrongly:

    surfaces      planes, spheres, axis-aligned cylinders and cones
    boundaries    transmission, vacuum, reflective
    cells         material, void, universe or rectangular-lattice fills,
                  with translation and rotation
    materials     one energy group, isotropic scattering, no fission,
                  from an OpenMC multigroup library (Macroscopic data)
    sources       point or box, isotropic or one direction
    tallies       no filter, cell, material or regular-mesh filters;
                  scores flux and absorption
"""

from dataclasses import dataclass, field

import numpy as np
import openmc

from . import geometry as G
from . import surfaces as S


class Unsupported(ValueError):
    """The model uses something this stage of gmc3d does not handle."""


# ------------------------------------------------------------- containers
@dataclass
class Materials:
    """One-group macroscopic cross sections, one entry per material."""
    ids: list                    # OpenMC material ids, in index order
    names: list
    sig_t: np.ndarray            # 1/cm
    sig_s: np.ndarray
    sig_a: np.ndarray


@dataclass
class Source:
    """Where particles are born and which way they go (one or more parts,
    each chosen with probability proportional to its strength)."""
    kind: np.ndarray             # 0 point, 1 box
    lo: np.ndarray               # (n, 3) point, or box lower corner
    hi: np.ndarray               # (n, 3) box upper corner (= lo for a point)
    mono: np.ndarray             # 1 if one direction, 0 if isotropic
    direction: np.ndarray        # (n, 3) unit vector when mono
    cdf: np.ndarray              # cumulative strength, last entry 1


@dataclass
class Tally:
    """One OpenMC tally, as the solver will fill it.

    filter is 'total', 'cell', 'material' or 'mesh'.  For 'mesh', mesh holds
    (lower_left, upper_right, dimension).  bins are internal cell or material
    indices (cell/material filters only).
    """
    id: int
    name: str
    filter: str
    scores: list
    bins: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    bin_ids: list = field(default_factory=list)       # OpenMC ids of bins
    mesh: tuple = None


@dataclass
class Problem:
    model: openmc.Model          # the original, for the OpenMC run
    geom: G.Geometry
    mats: Materials
    source: Source
    tallies: list
    cell_ids: np.ndarray         # OpenMC id of each internal cell index
    cell_names: list


# ------------------------------------------------------------------ surfaces
_SURFACE_KINDS = {
    openmc.XPlane: (S.X_PLANE, ("x0",)),
    openmc.YPlane: (S.Y_PLANE, ("y0",)),
    openmc.ZPlane: (S.Z_PLANE, ("z0",)),
    openmc.Plane: (S.PLANE, ("a", "b", "c", "d")),
    openmc.Sphere: (S.SPHERE, ("x0", "y0", "z0", "r")),
    openmc.XCylinder: (S.X_CYLINDER, ("y0", "z0", "r")),
    openmc.YCylinder: (S.Y_CYLINDER, ("x0", "z0", "r")),
    openmc.ZCylinder: (S.Z_CYLINDER, ("x0", "y0", "r")),
    openmc.XCone: (S.X_CONE, ("x0", "y0", "z0", "r2")),
    openmc.YCone: (S.Y_CONE, ("x0", "y0", "z0", "r2")),
    openmc.ZCone: (S.Z_CONE, ("x0", "y0", "z0", "r2")),
}
_BCS = {"transmission": G.BC_TRANSMISSION, "vacuum": G.BC_VACUUM,
        "reflective": G.BC_REFLECTIVE}


def _surface(surf):
    kind = _SURFACE_KINDS.get(type(surf))
    if kind is None:
        raise Unsupported(f"surface {surf.id} is a {type(surf).__name__}; "
                          f"stage 1 handles {sorted(S.KIND_NAMES.values())}")
    if surf.boundary_type not in _BCS:
        raise Unsupported(f"surface {surf.id} has boundary "
                          f"'{surf.boundary_type}'; stage 1 handles "
                          f"{sorted(_BCS)}")
    code, names = kind
    coef = np.zeros(4)
    coef[:len(names)] = [getattr(surf, n) for n in names]
    return code, coef, _BCS[surf.boundary_type]


def _region_tokens(region, surf_index, out, used):
    """Write a region as postfix tokens (children first, operator after)."""
    if region is None:
        return
    if isinstance(region, openmc.Halfspace):
        i = surf_index[region.surface.id]
        used.add(i)
        out.append(i + 1 if region.side == "+" else -(i + 1))
    elif isinstance(region, openmc.Intersection):
        for k, child in enumerate(region):
            _region_tokens(child, surf_index, out, used)
            if k:
                out.append(G.OP_AND)
    elif isinstance(region, openmc.Union):
        for k, child in enumerate(region):
            _region_tokens(child, surf_index, out, used)
            if k:
                out.append(G.OP_OR)
    elif isinstance(region, openmc.Complement):
        _region_tokens(region.node, surf_index, out, used)
        out.append(G.OP_NOT)
    else:
        raise Unsupported(f"region node {type(region).__name__}")


# --------------------------------------------------------------- materials
def _materials(model):
    mats = list(model.materials) or sorted(
        model.geometry.get_all_materials().values(), key=lambda m: m.id)
    if model.settings.energy_mode != "multi-group":
        raise Unsupported("settings.energy_mode must be 'multi-group' "
                          "(stage 1 is one energy group)")
    path = model.materials.cross_sections
    if path is None:
        raise Unsupported("materials.cross_sections must name the "
                          "multigroup library file")
    lib = openmc.MGXSLibrary.from_hdf5(str(path))
    if lib.energy_groups.num_groups != 1:
        raise Unsupported(f"library has {lib.energy_groups.num_groups} "
                          f"groups; stage 1 handles 1 (multigroup is stage 3)")
    st, ss, sa = [], [], []
    for m in mats:
        macro = m._macroscopic          # OpenMC keeps it private
        if macro is None or m.get_nuclides():
            raise Unsupported(f"material {m.id} must hold exactly one "
                              f"Macroscopic and no nuclides")
        xs = lib.get_by_name(macro)
        dens = m.density if m.density is not None else 1.0
        tot = float(np.ravel(xs.total[0])[0])
        absn = float(np.ravel(xs.absorption[0])[0])
        sm = np.asarray(xs.scatter_matrix[0])
        if xs.order and np.any(np.abs(sm[..., 1:]) > 0.0):
            raise Unsupported(f"{macro} scatters anisotropically; "
                              f"that is stage 2")
        sct = float(np.sum(sm[..., 0]) if sm.ndim == 3 else np.sum(sm))
        if xs.fissionable:
            raise Unsupported(f"{macro} is fissile; criticality is "
                              f"stage 4")
        if abs(tot - absn - sct) > 1e-6 * max(tot, 1e-30):
            raise Unsupported(f"{macro}: total {tot} != absorption "
                              f"{absn} + scattering {sct}")
        st.append(dens * tot)
        ss.append(dens * sct)
        sa.append(dens * absn)
    return Materials([m.id for m in mats], [m.name for m in mats],
                     np.array(st), np.array(ss), np.array(sa))


# ------------------------------------------------------------------ source
def _source(model):
    srcs = model.settings.source
    srcs = srcs if isinstance(srcs, (list, tuple)) else [srcs]
    kind, lo, hi, mono, dirn, w = [], [], [], [], [], []
    for s in srcs:
        if not isinstance(s, openmc.IndependentSource):
            raise Unsupported(f"source type {type(s).__name__}")
        sp = s.space
        if isinstance(sp, openmc.stats.Point):
            kind.append(0)
            lo.append(sp.xyz)
            hi.append(sp.xyz)
        elif isinstance(sp, openmc.stats.Box):
            kind.append(1)
            lo.append(sp.lower_left)
            hi.append(sp.upper_right)
        else:
            raise Unsupported(f"source space {type(sp).__name__}; stage 1 "
                              f"handles Point and Box")
        an = s.angle
        if an is None or isinstance(an, openmc.stats.Isotropic):
            mono.append(0)
            dirn.append((1.0, 0.0, 0.0))
        elif isinstance(an, openmc.stats.Monodirectional):
            d = np.asarray(an.reference_uvw, float)
            mono.append(1)
            dirn.append(d / np.linalg.norm(d))
        else:
            raise Unsupported(f"source angle {type(an).__name__}")
        w.append(s.strength)
    cdf = np.cumsum(w) / np.sum(w)
    return Source(np.array(kind), np.array(lo, float), np.array(hi, float),
                  np.array(mono), np.array(dirn, float), cdf)


# ------------------------------------------------------------------ tallies
def _tallies(model, cell_index, mat_index):
    out = []
    for t in model.tallies:
        scores = list(t.scores)
        bad = set(scores) - {"flux", "absorption"}
        if bad:
            raise Unsupported(f"tally {t.id} scores {sorted(bad)}; stage 1 "
                              f"handles flux and absorption")
        if len(t.filters) > 1:
            raise Unsupported(f"tally {t.id} has {len(t.filters)} filters; "
                              f"stage 1 handles one")
        tl = Tally(t.id, t.name, "total", scores)
        if t.filters:
            f = t.filters[0]
            if isinstance(f, openmc.CellFilter):
                tl.filter = "cell"
                tl.bin_ids = [int(b) for b in f.bins]
                tl.bins = np.array([cell_index[b] for b in tl.bin_ids])
            elif isinstance(f, openmc.MaterialFilter):
                tl.filter = "material"
                tl.bin_ids = [int(b) for b in f.bins]
                tl.bins = np.array([mat_index[b] for b in tl.bin_ids])
            elif isinstance(f, openmc.MeshFilter) and \
                    isinstance(f.mesh, openmc.RegularMesh) and \
                    len(f.mesh.dimension) == 3:
                m = f.mesh
                tl.filter = "mesh"
                tl.mesh = (np.array(m.lower_left, float),
                           np.array(m.upper_right, float),
                           np.array(m.dimension, np.int64))
            else:
                raise Unsupported(f"tally {t.id} filter "
                                  f"{type(f).__name__}; stage 1 handles "
                                  f"Cell, Material and 3D RegularMesh")
        out.append(tl)
    return out


# ------------------------------------------------------------------- load
def load(model):
    """Compile an openmc.Model into a Problem.  Raises Unsupported."""
    geo = model.geometry
    surfs = sorted(geo.get_all_surfaces().values(), key=lambda s: s.id)
    surf_index = {s.id: i for i, s in enumerate(surfs)}
    kinds, coefs, bcs = zip(*[_surface(s) for s in surfs]) if surfs else \
        ((), (), ())

    mats = _materials(model)
    mat_index = {mid: i for i, mid in enumerate(mats.ids)}

    univs = sorted(geo.get_all_universes().values(), key=lambda u: u.id)
    univ_index = {u.id: i for i, u in enumerate(univs)}
    lats = sorted(geo.get_all_lattices().values(), key=lambda l: l.id)
    for lat in lats:
        if not isinstance(lat, openmc.RectLattice):
            raise Unsupported(f"lattice {lat.id} is a {type(lat).__name__}; "
                              f"stage 1 handles RectLattice")
    lat_index = {l.id: i for i, l in enumerate(lats)}
    cells = sorted(geo.get_all_cells().values(), key=lambda c: c.id)
    cell_index = {c.id: i for i, c in enumerate(cells)}

    nc = len(cells)
    tok_start, tok_len, tokens = np.zeros(nc, np.int64), \
        np.zeros(nc, np.int64), []
    cs_start, cs_len, cell_surfs = np.zeros(nc, np.int64), \
        np.zeros(nc, np.int64), []
    fill_type, fill = np.zeros(nc, np.int64), np.zeros(nc, np.int64)
    trans = np.zeros((nc, 3))
    rot = np.tile(np.eye(3), (nc, 1, 1))
    for i, c in enumerate(cells):
        toks, used = [], set()
        _region_tokens(c.region, surf_index, toks, used)
        tok_start[i], tok_len[i] = len(tokens), len(toks)
        tokens += toks
        cs_start[i], cs_len[i] = len(cell_surfs), len(used)
        cell_surfs += sorted(used)
        ft = c.fill_type
        if ft == "material":
            fill_type[i], fill[i] = G.FILL_MATERIAL, mat_index[c.fill.id]
        elif ft == "void":
            fill_type[i] = G.FILL_VOID
        elif ft == "universe":
            fill_type[i], fill[i] = G.FILL_UNIVERSE, univ_index[c.fill.id]
        elif ft == "lattice":
            fill_type[i], fill[i] = G.FILL_LATTICE, lat_index[c.fill.id]
        else:
            raise Unsupported(f"cell {c.id} fill type '{ft}'")
        if c.translation is not None:
            trans[i] = c.translation
        if c.rotation is not None:
            rot[i] = c.rotation_matrix

    uc_start, uc_len, univ_cells = np.zeros(len(univs), np.int64), \
        np.zeros(len(univs), np.int64), []
    for i, u in enumerate(univs):
        uc_start[i], uc_len[i] = len(univ_cells), len(u.cells)
        univ_cells += [cell_index[cid] for cid in u.cells]

    nl = max(len(lats), 1)
    lat_ll, lat_pitch = np.zeros((nl, 3)), np.ones((nl, 3))
    lat_shape, lat_ndim = np.ones((nl, 3), np.int64), np.full(nl, 2)
    lat_ustart, lat_outer, lat_univs = np.zeros(nl, np.int64), \
        np.full(nl, -1), []
    for i, lat in enumerate(lats):
        nd = lat.ndim
        lat_ndim[i] = nd
        lat_ll[i, :nd] = lat.lower_left
        lat_pitch[i, :nd] = lat.pitch
        lat_shape[i, :nd] = lat.shape
        lat_ustart[i] = len(lat_univs)
        nx, ny, nz = lat_shape[i]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx = (ix, iy, iz) if nd == 3 else (ix, iy)
                    lat_univs.append(univ_index[lat.get_universe(idx).id])
        if lat.outer is not None:
            lat_outer[i] = univ_index[lat.outer.id]

    as_int = lambda a: np.asarray(a, np.int64)
    geom = G.Geometry(
        as_int(kinds), np.asarray(coefs, float).reshape(-1, 4), as_int(bcs),
        tok_start, tok_len, as_int(tokens),
        cs_start, cs_len, as_int(cell_surfs),
        fill_type, fill, trans, rot,
        uc_start, uc_len, as_int(univ_cells),
        lat_ll, lat_pitch, lat_shape, as_int(lat_ndim),
        lat_ustart, as_int(lat_univs), as_int(lat_outer),
        univ_index[geo.root_universe.id])
    return Problem(model, geom, mats, _source(model),
                   _tallies(model, cell_index, mat_index),
                   np.array([c.id for c in cells]), [c.name for c in cells])
