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
    sources       every OpenMC source except CompiledSource: independent
                  sources with any spatial and angular distribution,
                  FileSource (.h5), MeshSource, and domain constraints
                  (see sources.py)
    tallies       no filter, cell, material or mesh filters (1D, 2D or 3D
                  RegularMesh, RectilinearMesh), scoring flux, absorption,
                  scatter and total with the track-length estimator; or a
                  surface filter on root-universe vacuum or transmission
                  surfaces, scoring current (net weight crossing, +
                  towards the positive side, as OpenMC defines it)
    settings      fixed-source runs.  Settings that change only the noise
                  or the run's length, never the expected answer (survival
                  biasing, weight windows, cutoffs, temperatures, tally
                  triggers, ...), are not used: each one is named in a
                  warning (`IgnoredSetting`, and `Problem.warnings`) so
                  nothing is ignored silently.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import openmc

from . import geometry as G
from . import sources
from . import surfaces as S


class Unsupported(ValueError):
    """The model uses something this stage of gmc3d does not handle."""


class IgnoredSetting(UserWarning):
    """The model sets something gmc3d does not use.  Only things that
    cannot change the expected answer are let through this way."""


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
class Tally:
    """One OpenMC tally, as the solver will fill it.

    filter is 'total', 'cell', 'material', 'mesh' or 'surface'.  For 'mesh',
    mesh holds (lower_left, upper_right, dimension) of the bounding box,
    always 3 long (a 1D or 2D mesh gets one bin of +-tallies.BIG on each
    missing axis), and edges the bin edges on each axis.  bins are
    internal cell, material or surface indices (cell/material/surface
    filters).
    """
    id: int
    name: str
    filter: str
    scores: list
    bins: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    bin_ids: list = field(default_factory=list)       # OpenMC ids of bins
    mesh: tuple = None
    edges: list = None


@dataclass
class Problem:
    model: openmc.Model          # the original, for the OpenMC run
    geom: G.Geometry
    mats: Materials
    source: sources.SourceSampler
    tallies: list
    cell_ids: np.ndarray         # OpenMC id of each internal cell index
    cell_names: list
    warnings: list = field(default_factory=list)   # settings not used


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


# ----------------------------------------------------------------- settings
# Settings elements (as OpenMC writes them to settings.xml) that gmc3d
# reads.  Everything else present in a model is either refused below or
# named in a warning.
_SETTINGS_USED = {"run_mode", "particles", "batches", "source",
                  "energy_mode", "seed",
                  "mesh"}          # meshes a MeshSpatial / MeshSource uses
# Settings that only say where OpenMC writes its own files or how much it
# prints: no effect on any answer and nothing for gmc3d to do, so no
# warning either.
_SETTINGS_QUIET = {"output", "verbosity", "statepoint", "sourcepoint",
                   "track", "max_tracks"}


def _settings(model):
    """Refuse settings that change the problem; return a warning line
    for each setting that only changes noise or run length."""
    s = model.settings
    if s.run_mode not in (None, "fixed source"):
        raise Unsupported(f"settings.run_mode '{s.run_mode}'; stage 1 "
                          f"solves fixed-source problems (criticality is "
                          f"stage 4)")
    if s.photon_transport:
        raise Unsupported("settings.photon_transport; stage 1 transports "
                          "neutrons only")
    tags = sorted({c.tag for c in s.to_xml_element()}
                  - _SETTINGS_USED - _SETTINGS_QUIET)
    return [f"settings: '{tag}' is set but not used by gmc3d (it can "
            f"change only the noise or the run's length, not the answer)"
            for tag in tags]


# ------------------------------------------------------------------ tallies
SCORES = ("flux", "absorption", "scatter", "total")


def _mesh(t, m):
    """An OpenMC mesh -> (bounding box, bin edges on x, y, z)."""
    from .tallies import BIG
    if isinstance(m, openmc.RegularMesh):
        nd = len(m.dimension)
        ll = np.asarray(m.lower_left, float)
        ur = np.asarray(m.upper_right, float)
        edges = [np.linspace(ll[a], ur[a], int(m.dimension[a]) + 1)
                 for a in range(nd)]
    elif isinstance(m, openmc.RectilinearMesh):
        edges = [np.asarray(g, float) for g in
                 (m.x_grid, m.y_grid, m.z_grid)]
        for a, e in enumerate(edges):
            if e.size < 2 or np.any(np.diff(e) <= 0):
                raise Unsupported(f"tally {t.id}: mesh {m.id} "
                                  f"{'xyz'[a]}_grid must rise strictly")
    else:
        raise Unsupported(f"tally {t.id}: mesh {m.id} is a "
                          f"{type(m).__name__}; stage 1 handles "
                          f"RegularMesh (1D, 2D, 3D) and RectilinearMesh")
    edges += [np.array([-BIG, BIG])] * (3 - len(edges))
    box = (np.array([e[0] for e in edges]), np.array([e[-1] for e in edges]),
           np.array([e.size - 1 for e in edges], np.int64))
    return box, edges


def _check_tally(t, warn):
    """Tally settings gmc3d cannot honour: refuse what would change the
    answer, warn about what changes only when the run stops."""
    est = t.estimator
    if est not in (None, "tracklength"):
        raise Unsupported(f"tally {t.id} asks for the '{est}' estimator; "
                          f"gmc3d scores track length only (inside a ball "
                          f"the collisions are not seen)")
    nucs = [str(n) for n in t.nuclides]
    if nucs and nucs != ["total"]:
        raise Unsupported(f"tally {t.id} lists nuclides {nucs}; stage 1 "
                          f"materials are macroscopic (one group)")
    if t.derivative is not None:
        raise Unsupported(f"tally {t.id} has a derivative; gmc3d has no "
                          f"differential tallies")
    if t.triggers:
        warn.append(f"tally {t.id}: triggers are set but not used (gmc3d "
                    f"runs the batches asked for; triggers only stop a "
                    f"run early)")


def _tallies(model, cell_index, mat_index, surf_index, root_surfs, warn):
    out = []
    for t in model.tallies:
        scores = list(t.scores)
        _check_tally(t, warn)
        if len(t.filters) > 1:
            raise Unsupported(f"tally {t.id} has {len(t.filters)} filters; "
                              f"stage 1 handles one")
        f = t.filters[0] if t.filters else None
        if isinstance(f, openmc.SurfaceFilter):
            out.append(_surface_tally(t, f, model, surf_index, root_surfs))
            continue
        bad = set(scores) - set(SCORES)
        if bad:
            raise Unsupported(f"tally {t.id} scores {sorted(bad)}; stage 1 "
                              f"handles {list(SCORES)} (and current "
                              f"with a SurfaceFilter)")
        tl = Tally(t.id, t.name, "total", scores)
        if f is not None:
            if isinstance(f, openmc.CellFilter):
                tl.filter = "cell"
                tl.bin_ids = [int(b) for b in f.bins]
                all_cells = model.geometry.get_all_cells()
                for cid in tl.bin_ids:
                    if all_cells[cid].fill_type not in ("material", "void"):
                        raise Unsupported(
                            f"tally {t.id} filters on cell {cid}, which "
                            f"holds a universe or lattice; stage 1 scores "
                            f"material and void cells only")
                tl.bins = np.array([cell_index[b] for b in tl.bin_ids])
            elif isinstance(f, openmc.MaterialFilter):
                tl.filter = "material"
                tl.bin_ids = [int(b) for b in f.bins]
                tl.bins = np.array([mat_index[b] for b in tl.bin_ids])
            elif type(f) is openmc.MeshFilter:
                tl.filter = "mesh"
                tl.mesh, tl.edges = _mesh(t, f.mesh)
            else:
                raise Unsupported(f"tally {t.id} filter "
                                  f"{type(f).__name__}; stage 1 handles "
                                  f"Cell, Material, Surface and Mesh")
        out.append(tl)
    return out


def _surface_tally(t, f, model, surf_index, root_surfs):
    """A SurfaceFilter tally: net current through each listed surface.

    Only surfaces of the root universe are accepted (inside a translated
    or rotated universe the crossing direction would need the local
    frame), and not reflective ones (a reflection is a crossing that comes
    straight back).  Balls never cross a surface -- a ball lies inside one
    cell -- so every crossing is an ordinary flight and the tally is exact
    whatever the ball backend."""
    if list(t.scores) != ["current"]:
        raise Unsupported(f"tally {t.id} has a SurfaceFilter; stage 1 "
                          f"scores only current there, alone")
    all_surfs = model.geometry.get_all_surfaces()
    tl = Tally(t.id, t.name, "surface", ["current"])
    tl.bin_ids = [int(b) for b in f.bins]
    for sid in tl.bin_ids:
        if sid not in root_surfs:
            raise Unsupported(f"tally {t.id} filters on surface {sid}, "
                              f"which is not a surface of the root "
                              f"universe's cells")
        if all_surfs[sid].boundary_type == "reflective":
            raise Unsupported(f"tally {t.id} filters on surface {sid}, "
                              f"which is reflective")
    tl.bins = np.array([surf_index[b] for b in tl.bin_ids], np.int64)
    return tl


# ------------------------------------------------------------------- load
def load(model):
    """Compile an openmc.Model into a Problem.  Raises Unsupported."""
    warn = _settings(model)                 # refuse early
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
    root_surfs = set()
    for c in geo.root_universe.cells.values():
        _region_tokens(c.region, surf_index, [], root_surfs)
    root_surfs = {surfs[i].id for i in root_surfs}
    tallies = _tallies(model, cell_index, mat_index, surf_index, root_surfs,
                       warn)
    src = sources.make(model, geom, cell_index, mat_index, univ_index,
                       Unsupported)
    warn += src.warnings
    for line in warn:
        warnings.warn(line, IgnoredSetting, stacklevel=2)
    return Problem(model, geom, mats, src, tallies,
                   np.array([c.id for c in cells]), [c.name for c in cells],
                   warn)
