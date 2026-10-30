"""Turn an OpenMC model into the arrays the solver runs on.

Used by: run.py and checks/ (every problem enters the solver here).
Builds: geometry.Geometry, plus the materials, source and tallies below.

One Python file describes a problem with OpenMC's own API.  The SAME
`openmc.Model` is handed to OpenMC for the reference run and to `load()` for
ours, so the two codes cannot disagree about what the problem is (SRS
S1-GEO-1).

Anything this stage cannot do is refused here with a message naming the
feature (S1-GEO-6), instead of being run wrongly:

    surfaces      planes, spheres, cylinders and cones (on x, y, z or any
                  axis), tori (horn and spindle ones too) and quadrics
    boundaries    transmission, vacuum, reflective, white, periodic
                  (translational, or rotational about z on either side of
                  its planes), with any albedo
    cells         material, void, distributed-material, universe or
                  lattice (rectangular or hexagonal) fills, with
                  translation and rotation
    materials     one energy group, isotropic scattering, no fission,
                  from an OpenMC multigroup library (Macroscopic data)
    sources       every OpenMC source except CompiledSource: independent
                  sources with any spatial, angular (biased too) and time
                  distribution, FileSource (.h5 and MCPL), MeshSource, and
                  domain constraints (see sources.py)
    tallies       every filter and score of OpenMC 0.16 that means
                  something in a one-group, non-fissile, fixed-source
                  problem: the common ones by the fast kernels of
                  tallies.py, the rest by the general scorer (ext.py);
                  fission scores and the filters only fission or other
                  particles reach give 0, as OpenMC does; tally triggers
                  are honoured.  Refused, with the reason: what OpenMC
                  itself refuses in multigroup mode (EnergyFunctionFilter,
                  heating, (n,gamma), flux with an Energyout or DelayedGroup
                  filter), the 'events' score (a property of the transport
                  scheme, not the problem) and derivatives
    detectors     point detectors (gmc3d's own; see detectors.py)
    settings      fixed-source runs.  Settings that change only the noise
                  or the run's length, never the expected answer (survival
                  biasing, weight windows, cutoffs, temperatures, the
                  estimator, ...), are not used: each one is named in a
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
    energy: float = 1.0          # eV: the group's midpoint, the energy
                                 # OpenMC gives a multigroup particle
                                 # (what an EnergyFilter sees)
    inv_vel: np.ndarray = None   # s/cm per material, then void's (time
                                 # and the inverse-velocity score, v1.6.1)


@dataclass
class Filter:
    """One OpenMC filter of a tally, as the solver uses it.

    kind is 'cell', 'cellfrom', 'material', 'mesh', 'surface' or 'energy'.
    bins are internal cell, material or surface indices; for 'energy' it
    holds the one bin every particle falls in (one group: the group's
    midpoint energy, as OpenMC does), or -1 if none.  For 'mesh', mesh
    holds (lower_left, upper_right, dimension) of the bounding box, always
    3 long (a 1D or 2D mesh gets one bin of +-tallies.BIG on each missing
    axis), and edges the bin edges on each axis.  mesh_type is 'rect',
    'cyl' (axes r, phi, z) or 'sph' (r, theta, phi), origin a curved
    mesh's origin; a curved mesh's box is in its own coordinates.
    """
    kind: str
    nbins: int
    bins: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    bin_ids: list = field(default_factory=list)   # OpenMC ids / energies
    mesh: tuple = None
    edges: list = None
    mesh_type: str = "rect"
    origin: np.ndarray = None
    ndim: int = 3                 # a MeshSurfaceFilter's mesh axes
    extra: dict = None            # what a filter of the general scorer
                                  # needs besides its bins (ext.build)
    slot: int = -1                # its mesh's slot (tallies.py sets it)


@dataclass
class Tally:
    """One OpenMC tally, as the solver will fill it.

    filters are its Filters in OpenMC's order; the combined bin of a score
    is OpenMC's too: the LAST filter's bin varies fastest.  A tally with a
    SurfaceFilter scores current at surface crossings; any other scores
    flux-type scores along tracks.

    filter names the filters for reports: 'total' (none), the kind of a
    lone filter ('cell', 'mesh', ...), or the kinds joined by ' x '.  bins,
    bin_ids, mesh and edges repeat a lone filter's (and mesh / edges the
    mesh filter's of a combined tally), as before v1.3.
    """
    id: int
    name: str
    filter: str
    scores: list
    filters: list = field(default_factory=list)
    bins: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    bin_ids: list = field(default_factory=list)       # OpenMC ids of bins
    mesh: tuple = None
    edges: list = None
    mesh_type: str = "rect"
    ext: bool = False            # scored by the general scorer (ext.py)
    surface_type: bool = False   # scored at crossings (current, and
                                 # OpenMC's surface-crossing flux)

    @property
    def nbins(self):
        return int(np.prod([f.nbins for f in self.filters])) \
            if self.filters else 1


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
    detectors: list = field(default_factory=list)  # (name, xyz, r0), v1.6
    det_images: tuple = None     # (points, owner, cut), detectors.py v1.6.1
    ext_info: dict = None        # what ext.build needs (v1.6.1)
    triggers: list = field(default_factory=list)   # tally triggers, v1.6.1
    trigger_settings: dict = None


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
    openmc.Cylinder: (S.CYLINDER, ("x0", "y0", "z0", "r", "dx", "dy", "dz")),
    openmc.Cone: (S.CONE, ("x0", "y0", "z0", "r2", "dx", "dy", "dz")),
    openmc.XTorus: (S.X_TORUS, ("x0", "y0", "z0", "a", "b", "c")),
    openmc.YTorus: (S.Y_TORUS, ("x0", "y0", "z0", "a", "b", "c")),
    openmc.ZTorus: (S.Z_TORUS, ("x0", "y0", "z0", "a", "b", "c")),
    openmc.Quadric: (S.QUADRIC, ("a", "b", "c", "d", "e", "f", "g", "h", "j",
                                 "k")),
}
_BCS = {"transmission": G.BC_TRANSMISSION, "vacuum": G.BC_VACUUM,
        "reflective": G.BC_REFLECTIVE, "white": G.BC_WHITE,
        "periodic": G.BC_PERIODIC}
_PLANES = (openmc.XPlane, openmc.YPlane, openmc.ZPlane, openmc.Plane)


def _surface(surf):
    kind = _SURFACE_KINDS.get(type(surf))
    if kind is None:
        raise Unsupported(f"surface {surf.id} is a {type(surf).__name__}; "
                          f"stage 1 handles {sorted(S.KIND_NAMES.values())}")
    if surf.boundary_type not in _BCS:
        raise Unsupported(f"surface {surf.id} has boundary "
                          f"'{surf.boundary_type}'; stage 1 handles "
                          f"{sorted(_BCS)}")
    if not surf.albedo >= 0.0:
        raise Unsupported(f"surface {surf.id} has albedo {surf.albedo}; "
                          f"an albedo is a weight factor, it must be >= 0")
    code, names = kind
    coef = np.zeros(S.NCOEF)
    coef[:len(names)] = [getattr(surf, n) for n in names]
    if code in (S.CYLINDER, S.CONE):
        d = coef[4:7]
        if not np.linalg.norm(d) > 0.0:
            raise Unsupported(f"surface {surf.id}: axis direction is zero")
        coef[4:7] = d / np.linalg.norm(d)
    if code in (S.X_TORUS, S.Y_TORUS, S.Z_TORUS):
        a, b, c = coef[3:6]
        if not (b > 0.0 and c > 0.0):
            raise Unsupported(f"surface {surf.id}: torus with b = {b}, "
                              f"c = {c}; both must be positive")
    if code == S.QUADRIC:
        A, B, C, D, E, F = coef[:6]
        M = np.array([[A, D / 2, F / 2], [D / 2, B, E / 2],
                      [F / 2, E / 2, C]])
        coef[10] = np.abs(np.linalg.eigvalsh(M)).max()
        if not np.abs(coef[:9]).max() > 0.0:
            raise Unsupported(f"surface {surf.id}: quadric with no x, y or "
                              f"z terms")
    return code, coef, _BCS[surf.boundary_type]


def _plane(surf):
    """(unit normal n, d) with the plane n . p = d."""
    if isinstance(surf, openmc.XPlane):
        n, d = np.array([1.0, 0, 0]), surf.x0
    elif isinstance(surf, openmc.YPlane):
        n, d = np.array([0, 1.0, 0]), surf.y0
    elif isinstance(surf, openmc.ZPlane):
        n, d = np.array([0, 0, 1.0]), surf.z0
    else:
        n, d = np.array([surf.a, surf.b, surf.c], float), surf.d
    k = np.linalg.norm(n)
    return n / k, d / k


def _periodic_shifts(surfs):
    """For each surface: how a particle leaving through it carries on
    (shift with the partner's index in column 3 and the albedo in column
    4, angle).

    Albedo (v1.6.1): the factor a particle's weight is multiplied by when
    a reflective, white or periodic boundary sends it back, the rest
    counting as leakage (OpenMC's rule, measured 30 Oct 2026: with albedo
    0.5 on one reflective face its leakage rises by the weight lost there
    and the current through that face stays 0).  1 everywhere else.

    Periodic partners as OpenMC pairs them: `periodic_surface` if set,
    otherwise the one other periodic plane of the same type (X, Y or Z
    planes; for general `Plane`s, the one other periodic general plane
    without a partner of its own, as OpenMC 0.16 does: measured 30 Oct
    2026).  Parallel partners are translational: shift = the jump between
    them.  Partners at an angle are rotational, as in OpenMC: both planes
    must contain the z axis, and a particle leaving through one is turned
    about z by angle = theta(partner normal) + pi - theta(own normal).
    That takes the problem to lie on the + side of both planes, as OpenMC
    does; `_rotation_sides` corrects it when the problem lies on a - side
    (v1.6.1)."""
    shift = np.zeros((len(surfs), 5))
    shift[:, 3] = -1.0                    # the partner's index, as a float
    shift[:, 4] = [s.albedo if s.boundary_type in ("reflective", "white",
                                                    "periodic") else 1.0
                   for s in surfs]
    angle = np.zeros(len(surfs))
    per = [s for s in surfs if s.boundary_type == "periodic"]
    for s in per:
        if not isinstance(s, _PLANES):
            raise Unsupported(f"surface {s.id} is periodic but a "
                              f"{type(s).__name__}; periodic boundaries "
                              f"must be planes (OpenMC refuses it too)")
    index = {s.id: i for i, s in enumerate(surfs)}
    named = {o.periodic_surface.id for o in per
             if o.periodic_surface is not None}
    pairs = {}
    for s in per:
        partner = s.periodic_surface
        if partner is None:
            if type(s) is openmc.Plane:
                same = [o for o in per if o is not s
                        and type(o) is openmc.Plane
                        and o.periodic_surface in (None, s)
                        and o.id not in named]
            else:
                same = [o for o in per if o is not s and type(o) is type(s)]
            if len(same) != 1:
                raise Unsupported(f"periodic surface {s.id} has "
                                  f"{len(same)} candidate partners: set "
                                  f"periodic_surface")
            partner = same[0]
        if partner.boundary_type != "periodic":
            raise Unsupported(f"surface {s.id}'s periodic partner "
                              f"{partner.id} is not periodic")
        pairs[s.id] = partner.id
        shift[index[s.id], 3] = index.get(partner.id, -1)   # -1: unused
        n1, d1 = _plane(s)
        n2, d2 = _plane(partner)
        if np.linalg.norm(np.cross(n1, n2)) <= 1e-9:
            if np.dot(n1, n2) < 0:
                n2, d2 = -n2, -d2
            shift[index[s.id], :3] = (d2 - d1) * n1
            continue
        for o, n, d in ((s, n1, d1), (partner, n2, d2)):
            if abs(n[2]) > 1e-12 or abs(d) > 1e-10:
                raise Unsupported(
                    f"periodic surfaces {s.id} and {partner.id} are not "
                    f"parallel, so rotational; surface {o.id} does not "
                    f"contain the z axis (OpenMC turns only about z and "
                    f"refuses it too)")
        a = np.arctan2(n2[1], n2[0]) + np.pi - np.arctan2(n1[1], n1[0])
        angle[index[s.id]] = (a + np.pi) % (2 * np.pi) - np.pi
    return shift, angle


def _side(geom, n):
    """Which side of a plane through the z axis (unit normal n) the
    geometry lies on, next to the plane: +1, -1, or 0 if points on both
    sides (or neither) are inside it."""
    t = np.array([-n[1], n[0], 0.0])
    hits = np.zeros(2, np.int64)
    for rho in np.logspace(-3, 4, 29):
        for sg in (1.0, -1.0):
            for z in (0.0, 0.37, -0.37, 3.1, -3.1, 31.0, -31.0):
                p = sg * rho * t + np.array([0.0, 0.0, z])
                eps = 1e-6 * (1.0 + rho)
                for k, side in enumerate((1.0, -1.0)):
                    q = p + side * eps * n
                    if G.locate(geom, q[0], q[1], q[2])[1] != G.LOST:
                        hits[k] += 1
    if hits[0] and not hits[1]:
        return 1
    if hits[1] and not hits[0]:
        return -1
    return 0


def _rotation_sides(geom, surfs):
    """Correct each rotational periodic turn for the side the problem is
    on (v1.6.1).  OpenMC 0.16 assumes the + side of both planes and turns
    the wrong way otherwise (a silent wrong answer there).  The turn that
    brings a particle leaving through plane 1 in through plane 2 maps
    the inward normal m1 = s1 n1 onto -m2 = -s2 n2: angle = theta(s2 n2) +
    pi - theta(s1 n1).  Returns warning lines for the planes where this
    differs from OpenMC."""
    warn = []
    for i, s in enumerate(surfs):
        if geom.surf_angle[i] == 0.0:
            continue
        j = int(geom.surf_shift[i, 3])
        n1, _ = _plane(s)
        n2, _ = _plane(surfs[j])
        s1, s2 = _side(geom, n1), _side(geom, n2)
        if 0 in (s1, s2) or (s1, s2) == (1, 1):
            continue
        m1, m2 = s1 * n1, s2 * n2
        a = np.arctan2(m2[1], m2[0]) + np.pi - np.arctan2(m1[1], m1[0])
        geom.surf_angle[i] = (a + np.pi) % (2 * np.pi) - np.pi
        warn.append(f"rotational periodic surfaces {s.id} and "
                    f"{surfs[j].id}: the problem lies on the - side of at "
                    f"least one; gmc3d turns particles the right way, "
                    f"OpenMC 0.16 assumes the + side and turns them the "
                    f"wrong way, so its answer for this model is wrong")
    return sorted(set(warn))


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
    edges = np.asarray(lib.energy_groups.group_edges, float)
    st, ss, sa, iv = [], [], [], []
    from .ext import default_inverse_velocity
    iv_default = float(default_inverse_velocity(edges[0], edges[-1]))
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
        mult = getattr(xs, "multiplicity_matrix", None)
        if mult is not None and mult[0] is not None and \
                np.any(np.abs(np.asarray(mult[0], float) - 1.0) > 1e-12):
            raise Unsupported(f"{macro} has a scattering multiplicity other "
                              f"than 1 ((n,xn)); stage 1 has none")
        v = getattr(xs, "inverse_velocity", None)
        iv.append(float(np.ravel(v[0])[0]) if v is not None and
                  v[0] is not None else iv_default)
        if abs(tot - absn - sct) > 1e-6 * max(tot, 1e-30):
            raise Unsupported(f"{macro}: total {tot} != absorption "
                              f"{absn} + scattering {sct}")
        st.append(dens * tot)
        ss.append(dens * sct)
        sa.append(dens * absn)
    return Materials([m.id for m in mats], [m.name for m in mats],
                     np.array(st), np.array(ss), np.array(sa),
                     0.5 * (edges[0] + edges[-1]),
                     np.array(iv + [iv_default], float))


# ----------------------------------------------------------------- settings
# Settings elements (as OpenMC writes them to settings.xml) that gmc3d
# reads.  Everything else present in a model is either refused below or
# named in a warning.
_SETTINGS_USED = {"run_mode", "particles", "batches", "source",
                  "energy_mode", "seed", "trigger",
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
def _mesh(t, m, shift=None):
    """An OpenMC mesh -> (bounding box, bin edges on each axis, type,
    origin).  Curved meshes keep OpenMC's axes and origin.  shift: a mesh
    filter's translation (v1.6.1), added to the edges or the origin."""
    box, edges, mtype, origin = _mesh0(t, m)
    if shift is None:
        return box, edges, mtype, origin
    shift = np.asarray(shift, float)
    if mtype != "rect":
        return box, edges, mtype, origin + shift
    edges = [e + shift[a] if e.size > 2 or abs(e[0]) < 1e99 else e
             for a, e in enumerate(edges)]
    box = (np.array([e[0] for e in edges]), np.array([e[-1] for e in edges]),
           box[2])
    return box, edges, mtype, origin


def _mesh0(t, m):
    from .tallies import BIG
    if isinstance(m, (openmc.CylindricalMesh, openmc.SphericalMesh)):
        cyl = isinstance(m, openmc.CylindricalMesh)
        names = ("r", "phi", "z") if cyl else ("r", "theta", "phi")
        edges = [np.asarray(getattr(m, f"{a}_grid"), float) for a in names]
        top = {"r": np.inf, "phi": 2 * np.pi, "z": np.inf, "theta": np.pi}
        for a, e in zip(names, edges):
            if e.size < 2 or np.any(np.diff(e) <= 0):
                raise Unsupported(f"tally {t.id}: mesh {m.id} {a}_grid "
                                  f"must rise strictly")
            if a != "z" and (e[0] < 0.0 or e[-1] > top[a] * (1 + 1e-12)):
                raise Unsupported(f"tally {t.id}: mesh {m.id} {a}_grid "
                                  f"must lie in [0, {top[a]:g}]")
        box = (np.array([e[0] for e in edges]),
               np.array([e[-1] for e in edges]),
               np.array([e.size - 1 for e in edges], np.int64))
        return box, edges, "cyl" if cyl else "sph", \
            np.asarray(m.origin, float)
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
                          f"RegularMesh (1D, 2D, 3D), RectilinearMesh, "
                          f"CylindricalMesh and SphericalMesh")
    edges += [np.array([-BIG, BIG])] * (3 - len(edges))
    box = (np.array([e[0] for e in edges]), np.array([e[-1] for e in edges]),
           np.array([e.size - 1 for e in edges], np.int64))
    return box, edges, "rect", None


def _check_tally(t, warn):
    """Tally settings gmc3d cannot honour: refuse what would change the
    answer, warn about what changes only the noise."""
    est = t.estimator
    if est not in (None, "tracklength"):
        warn.append(f"tally {t.id}: the '{est}' estimator is asked for; "
                    f"gmc3d scores track length (the same expected value, "
                    f"other noise: inside a ball the collisions are not "
                    f"seen)")
    nucs = [str(n) for n in t.nuclides]
    if nucs and nucs != ["total"]:
        raise Unsupported(f"tally {t.id} lists nuclides {nucs}; stage 1 "
                          f"materials are macroscopic (one group)")
    if t.derivative is not None:
        raise Unsupported(f"tally {t.id} has a derivative; gmc3d has no "
                          f"differential tallies (they need the collisions "
                          f"a ball hides)")


# OpenMC filter classes, by exact type (several of OpenMC's filters are
# subclasses of others and mean something else: MeshBornFilter of
# MeshFilter, for instance).  The fast kernels of tallies.py take these:
_FILTER_KIND = {openmc.CellFilter: "cell", openmc.CellFromFilter: "cellfrom",
                openmc.MaterialFilter: "material", openmc.MeshFilter: "mesh",
                openmc.MeshSurfaceFilter: "meshsurface",
                openmc.SurfaceFilter: "surface",
                openmc.EnergyFilter: "energy"}
# ... and the general scorer (ext.py) the rest (v1.6.1):
_EXT_FILTER = {openmc.UniverseFilter: "universe",
               openmc.CellInstanceFilter: "cellinstance",
               openmc.DistribcellFilter: "distribcell",
               openmc.MaterialFromFilter: "materialfrom",
               openmc.CellBornFilter: "cellborn",
               openmc.MeshBornFilter: "meshborn",
               openmc.MeshMaterialFilter: "meshmaterial",
               openmc.ParticleFilter: "particle",
               openmc.PolarFilter: "polar",
               openmc.AzimuthalFilter: "azimuthal",
               openmc.SphericalHarmonicsFilter: "sph",
               openmc.MuFilter: "mu", openmc.MuSurfaceFilter: "musurface",
               openmc.CollisionFilter: "collision",
               openmc.TimeFilter: "time",
               openmc.SpatialLegendreFilter: "sleg",
               openmc.ZernikeFilter: "zernike",
               openmc.ZernikeRadialFilter: "zernikeradial",
               openmc.LegendreFilter: "legendre",
               openmc.EnergyoutFilter: "energyout",
               openmc.WeightFilter: "weight"}
# Filters whose every bin is 0 in a one-group, non-fissile, neutron-only
# problem (measured in OpenMC 0.16, 30 Oct 2026: all give 0): no reaction
# has an MT number in multigroup mode, no nuclide decays, no fission makes
# a particle or a delayed neutron.
_ZERO_FILTER = {openmc.ReactionFilter: "ReactionFilter",
                openmc.ParentNuclideFilter: "ParentNuclideFilter",
                openmc.ParticleProductionFilter: "ParticleProductionFilter",
                openmc.DelayedGroupFilter: "DelayedGroupFilter"}
SCORES = ("flux", "absorption", "scatter", "total")
_ZERO_SCORES = ("fission", "nu-fission", "kappa-fission",
                "prompt-nu-fission", "delayed-nu-fission", "decay-rate")
_TRACK_SCORES = SCORES + ("nu-scatter", "inverse-velocity") + _ZERO_SCORES
# OpenMC 0.16 itself stops on these in multigroup mode (measured 30 Oct)
_MG_REFUSED = {"heating": "heating", "(n,gamma)": "(n,gamma)"}


def _tallies(model, mats, cell_index, mat_index, surf_index, warn, ctx):
    out = []
    for t in model.tallies:
        _check_tally(t, warn)
        out.append(_tally(t, model, mats, cell_index, mat_index,
                          surf_index, ctx, warn))
    return out


def _tally(t, model, mats, cell_index, mat_index, surf_index, ctx, warn):
    scores = list(t.scores)
    kinds, zero = [], False
    for f in t.filters:
        if type(f) in _ZERO_FILTER:
            if isinstance(f, openmc.DelayedGroupFilter) and \
                    set(scores) - {"delayed-nu-fission", "decay-rate"}:
                raise Unsupported(f"tally {t.id}: a DelayedGroupFilter with "
                                  f"scores {scores} (OpenMC refuses it too: "
                                  f"'Cannot tally flux with a delayedgroup "
                                  f"filter')")
            kinds.append("zero")
            zero = True
            continue
        if isinstance(f, openmc.EnergyFunctionFilter):
            raise Unsupported(f"tally {t.id}: EnergyFunctionFilter (OpenMC "
                              f"refuses it too in multigroup mode: 'only "
                              f"supported for continuous-energy')")
        k = _FILTER_KIND.get(type(f)) or _EXT_FILTER.get(type(f))
        if k is None:
            raise Unsupported(f"tally {t.id} filter {type(f).__name__}")
        kinds.append(k)
    for sc in scores:
        if sc in _MG_REFUSED:
            raise Unsupported(f"tally {t.id} scores '{sc}' (OpenMC 0.16 "
                              f"refuses it too in multigroup mode)")
        if sc == "events":
            raise Unsupported(f"tally {t.id} scores 'events': the number of "
                              f"events a code takes depends on how it "
                              f"transports, not on the problem (OpenMC's "
                              f"is 4.4 per particle here; gmc3d's is "
                              f"another number)")
    surface = "surface" in kinds
    meshsurf = "meshsurface" in kinds
    # OpenMC 0.16 (tally.cpp) makes a tally a SURFACE tally, scored at
    # crossings, if it has a Surface filter, or a CellFrom or MaterialFrom
    # filter and scores flux or current; its flux there is the surface-
    # crossing estimator, sum of w / |mu|.
    crossing = surface or (bool({"cellfrom", "materialfrom"} &
                                set(kinds)) and
                           bool({"flux", "current"} & set(scores)))
    if meshsurf:
        if scores != ["current"]:
            raise Unsupported(f"tally {t.id}: a MeshSurfaceFilter tally "
                              f"scores current only (OpenMC refuses other "
                              f"scores too)")
        both = {"surface", "cell", "cellfrom", "material",
                "materialfrom"} & set(kinds)
        if both:
            raise Unsupported(f"tally {t.id}: a MeshSurfaceFilter with "
                              f"{sorted(both)} filters (OpenMC refuses it "
                              f"too: 'Cannot tally mesh surface currents in "
                              f"the same tally as normal surface currents')")
    elif crossing:
        if set(scores) - {"current", "flux"}:
            raise Unsupported(f"tally {t.id} is scored at surface "
                              f"crossings (a Surface, CellFrom or "
                              f"MaterialFrom filter with flux or current), "
                              f"so it may score only current and flux "
                              f"(OpenMC refuses others too)")
    else:
        bad = [sc for sc in scores if sc not in _TRACK_SCORES]
        if bad:
            raise Unsupported(f"tally {t.id} scores {bad}; gmc3d handles "
                              f"{list(_TRACK_SCORES)}, and current with a "
                              f"Surface or MeshSurface filter")
        if "musurface" in kinds:
            raise Unsupported(f"tally {t.id}: a MuSurfaceFilter needs a "
                              f"SurfaceFilter (it bins the cosine at a "
                              f"surface crossing)")
        if "energyout" in kinds and "flux" in scores:
            raise Unsupported(f"tally {t.id}: an EnergyoutFilter with flux "
                              f"(OpenMC refuses it too: 'Cannot tally flux "
                              f"with an outgoing energy filter')")
    filters = []
    for f, k in zip(t.filters, kinds):
        if k == "zero":
            filters.append(Filter("zero", len(f.bins), np.array([-1]),
                                  list(f.bins)))
        else:
            filters.append(_filter(t, f, k, model, mats, cell_index,
                                   mat_index, surf_index, ctx))
    fast = set(kinds) <= set(_FILTER_KIND.values()) and \
        len(set(kinds)) == len(kinds) and \
        not ("cellfrom" in kinds and not surface) and \
        not (surface and "mesh" in kinds) and \
        not (crossing and scores != ["current"]) and \
        not (meshsurf and (kinds != ["meshsurface"] or
                           filters[0].mesh_type != "rect")) and \
        "inverse-velocity" not in scores
    tl = Tally(t.id, t.name, "total" if not kinds else " x ".join(kinds),
               scores, filters, ext=not fast,
               surface_type=crossing or meshsurf)
    if len(filters) == 1:
        tl.bins, tl.bin_ids = filters[0].bins, filters[0].bin_ids
    for f in filters:
        if f.edges is not None and f.kind in ("mesh", "meshsurface"):
            tl.mesh, tl.edges, tl.mesh_type = f.mesh, f.edges, f.mesh_type
    if zero:
        tl.filter += " (0 in a one-group, non-fissile problem)"
    ctx["inst"] |= bool({"cellinstance", "distribcell"} & set(kinds))
    return tl


def _edges(t, f, lo=None, hi=None):
    e = np.asarray(f.values if hasattr(f, "values") else f.bins, float)
    e = np.unique(np.ravel(e)) if e.ndim > 1 else e
    if e.size < 2 or np.any(np.diff(e) <= 0):
        raise Unsupported(f"tally {t.id}: {type(f).__name__} edges must "
                          f"rise strictly")
    return e


def _filter(t, f, kind, model, mats, cell_index, mat_index, surf_index,
            ctx):
    shift = getattr(f, "translation", None)
    if kind in ("cell", "cellfrom", "cellborn"):
        # Cells at ANY level count, filled ones included: a filled cell
        # scores whatever happens inside it (OpenMC's rule; transport.py
        # passes the cell at every level of the particle's path).  A
        # CellBorn filter sees the deepest cell of the birth point.
        ids = [int(b) for b in f.bins]
        return Filter(kind, len(ids), np.array([cell_index[b] for b in ids],
                                               np.int64), ids)
    if kind == "universe":
        ids = [int(b) for b in f.bins]
        return Filter(kind, len(ids), np.array(
            [ctx["univ_index"][b] for b in ids], np.int64), ids)
    if kind in ("material", "materialfrom"):
        ids = [int(b) for b in f.bins]
        return Filter(kind, len(ids), np.array([mat_index[b] for b in ids],
                                               np.int64), ids)
    if kind in ("mesh", "meshborn"):
        box, edges, mtype, origin = _mesh(t, f.mesh, shift)
        return Filter(kind, int(np.prod(box[2])), mesh=box, edges=edges,
                      mesh_type=mtype, origin=origin)
    if kind == "meshmaterial":
        box, edges, mtype, origin = _mesh(t, f.mesh, shift)
        pairs = [(int(e), mat_index[int(m)]) for e, m in np.asarray(f.bins)]
        return Filter(kind, len(pairs), mesh=box, edges=edges,
                      mesh_type=mtype, origin=origin,
                      extra={"pairs": pairs,
                             "n_elements": int(np.prod(box[2]))})
    if kind == "meshsurface":
        # OpenMC's bins: per element (first axis fastest), per axis of the
        # mesh, (out through the low face, in through it, out through the
        # high face, in through it): 4 x ndim bins per element.  A curved
        # mesh goes to ext.py.
        box, edges, mtype, origin = _mesh(t, f.mesh, shift)
        ndim = len(f.mesh.dimension) if isinstance(
            f.mesh, openmc.RegularMesh) else 3
        return Filter(kind, int(np.prod(box[2])) * 4 * ndim, mesh=box,
                      edges=edges, mesh_type=mtype, origin=origin, ndim=ndim)
    if kind in ("energy", "energyout"):
        # one group: every particle has the group's midpoint energy, before
        # and after a scatter
        e = np.asarray(f.values, float)
        hit = int(np.searchsorted(e, mats.energy, side="right")) - 1
        if not 0 <= hit < e.size - 1:
            hit = -1                      # the group lies outside every bin
        return Filter(kind, e.size - 1, np.array([hit], np.int64), list(e))
    if kind == "particle":
        names = [str(p) for p in f.bins]
        hit = [k for k, p in enumerate(names) if p in
               ("neutron", "ParticleType.NEUTRON")]
        return Filter(kind, len(names), np.array([hit[0] if hit else -1],
                                                 np.int64), names)
    if kind in ("polar", "azimuthal", "mu", "musurface", "time", "weight"):
        e = _edges(t, f)
        return Filter(kind, e.size - 1, bin_ids=list(e),
                      extra={"edges": e})
    if kind == "collision":
        cnt = [int(c) for c in f.bins]
        if min(cnt) < 0:
            raise Unsupported(f"tally {t.id}: negative collision counts")
        return Filter(kind, len(cnt), bin_ids=cnt, extra={"counts": cnt})
    if kind == "sph":
        if f.cosine == "scatter":
            # the cosine of the scatter being scored: isotropic, so every
            # term but (0, 0) is 0 on average, and that one is the score
            # (OpenMC's analog estimate has the same mean; measured)
            return Filter("legendre", (f.order + 1) ** 2,
                          bin_ids=list(range((f.order + 1) ** 2)))
        return Filter(kind, (f.order + 1) ** 2, extra={"order": f.order})
    if kind == "legendre":
        # Legendre moments of the scattering cosine (any score: OpenMC's
        # analog estimate at a scatter); isotropic, so P_0 = 1 and every
        # higher moment 0 on average, scored exactly so
        return Filter(kind, f.order + 1, bin_ids=list(range(f.order + 1)))
    if kind == "sleg":
        return Filter(kind, f.order + 1, extra={
            "order": f.order, "axis": "xyz".index(f.axis),
            "lo": float(f.minimum), "hi": float(f.maximum)})
    if kind in ("zernike", "zernikeradial"):
        n = f.order
        nb = n // 2 + 1 if kind == "zernikeradial" else \
            (n + 1) * (n + 2) // 2
        return Filter(kind, nb, extra={"order": n, "x": float(f.x),
                                       "y": float(f.y), "r": float(f.r)})
    if kind == "cellinstance":
        pairs = [(cell_index[int(c)], int(i)) for c, i in np.asarray(f.bins)]
        return Filter(kind, len(pairs), bin_ids=[tuple(map(int, b)) for b
                                                 in np.asarray(f.bins)],
                      extra={"pairs": pairs})
    if kind == "distribcell":
        c = cell_index[int(np.ravel(f.bins)[0])]
        return Filter(kind, -1, np.array([c], np.int64),
                      [int(np.ravel(f.bins)[0])], extra={"cell": c})
    # surface: any surface, at any universe level and with any boundary
    # condition (v1.5; transport.py scores each kind as OpenMC does).
    # Balls never cross a surface -- a ball lies inside one cell -- so
    # every crossing is an ordinary flight and the tally is exact whatever
    # the ball backend.
    ids = [int(b) for b in f.bins]
    for sid in ids:
        if sid not in surf_index:
            raise Unsupported(f"tally {t.id} filters on surface {sid}, "
                              f"which no cell of the geometry uses")
    return Filter(kind, len(ids), np.array([surf_index[b] for b in ids],
                                           np.int64), ids,
                  extra={"n_surfs": len(surf_index)})


def _finish_instances(tallies, occ):
    """Distribcell filters get their bins (every instance of the cell);
    CellInstance pairs are checked against the instances there are."""
    for tl in tallies:
        for f in tl.filters:
            if f.kind == "distribcell":
                c = f.extra["cell"]
                f.nbins = len(occ[c])
                f.extra = {"pairs": [(c, i) for i in range(f.nbins)]}
                if f.nbins == 0:
                    raise Unsupported(f"tally {tl.id}: cell {f.bin_ids[0]} "
                                      f"has no instance gmc3d can number "
                                      f"(inside a lattice's outer "
                                      f"universe)")
            elif f.kind == "cellinstance":
                for (c, i), b in zip(f.extra["pairs"], f.bin_ids):
                    if i >= len(occ[c]):
                        raise Unsupported(f"tally {tl.id}: cell {b[0]} has "
                                          f"{len(occ[c])} instances, not "
                                          f"{i + 1}")


def _hex_lattice(lat, i, univ_index, ll, pitch, shape, ndim, kind, rings):
    """Fill row i of the lattice arrays for a HexLattice (layout in
    geometry.py) and return its universe list.  Each existing element's
    universe comes from OpenMC's own index mapping (get_universe), so the
    ring order and both orientations are OpenMC's."""
    n = lat.num_rings
    nz = lat.num_axial or 1
    nd = 3 if lat.num_axial is not None else 2
    ndim[i], rings[i] = nd, n
    kind[i] = G.LAT_HEX_Y if lat.orientation == "y" else G.LAT_HEX_X
    p = list(lat.pitch)
    pitch[i] = (p[0], p[0], p[1] if nd == 3 else 1.0)
    c = list(lat.center)
    ll[i] = (c[0], c[1], c[2] - 0.5 * nz * p[1] if nd == 3 else 0.0)
    shape[i] = (2 * n - 1, 2 * n - 1, nz)
    out = []
    for iz in range(nz):
        for b in range(-(n - 1), n):
            for a in range(-(n - 1), n):
                idx = (a, b, iz) if nd == 3 else (a, b)
                ok = max(abs(a), abs(b), abs(a + b)) < n
                out.append(univ_index[lat.get_universe(idx).id] if ok
                           else -1)
    return out


def _instances(geo, cells, cell_index, lats, lat_ustart, lat_univs,
               lat_shape, lat_type, lat_rings, univs, univ_index, mat_index,
               force=False):
    """Instance numbering for distributed materials (layout in
    geometry.py): (cell_dist, cell_off, lat_off, leaf_mat, occ).

    Numbers every material and void cell instance in OpenMC's order (as
    Geometry.determine_paths walks the tree), and gives instance j of a
    distributed cell the j-th material of its fill list.  With force (a
    CellInstance or Distribcell tally, v1.6.1) it numbers them even with
    no distributed material, and occ lists, per cell, the first leaf
    number of each of its instances in order (ext.py finds an event's
    instance from its leaf number); otherwise occ is None."""
    nc = len(cells)
    cell_dist = np.array([c.fill_type == "distribmat" for c in cells])
    cell_off = np.zeros(nc, np.int64)
    lat_off = np.zeros(max(len(lat_univs), 1), np.int64)
    if not cell_dist.any() and not force:
        return cell_dist, cell_off, lat_off, np.zeros(1, np.int64), None

    leaves, has_dist = {}, {}                    # per universe index

    def entries(li):                             # valid lattice entries
        k0 = lat_ustart[li]
        n = int(np.prod(lat_shape[li]))
        return [(k, lat_univs[k0 + k]) for k in range(n)
                if lat_univs[k0 + k] >= 0]

    def count(ui):
        if ui in leaves:
            return
        n, d = 0, False
        for cid in univs[ui].cells:
            ci = cell_index[cid]
            c = cells[ci]
            cell_off[ci] = n
            if c.fill_type == "universe":
                fu = univ_index[c.fill.id]
                count(fu)
                n, d = n + leaves[fu], d or has_dist[fu]
            elif c.fill_type == "lattice":
                li = lats.index(c.fill)
                m = 0
                for k, fu in entries(li):
                    count(fu)
                    lat_off[lat_ustart[li] + k] = m
                    m, d = m + leaves[fu], d or has_dist[fu]
                if c.fill.outer is not None:
                    ou = univ_index[c.fill.outer.id]
                    count(ou)
                    if has_dist[ou]:
                        raise Unsupported(
                            f"lattice {c.fill.id}'s outer universe holds a "
                            f"cell with distributed materials; OpenMC does "
                            f"not number instances there")
                n += m
            else:
                n, d = n + 1, d or bool(cell_dist[ci])
        leaves[ui], has_dist[ui] = n, d

    root = univ_index[geo.root_universe.id]
    count(root)
    leaf_mat = np.full(leaves[root], G.VOID, np.int64)
    seen = np.zeros(nc, np.int64)

    def walk(ui, base):
        if not has_dist[ui]:
            return
        for cid in univs[ui].cells:
            ci = cell_index[cid]
            c = cells[ci]
            g = base + cell_off[ci]
            if c.fill_type == "universe":
                walk(univ_index[c.fill.id], g)
            elif c.fill_type == "lattice":
                li = lats.index(c.fill)
                for k, fu in entries(li):
                    walk(fu, g + lat_off[lat_ustart[li] + k])
            elif cell_dist[ci]:
                j = seen[ci]
                if j < len(c.fill):
                    m = c.fill[j]
                    leaf_mat[g] = G.VOID if m is None else mat_index[m.id]
                seen[ci] += 1

    walk(root, 0)
    occ = None
    if force:
        occ = [[] for _ in range(nc)]

        def walk_all(ui, base):
            for cid in univs[ui].cells:
                ci = cell_index[cid]
                c = cells[ci]
                g = base + cell_off[ci]
                occ[ci].append(int(g))
                if c.fill_type == "universe":
                    walk_all(univ_index[c.fill.id], g)
                elif c.fill_type == "lattice":
                    li = lats.index(c.fill)
                    for k, fu in entries(li):
                        walk_all(fu, g + lat_off[lat_ustart[li] + k])

        walk_all(root, 0)
    for ci in np.flatnonzero(cell_dist):
        if seen[ci] != len(cells[ci].fill):
            raise Unsupported(
                f"cell {cells[ci].id} has {len(cells[ci].fill)} materials "
                f"for {seen[ci]} instances")
    return cell_dist, cell_off, lat_off, leaf_mat, occ


# ------------------------------------------------------------------- load
def _detectors(model, geom, src, surfs, mats, warn):
    """gmc3d's point detectors (problems/common.point_detector), checked,
    and their wall images (detectors.py): (detectors, (images, owner of
    each image, cut-off distance)).

    The next-event estimator scores, from every emission, the flux that
    reaches the point uncollided.  Since v1.6.1 any source is accepted
    (its angular density towards the point is used) and any boundary but
    a curved reflective one (lines are followed through planar mirrors
    and periodic walls by image points; white walls re-emit)."""
    from . import detectors as DT
    dets = list(getattr(model, "gmc3d_detectors", []) or [])
    if not dets:
        return [], (np.zeros((0, 3)), np.zeros(0, np.int64), np.inf)
    names = set()
    for name, xyz, r0 in dets:
        if name in names:
            raise Unsupported(f"two point detectors named '{name}'")
        names.add(name)
        if r0 < 0:
            raise Unsupported(f"point detector '{name}': r0 {r0} < 0")
        if G.locate(geom, *xyz)[1] == G.LOST:
            raise Unsupported(f"point detector '{name}' at {xyz} is outside "
                              f"the geometry")
    try:
        maps = DT.wall_maps(model, surfs, geom)
    except ValueError as e:
        raise Unsupported(str(e)) from None
    sig = mats.sig_t[mats.sig_t > 0.0]
    cut = DT.IMAGE_MFP / sig.min() if sig.size else np.inf
    if not maps:
        img = np.array([d[1] for d in dets], float).reshape(-1, 3)
        return dets, (img, np.arange(len(dets)), np.inf)
    if not np.isfinite(cut):
        raise Unsupported("point detectors with reflective or periodic "
                          "walls in a problem with no material: the wall "
                          "images never fade, so their sum has no end")
    lo, hi = model.geometry.bounding_box
    lo = np.where(np.isfinite(lo), lo, -1e4)
    hi = np.where(np.isfinite(hi), hi, 1e4)
    img, own = [], []
    for j, (name, xyz, r0) in enumerate(dets):
        try:
            pts = DT.images(xyz, maps, (lo, hi), cut)
        except ValueError as e:
            raise Unsupported(f"point detector '{name}': {e}") from None
        img.append(pts)
        own += [j] * len(pts)
    if any(c.fill_type == "void" for c in
           model.geometry.get_all_cells().values()) or np.any(
               mats.sig_t == 0.0):
        warn.append(f"point detectors with reflective or periodic walls "
                    f"and a void region: wall images further than "
                    f"{cut:.3g} cm ({DT.IMAGE_MFP:g} mean free paths of "
                    f"the thinnest material) are dropped, which can cut "
                    f"long void streaming paths")
    return dets, (np.concatenate(img), np.asarray(own, np.int64), cut)


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
    if not np.any(mats.sig_a > 0.0) and \
            all(s.boundary_type != "vacuum" for s in surfs):
        raise Unsupported("no material absorbs and no surface is vacuum: "
                          "a history would never end (OpenMC would run "
                          "forever too)")

    univs = sorted(geo.get_all_universes().values(), key=lambda u: u.id)
    univ_index = {u.id: i for i, u in enumerate(univs)}
    lats = sorted(geo.get_all_lattices().values(), key=lambda l: l.id)
    for lat in lats:
        if not isinstance(lat, (openmc.RectLattice, openmc.HexLattice)):
            raise Unsupported(f"lattice {lat.id} is a {type(lat).__name__}; "
                              f"RectLattice and HexLattice are handled")
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
        elif ft == "distribmat":
            fill_type[i] = G.FILL_MATERIAL     # material from leaf_mat
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
    lat_type, lat_rings = np.zeros(nl, np.int64), np.zeros(nl, np.int64)
    lat_ustart, lat_outer, lat_univs = np.zeros(nl, np.int64), \
        np.full(nl, -1), []
    for i, lat in enumerate(lats):
        lat_ustart[i] = len(lat_univs)
        if lat.outer is not None:
            lat_outer[i] = univ_index[lat.outer.id]
        if isinstance(lat, openmc.HexLattice):
            lat_univs += _hex_lattice(lat, i, univ_index, lat_ll, lat_pitch,
                                      lat_shape, lat_ndim, lat_type,
                                      lat_rings)
            continue
        nd = lat.ndim
        lat_ndim[i] = nd
        lat_ll[i, :nd] = lat.lower_left
        lat_pitch[i, :nd] = lat.pitch
        lat_shape[i, :nd] = lat.shape
        nx, ny, nz = lat_shape[i]
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx = (ix, iy, iz) if nd == 3 else (ix, iy)
                    lat_univs.append(univ_index[lat.get_universe(idx).id])

    need_inst = any(isinstance(f, (openmc.CellInstanceFilter,
                                   openmc.DistribcellFilter))
                    for t in model.tallies for f in t.filters)
    cell_dist, cell_off, lat_off, leaf_mat, occ = _instances(
        geo, cells, cell_index, lats, lat_ustart, lat_univs, lat_shape,
        lat_type, lat_rings, univs, univ_index, mat_index, need_inst)
    as_int = lambda a: np.asarray(a, np.int64)
    geom = G.Geometry(
        as_int(kinds), np.asarray(coefs, float).reshape(-1, S.NCOEF),
        as_int(bcs), *_periodic_shifts(surfs),
        tok_start, tok_len, as_int(tokens),
        cs_start, cs_len, as_int(cell_surfs),
        fill_type, fill, trans, rot,
        cell_dist, cell_off, lat_off, leaf_mat,
        uc_start, uc_len, as_int(univ_cells),
        lat_ll, lat_pitch, lat_shape, as_int(lat_ndim),
        lat_type, lat_rings, lat_ustart, as_int(lat_univs), as_int(lat_outer),
        univ_index[geo.root_universe.id])
    warn += _rotation_sides(geom, surfs)
    for s in surfs:
        if s.albedo != 1.0 and s.boundary_type in ("vacuum",
                                                   "transmission"):
            warn.append(f"surface {s.id}: albedo {s.albedo} on a "
                        f"{s.boundary_type} surface is not used (OpenMC "
                        f"applies an albedo only where a boundary sends "
                        f"particles back)")
    ctx = {"univ_index": univ_index, "inst": False}
    tallies = _tallies(model, mats, cell_index, mat_index, surf_index,
                       warn, ctx)
    if occ is not None:
        _finish_instances(tallies, occ)
    cell_univ = np.zeros(max(nc, 1), np.int64)
    for i, u in enumerate(univs):
        for cid in u.cells:
            cell_univ[cell_index[cid]] = i
    ext_t = [tl for tl in tallies if tl.ext]
    from . import ext as EX
    if ext_t and EX.no_balls(ext_t):
        warn.append("tallies " + ", ".join(str(tl.id) for tl in ext_t) +
                    ": a filter on the direction or the collision count "
                    "along tracks (Polar, Azimuthal, SphericalHarmonics, Mu "
                    "or Collision) cannot be scored inside a ball, whose "
                    "walk is not seen: this model runs without balls "
                    "(plain Monte Carlo, exact)")
    triggers, trig_set = _triggers(model, tallies, warn)
    src = sources.make(model, geom, cell_index, mat_index, univ_index,
                       Unsupported)
    warn += src.warnings
    dets, det_images = _detectors(model, geom, src, surfs, mats, warn)
    for line in warn:
        warnings.warn(line, IgnoredSetting, stacklevel=2)
    return Problem(model, geom, mats, src, tallies,
                   np.array([c.id for c in cells]), [c.name for c in cells],
                   warn, dets, det_images,
                   {"cell_univ": cell_univ, "iv": mats.inv_vel, "occ": occ},
                   triggers, trig_set)


def _triggers(model, tallies, warn):
    """Tally triggers (v1.6.1), as OpenMC 0.16 applies them
    (trigger.cpp): after the batches asked for, and then every
    trigger_batch_interval batches, the run stops once every trigger is
    met or at trigger_max_batches.  A trigger with no scores covers all
    of its tally's; a bin that scored nothing fails it unless
    ignore_zeros.  Returns ([(tally index, type, threshold, score
    columns, ignore_zeros)], settings) -- settings None when triggers
    are off."""
    from .tallies import ALL_SCORES
    out = []
    for k, (t, tl) in enumerate(zip(model.tallies, tallies)):
        for tr in t.triggers:
            names = list(tr.scores) or list(t.scores)
            cols = [0 if sc == "current" else ALL_SCORES.index(sc)
                    for sc in names if sc in t.scores]
            out.append((k, tr.trigger_type, float(tr.threshold), cols,
                        bool(getattr(tr, "ignore_zeros", False))))
    s = model.settings
    on = bool(s.trigger_active) and bool(out)
    if out and not s.trigger_active:
        warn.append("tally triggers are set but settings.trigger_active "
                    "is not: they are not used (as in OpenMC)")
    if not on:
        return out, None
    if s.trigger_max_batches is None:
        raise Unsupported("settings.trigger_active needs "
                          "trigger_max_batches (OpenMC requires it too)")
    return out, {"max_batches": int(s.trigger_max_batches),
                 "interval": int(s.trigger_batch_interval or 1)}
