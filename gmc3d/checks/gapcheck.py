"""The v1.6.1 features, each checked against OpenMC (`python run.py gapcheck`).

Used by: run.py.  Writes gapcheck.json.

v1.6.1 closed the gaps of GAPS.md that needed no change to the method:
boundaries with an albedo, horn and spindle tori, periodic pairs of
general planes, rotational periodic walls on either side, MCPL source
files, MeshSource on a 1D or 2D mesh, a biased isotropic angle, every
OpenMC tally filter and score that has a meaning in a one-group fixed
source problem (core/ext.py), tally triggers, and point detectors behind
walls or with a beam source.  Each case here is one model, run by OpenMC
and by gmc3d (plain Monte Carlo, and exact balls where the model allows
them), compared with checks/validate.py's rule.

Two cases cannot use OpenMC as the reference and are checked otherwise:

    rotation_minus  OpenMC 0.16 turns a particle the wrong way at a
                    rotational periodic wall whose inside is the - side
                    of its planes (runs/2026-10-30/v1.5/openmc_bugs.log),
                    so the reference is OpenMC on the same wedge written
                    with both planes' normals pointing into it (the +
                    side, where OpenMC is right)
    detectors       OpenMC has no point detector: the reference is the
                    track-length flux in a small void sphere around the
                    point (as checks/detcheck.py), and plain Monte Carlo
                    against exact balls
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import openmc

from ball.oracle import Oracle
from core import openmc_import, transport
from problems import common
from . import compare, openmc_ref, validate

PARTICLES, BATCHES = 20_000, 20      # ours; OpenMC: REF_SCALE x as many
REF_SCALE = 4


def _cells_model(name, xs=None):
    """A 2 x 2 lattice of pin universes (a sphere of 'a' in 'b') in a
    vacuum box, a box source: the model most filter cases share."""
    mats, m = common.one_group_materials(
        name, xs or {"a": (0.6, 0.15), "b": (0.4, 0.05)})
    pin_s = openmc.Sphere(r=2.5)
    pin = openmc.Cell(name="pin", fill=m["a"], region=-pin_s)
    mod = openmc.Cell(name="mod", fill=m["b"], region=+pin_s)
    u = openmc.Universe(cells=[pin, mod])
    lat = openmc.RectLattice()
    lat.lower_left, lat.pitch = (-10.0, -10.0), (10.0, 10.0)
    lat.universes = [[u, u], [u, u]]
    box = openmc.model.RectangularParallelepiped(-10, 10, -10, 10, -10, 10,
                                                 boundary_type="vacuum")
    core = openmc.Cell(name="core", fill=lat, region=-box)
    geo = openmc.Geometry([core])
    st = common.settings(common.box_source((-9, -9, -9), (9, 9, 9)))
    return geo, mats, m, st, dict(pin=pin, mod=mod, u=u, core=core,
                                  pin_s=pin_s, box=box)


def _tal(name, filters, scores=("flux",)):
    t = openmc.Tally(name=name)
    t.filters, t.scores = list(filters), list(scores)
    return t


def _mesh(ll, ur, dim):
    me = openmc.RegularMesh()
    me.lower_left, me.upper_right, me.dimension = ll, ur, dim
    return me


def filters_model():
    """Every new filter that balls can score (positions, times, cells,
    births, crossings)."""
    geo, mats, m, st, c = _cells_model("gap_filters")
    m22 = _mesh((-10, -10, -10), (10, 10, 10), (2, 2, 1))
    m14 = _mesh((-10, -10, -10), (10, 10, 10), (1, 4, 1))
    T = [
        _tal("universe", [openmc.UniverseFilter([c["u"], geo.root_universe])]),
        _tal("cellinstance", [openmc.CellInstanceFilter(
            [(c["pin"], 0), (c["pin"], 3), (c["mod"], 1)])]),
        _tal("distribcell", [openmc.DistribcellFilter(c["pin"])],
             ("flux", "absorption")),
        _tal("cellborn", [openmc.CellBornFilter([c["pin"], c["mod"]])]),
        _tal("meshborn", [openmc.MeshBornFilter(m22)]),
        _tal("meshmaterial", [openmc.MeshMaterialFilter(
            m22, [(0, m["a"].id), (0, m["b"].id), (3, m["a"].id)])]),
        _tal("particle", [openmc.ParticleFilter(["neutron", "photon"])]),
        _tal("time", [openmc.TimeFilter([0, 1e-5, 5e-5, 1e-4, 3e-4, 1])]),
        _tal("sleg", [openmc.SpatialLegendreFilter(3, "x", -10, 10)]),
        _tal("zernike", [openmc.ZernikeFilter(3, 0, 0, 10)]),
        _tal("zernikeradial", [openmc.ZernikeRadialFilter(4, 0, 0, 10)]),
        _tal("legendre", [openmc.LegendreFilter(2)], ("scatter",)),
        _tal("energyout", [openmc.EnergyoutFilter([0, 1e6, 2e7])],
             ("scatter",)),
        _tal("weight", [openmc.WeightFilter([0, 0.5, 2])]),
        _tal("two_meshes", [openmc.MeshFilter(m22), openmc.MeshFilter(m14)]),
        _tal("two_cells", [openmc.CellFilter([c["core"], c["pin"]]),
                           openmc.CellFilter([c["mod"], c["pin"]])]),
        _tal("cellfrom_track", [openmc.CellFromFilter([c["pin"], c["mod"],
                                                       c["core"]])],
             ("absorption", "scatter")),
        _tal("scores", [openmc.CellFilter([c["pin"], c["mod"]])],
             ("flux", "nu-scatter", "inverse-velocity", "fission",
              "decay-rate")),
        _tal("reaction", [openmc.ReactionFilter(["(n,elastic)"])]),
    ]
    return openmc.Model(geo, mats, st, openmc.Tallies(T))


def surfaces_model():
    """Tallies scored at crossings, on surfaces of the root universe (OpenMC
    0.16 gets currents wrong on surfaces inside lattice elements and moved
    universes, runs/2026-10-30/v1.5/openmc_bugs.log): a pin sphere in a
    shell in a box, a box source."""
    mats, m = common.one_group_materials("gap_surf", {"a": (0.6, 0.15),
                                                      "b": (0.4, 0.05)})
    s1, s2 = openmc.Sphere(r=3.0), openmc.Sphere(r=6.0)
    box = openmc.model.RectangularParallelepiped(-10, 10, -10, 10, -10, 10,
                                                 boundary_type="vacuum")
    pin = openmc.Cell(name="pin", fill=m["a"], region=-s1)
    shell = openmc.Cell(name="shell", fill=m["b"], region=+s1 & -s2)
    out = openmc.Cell(name="out", fill=m["a"], region=+s2 & -box)
    m22 = _mesh((-10, -10, -10), (10, 10, 10), (2, 2, 1))
    cells = [pin, shell, out]
    T = [
        _tal("surface_mesh", [openmc.SurfaceFilter([s1, s2]),
                              openmc.MeshFilter(m22)], ("current",)),
        _tal("surface_flux", [openmc.SurfaceFilter([s1, s2])],
             ("current", "flux")),
        _tal("matfrom_surface", [openmc.MaterialFromFilter([m["a"], m["b"]]),
                                 openmc.SurfaceFilter([s1, s2])],
             ("current",)),
        _tal("musurface", [openmc.SurfaceFilter([s2]),
                           openmc.MuSurfaceFilter([0, 0.3, 0.7, 1])],
             ("current", "flux")),
        _tal("polar_surface", [openmc.SurfaceFilter([s2]),
                               openmc.PolarFilter([0, 1, 2, np.pi])],
             ("current",)),
        _tal("cellfrom_flux", [openmc.CellFromFilter(cells)],
             ("flux", "current")),
        _tal("matfrom_flux", [openmc.MaterialFromFilter([m["a"], m["b"]])],
             ("flux",)),
        _tal("cellfrom_track", [openmc.CellFromFilter(cells)],
             ("absorption",)),
        _tal("matfrom_track", [openmc.MaterialFromFilter([m["a"], m["b"]])],
             ("absorption", "total")),
    ]
    st = common.settings(common.box_source((-9, -9, -9), (9, 9, 9)))
    return openmc.Model(openmc.Geometry(cells), mats, st,
                        openmc.Tallies(T))


def lattice_crossings_model():
    """CellFrom and MaterialFrom crossing tallies in a lattice: OpenMC
    scores every lattice wall crossed.  A 3 x 3 checkerboard of one-cell
    universes (no surfaces inside the elements, where OpenMC 0.16 gets
    the crossing direction wrong), a vacuum box."""
    mats, m = common.one_group_materials("gap_latx", {"a": (0.6, 0.15),
                                                      "b": (0.4, 0.05)})
    ca = openmc.Cell(name="ca", fill=m["a"])
    cb = openmc.Cell(name="cb", fill=m["b"])
    ua, ub = openmc.Universe(cells=[ca]), openmc.Universe(cells=[cb])
    lat = openmc.RectLattice()
    lat.lower_left, lat.pitch = (-9.0, -9.0), (6.0, 6.0)
    lat.universes = [[ua, ub, ua], [ub, ua, ub], [ua, ub, ua]]
    box = openmc.model.RectangularParallelepiped(-9, 9, -9, 9, -9, 9,
                                                 boundary_type="vacuum")
    core = openmc.Cell(name="core", fill=lat, region=-box)
    T = [_tal("cellfrom_lattice", [openmc.CellFromFilter([ca, cb, core])],
              ("flux", "current")),
         _tal("matfrom_lattice", [openmc.MaterialFromFilter(
             [m["a"], m["b"]])], ("flux", "current")),
         _tal("cell_crossings", [openmc.CellFromFilter([ca, cb]),
                                 openmc.CellFilter([ca, cb])],
              ("current",))]
    st = common.settings(common.box_source((-8, -8, -8), (8, 8, 8)))
    return openmc.Model(openmc.Geometry([core]), mats, st,
                        openmc.Tallies(T))


def meshsurf_model():
    """MeshSurface filters the fast kernels do not take: on a curved mesh,
    and with another filter.  (Their own model: a ball may not cross a
    mesh face, so the faces shrink balls; a strong scatterer keeps some.)"""
    geo, mats, m, st, c = _cells_model("gap_msurf", {"a": (2.0, 0.2),
                                                     "b": (1.5, 0.05)})
    m22 = _mesh((-10, -10, -10), (10, 10, 10), (2, 2, 1))
    cyl = openmc.CylindricalMesh(r_grid=[0, 4, 8], z_grid=[-10, 0, 10],
                                 phi_grid=[0, np.pi, 2 * np.pi])
    sph = openmc.SphericalMesh(r_grid=[0, 3, 7], theta_grid=[0, 1, np.pi],
                               phi_grid=[0, 2, 2 * np.pi])
    T = [_tal("meshsurf_cyl", [openmc.MeshSurfaceFilter(cyl)],
              ("current",)),
         _tal("meshsurf_sph", [openmc.MeshSurfaceFilter(sph)],
              ("current",)),
         _tal("meshsurf_energy", [openmc.MeshSurfaceFilter(m22),
                                  openmc.EnergyFilter([0, 1e6, 2e7])],
              ("current",))]
    return openmc.Model(geo, mats, st, openmc.Tallies(T))


def directions_model():
    """The filters on direction and collision count along tracks: they
    turn balls off (ext.py), so plain Monte Carlo only."""
    geo, mats, m, st, c = _cells_model("gap_dirs")
    T = [
        _tal("polar", [openmc.PolarFilter([0, 0.8, 1.6, 2.4, np.pi])]),
        _tal("azimuthal", [openmc.AzimuthalFilter([-np.pi, -1, 0, 1,
                                                   np.pi])]),
        _tal("sph", [openmc.SphericalHarmonicsFilter(2)]),
        _tal("mu", [openmc.MuFilter([-1, -0.5, 0, 0.5, 1])],
             ("flux", "scatter")),
        _tal("collision", [openmc.CollisionFilter([0, 1, 2, 5])]),
        _tal("polar_cell", [openmc.CellFilter([c["pin"], c["mod"]]),
                            openmc.PolarFilter([0, 1, np.pi])]),
    ]
    return openmc.Model(geo, mats, st, openmc.Tallies(T))


def albedo_model(bc):
    mats, m = common.one_group_materials("gap_alb", {"a": (0.5, 0.1)})
    per = bc == "periodic"
    xl = openmc.XPlane(-5, boundary_type="periodic" if per else "vacuum",
                       albedo=0.5 if per else 1.0)
    xr = openmc.XPlane(5, boundary_type=bc, albedo=0.5)
    yl = openmc.YPlane(-5, boundary_type="vacuum")
    yr = openmc.YPlane(5, boundary_type="vacuum")
    zl = openmc.ZPlane(-5, boundary_type="vacuum")
    zr = openmc.ZPlane(5, boundary_type="vacuum")
    cell = openmc.Cell(fill=m["a"], region=+xl & -xr & +yl & -yr & +zl & -zr)
    me = _mesh((-5, -5, -5), (5, 5, 5), (10, 1, 1))
    T = [_tal("current", [openmc.SurfaceFilter([xr, xl])], ("current",)),
         _tal("flux", [], ("flux", "absorption")),
         _tal("mesh", [openmc.MeshFilter(me)])]
    return openmc.Model(openmc.Geometry([cell]), mats,
                        common.settings(common.point_source((3, 0, 0))),
                        openmc.Tallies(T))


def plane_pair_model():
    """Periodic general planes with no periodic_surface named: paired by
    being the only two."""
    mats, m = common.one_group_materials("gap_pp", {"a": (0.5, 0.1)})
    p1 = openmc.Plane(a=1, b=1, c=0, d=-5, boundary_type="periodic")
    p2 = openmc.Plane(a=1, b=1, c=0, d=5, boundary_type="periodic")
    q1 = openmc.Plane(a=1, b=-1, c=0, d=-5, boundary_type="vacuum")
    q2 = openmc.Plane(a=1, b=-1, c=0, d=5, boundary_type="vacuum")
    zl = openmc.ZPlane(-5, boundary_type="vacuum")
    zr = openmc.ZPlane(5, boundary_type="vacuum")
    cell = openmc.Cell(fill=m["a"], region=+p1 & -p2 & +q1 & -q2 & +zl & -zr)
    me = _mesh((-8, -8, -5), (8, 8, 5), (8, 8, 1))
    T = [_tal("flux", [], ("flux", "absorption")),
         _tal("mesh", [openmc.MeshFilter(me)])]
    return openmc.Model(openmc.Geometry([cell]), mats,
                        common.settings(common.point_source((1, 0, 0))),
                        openmc.Tallies(T))


def horn_model():
    """A horn torus (minor radius > major) and a spindle-like one, filled,
    in a scattering box."""
    mats, m = common.one_group_materials("gap_horn", {"a": (3.0, 0.3),
                                                      "b": (1.0, 0.05)})
    t1 = openmc.ZTorus(x0=-4.0, a=1.0, b=2.0, c=2.0)        # c > a: horn
    t2 = openmc.XTorus(x0=4.0, a=1.5, b=1.0, c=1.5)         # c = a
    box = openmc.model.RectangularParallelepiped(-9, 9, -6, 6, -6, 6,
                                                 boundary_type="vacuum")
    c1 = openmc.Cell(name="horn", fill=m["a"], region=-t1)
    c2 = openmc.Cell(name="touch", fill=m["a"], region=-t2)
    c3 = openmc.Cell(name="rest", fill=m["b"], region=-box & +t1 & +t2)
    me = _mesh((-9, -6, -6), (9, 6, 6), (18, 6, 6))
    T = [_tal("cells", [openmc.CellFilter([c1, c2, c3])],
              ("flux", "absorption")),
         _tal("mesh", [openmc.MeshFilter(me)])]
    return openmc.Model(openmc.Geometry([c1, c2, c3]), mats,
                        common.settings(common.box_source((-8, -5, -5),
                                                          (8, 5, 5))),
                        openmc.Tallies(T))


def rotation_model(plus):
    """A 60-degree wedge with rotational periodic walls (y = 0 and the
    plane at 60 degrees through the z axis), a pin in it.  plus=False:
    the second plane is written with its normal pointing out of the wedge,
    so the wedge is on its - side (OpenMC 0.16 turns particles the wrong
    way there); plus=True: the same wedge written on the + side of both,
    OpenMC's reference for it."""
    mats, m = common.one_group_materials("gap_rot", {"a": (2.0, 0.1),
                                                     "b": (1.0, 0.03)})
    th = np.pi / 3
    p1 = openmc.Plane(a=0, b=1, c=0, d=0, boundary_type="periodic")
    sg = 1.0 if plus else -1.0
    p2 = openmc.Plane(a=sg * np.sin(th), b=-sg * np.cos(th), c=0, d=0,
                      boundary_type="periodic")
    p1.periodic_surface = p2
    cyl = openmc.ZCylinder(r=10.0, boundary_type="vacuum")
    zl = openmc.ZPlane(-5, boundary_type="vacuum")
    zr = openmc.ZPlane(5, boundary_type="vacuum")
    wedge = +p1 & (+p2 if plus else -p2)
    pin_s = openmc.Sphere(x0=6.0, y0=2.0, r=1.5)
    pin = openmc.Cell(name="pin", fill=m["a"], region=-pin_s)
    rest = openmc.Cell(name="rest", fill=m["b"],
                       region=wedge & -cyl & +zl & -zr & +pin_s)
    # (a RegularMesh: OpenMC 0.16's own CylindricalMesh tally here sums to
    # 6% less than its cell tally -- its error, not ours: gmc3d's mesh and
    # cells agree, measured 30 Oct 2026)
    me = _mesh((0, 0, -5), (10, 9, 5), (4, 3, 1))
    T = [_tal("cells", [openmc.CellFilter([pin, rest])],
              ("flux", "absorption")),
         _tal("mesh", [openmc.MeshFilter(me)])]
    src = common.point_source((4.0, 1.0, 0.0))
    return openmc.Model(openmc.Geometry([pin, rest]), mats,
                        common.settings(src), openmc.Tallies(T))


def _box_world(name, src):
    mats, m = common.one_group_materials(name, {"a": (0.3, 0.05)})
    box = openmc.model.RectangularParallelepiped(-6, 6, -6, 6, -6, 6,
                                                 boundary_type="vacuum")
    me = _mesh((-6, -6, -6), (6, 6, 6), (6, 6, 1))
    return openmc.Model(openmc.Geometry([openmc.Cell(fill=m["a"],
                                                     region=-box)]), mats,
                        common.settings(src),
                        openmc.Tallies([_tal("flux", [], ("flux",
                                                          "absorption")),
                                        _tal("mesh",
                                             [openmc.MeshFilter(me)])]))


def mesh_source_model(nd):
    sm = _mesh((-3,) * nd, (3,) * nd, (2,) * nd)
    src = openmc.IndependentSource(space=openmc.stats.MeshSpatial(
        sm, strengths=[1.0, 3.0] if nd == 1 else [1.0, 2.0, 3.0, 4.0]))
    src.energy = openmc.stats.Discrete([1e6], [1.0])
    return _box_world(f"gap_ms{nd}", src)


def biased_iso_model():
    bias = openmc.stats.PolarAzimuthal(
        mu=openmc.stats.Tabular([-1, 1], [0.25, 0.75], "linear-linear"),
        phi=openmc.stats.Uniform(0, 2 * np.pi), reference_uvw=(1.0, 0.0, 0.0),
        reference_vwu=(0.0, 1.0, 0.0))
    src = common.point_source((0, 0, 0))
    src.angle = openmc.stats.Isotropic(bias=bias)
    return _box_world("gap_biso", src)


def mcpl_model(workdir):
    """A FileSource reading an MCPL file that OpenMC itself wrote: the
    particles crossing a sphere in a first run (surf_source_write), so
    positions on the sphere and directions outwards."""
    d = Path(workdir) / "mcpl_make"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "surface_source.mcpl"
    if not path.exists():
        mats, m = common.one_group_materials("gap_mcpl0", {"a": (0.3,
                                                                 0.05)})
        sph = openmc.Sphere(r=3.0)
        box = openmc.model.RectangularParallelepiped(
            -6, 6, -6, 6, -6, 6, boundary_type="vacuum")
        maker = openmc.Model(openmc.Geometry([
            openmc.Cell(fill=m["a"], region=-sph),
            openmc.Cell(fill=m["a"], region=+sph & -box)]), mats,
            common.settings(common.point_source((1, 0.5, 0)), 20_000, 1))
        maker.settings.surf_source_write = {"surface_ids": [sph.id],
                                            "max_particles": 50_000,
                                            "mcpl": True}
        maker.run(cwd=str(d), output=False)
    return _box_world("gap_mcpl", openmc.FileSource(path=str(path)))


def triggers_model():
    geo, mats, m, st, c = _cells_model("gap_trig")
    t = _tal("trig", [openmc.CellFilter([c["pin"], c["mod"]])])
    t.triggers = [openmc.Trigger("rel_err", 0.004)]
    model = openmc.Model(geo, mats, st, openmc.Tallies([t]))
    model.settings.trigger_active = True
    model.settings.trigger_max_batches = 200
    model.settings.trigger_batch_interval = 5
    return model


def detector_model(kind):
    """Point detectors behind reflective / white / periodic walls, or with
    a beam source, and a small void sphere round each for OpenMC."""
    mats, m = common.one_group_materials("gap_det", {"a": (0.4, 0.05)})
    walls = {"reflective": "reflective", "white": "white",
             "periodic": "periodic", "beam": "vacuum"}[kind]
    xl = openmc.XPlane(-6, boundary_type=walls if kind != "beam" else
                       "vacuum")
    xr = openmc.XPlane(6, boundary_type="periodic" if kind == "periodic"
                       else "vacuum")
    if kind == "periodic":
        xl.periodic_surface = xr
    ys = [openmc.YPlane(-6, boundary_type="vacuum"),
          openmc.YPlane(6, boundary_type="vacuum")]
    zs = [openmc.ZPlane(-6, boundary_type="vacuum"),
          openmc.ZPlane(6, boundary_type="vacuum")]
    probe = openmc.Sphere(x0=-4.0, y0=3.0, r=0.6)
    reg = +xl & -xr & +ys[0] & -ys[1] & +zs[0] & -zs[1]
    c_probe = openmc.Cell(name="probe", region=-probe)
    cells = [openmc.Cell(fill=m["a"], region=reg & +probe), c_probe]
    if kind == "beam":
        src = common.point_source((-5.0, 0.0, 0.0), direction=(1, 0.2, 0))
    else:
        src = common.point_source((-3.0, 0.0, 0.0))
    model = openmc.Model(openmc.Geometry(cells), mats,
                         common.settings(src), openmc.Tallies([
                             _tal("probe", [openmc.CellFilter([c_probe])])]))
    common.point_detector(model, "det", (-4.0, 3.0, 0.0))
    return model, 4.0 / 3.0 * np.pi * 0.6 ** 3


# ----------------------------------------------------------------- driver
CASES = {
    "filters": (filters_model, True),
    "surfaces": (surfaces_model, True),
    "lattice_crossings": (lattice_crossings_model, True),
    "meshsurf": (meshsurf_model, True),
    "directions": (directions_model, False),
    "albedo_reflective": (lambda: albedo_model("reflective"), True),
    "albedo_white": (lambda: albedo_model("white"), True),
    "albedo_periodic": (lambda: albedo_model("periodic"), True),
    "plane_pair": (plane_pair_model, True),
    "horn_torus": (horn_model, True),
    "mesh_source_1d": (lambda: mesh_source_model(1), True),
    "mesh_source_2d": (lambda: mesh_source_model(2), True),
    "biased_isotropic": (biased_iso_model, True),
}


# OpenMC 0.16 files the INWARD crossings of a curved mesh's faces by the
# next face of that axis along the ray in the element entered (mesh.cpp,
# raytrace_mesh: tally.surface(ijk, k, !distances[k].max_surface, true)
# after distances[k] is recomputed), not by the face crossed: it reports
# currents in through the r = 0 'face' of the innermost shell, which does
# not exist.  Out-crossings it files right.  So these are compared with
# each element's two inward slots of an axis summed.
FOLD_INWARD = ("meshsurf_cyl", "meshsurf_sph")


def _fold_inward(tallies):
    for n in FOLD_INWARD:
        if n not in tallies:
            continue
        t = tallies[n]
        m, se = t["mean"].copy(), t["se"].copy()
        m[1::4] += m[3::4]
        se[1::4] = np.hypot(se[1::4], se[3::4])
        m[3::4], se[3::4] = 0.0, 0.0
        t["mean"], t["se"] = m, se


def _run_case(name, model, balls, workdir, particles, batches, log,
              ref_model=None):
    pr = openmc_import.load(model)
    ref = openmc_ref.run(ref_model or model, f"{workdir}/gap_{name}",
                         particles * batches * REF_SCALE //
                         validate.REF_BATCHES, validate.REF_BATCHES,
                         seed=11)
    _fold_inward(ref.tallies)
    out = {}
    for label, be in (("mc", None),) + ((("balls", Oracle()),) if balls
                                         else ()):
        r = transport.run(pr, particles, batches, seed=3, backend=be,
                          r_star=1.0)
        _fold_inward(r.tallies)
        cmp = compare.compare(r, ref)
        ok, bad = validate.verdict(cmp)
        if r.counters["lost"]:
            ok, bad = False, bad + [f"{r.counters['lost']} lost"]
        out[label] = {"pass": ok, "bad": bad, "balls": r.counters["balls"],
                      "cmp": cmp}
        log(f"  {name:20s} {label:5s} {'PASS' if ok else 'FAIL'}  "
            f"balls {r.counters['balls']:,}  " + "; ".join(bad))
    return out


def _detectors(kind, workdir, particles, batches, log):
    model, vol = detector_model(kind)
    pr = openmc_import.load(model)
    ref = openmc_ref.run(model, f"{workdir}/gap_det_{kind}",
                         particles * batches * 10 // validate.REF_BATCHES,
                         validate.REF_BATCHES, seed=12)
    e, se = ref.tallies["probe"]["mean"][0, 0] / vol, \
        ref.tallies["probe"]["se"][0, 0] / vol
    out = {"openmc_sphere": (e, se)}
    ok = True
    for label, be in (("mc", None), ("balls", Oracle())):
        r = transport.run(pr, particles, batches, seed=3, backend=be,
                          r_star=1.0)
        v, sv = r.tallies["det"]["mean"][0, 0], r.tallies["det"]["se"][0, 0]
        z = (v - e) / np.hypot(sv, se)
        good = abs(z) <= validate.Z1_MAX
        ok &= good
        out[label] = {"value": v, "se": sv, "z": z, "pass": bool(good)}
        log(f"  det_{kind:15s} {label:5s} {'PASS' if good else 'FAIL'}  "
            f"{v:.5g} +- {sv:.2g} vs OpenMC sphere {e:.5g} +- {se:.2g} "
            f"(z {z:+.2f})")
    out["pass"] = bool(ok)
    return out


def suite(workdir, particles=PARTICLES, batches=BATCHES, names=None,
          log=print):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    rows = {}

    def guard(name, f):
        if names and name not in names:
            return
        t0 = time.perf_counter()
        try:
            rows[name] = f()
        except Exception as e:                    # noqa: BLE001
            rows[name] = {"error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()}
            log(f"  {name:20s} ERROR {type(e).__name__}: {e}")
        rows[name]["seconds"] = time.perf_counter() - t0

    for name, (build, balls) in CASES.items():
        guard(name, lambda: _run_case(name, build(), balls, workdir,
                                      particles, batches, log))
    guard("mcpl", lambda: _run_case("mcpl", mcpl_model(workdir), True,
                                    workdir, particles, batches, log))
    guard("rotation_minus", lambda: _run_case(
        "rotation_minus", rotation_model(False), True, workdir, particles,
        batches, log, ref_model=rotation_model(True)))
    guard("triggers", lambda: _triggers(workdir, log))
    for kind in ("reflective", "white", "periodic", "beam"):
        guard(f"det_{kind}", lambda: _detectors(kind, workdir, particles,
                                                batches, log))
    passed = sum(_passed(r) for r in rows.values())
    log(f"gapcheck: {passed} of {len(rows)} cases pass")
    Path(workdir, "gapcheck.json").write_text(json.dumps(
        rows, indent=1, default=_json))
    return rows


def _triggers(workdir, log):
    model = triggers_model()
    pr = openmc_import.load(model)
    r = transport.run(pr, 5_000, 10, seed=3)
    rel = r.tallies["trig"]["se"][:, 0] / r.tallies["trig"]["mean"][:, 0]
    ok = bool(r.settings["triggers_met"]) and float(rel.max()) <= 0.004 \
        and r.settings["batches"] > 10
    log(f"  triggers             mc    {'PASS' if ok else 'FAIL'}  ran "
        f"{r.settings['batches']} batches (asked 10, max 200), worst "
        f"rel. error {rel.max():.4f} (trigger 0.004)")
    return {"pass": ok, "batches": r.settings["batches"],
            "rel_err": float(rel.max())}


def _passed(r):
    if "error" in r:
        return False
    if "pass" in r:
        return bool(r["pass"])
    return all(v["pass"] for k, v in r.items() if isinstance(v, dict))


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
