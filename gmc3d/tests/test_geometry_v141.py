"""v1.4.1: hexagonal lattices, tori, quadrics, rotational periodic walls,
distributed materials.

Run:  python -m pytest -q tests   (from gmc3d/)

Agreement with OpenMC's own transport is checked by `python run.py
validate hexcore sector`; locating random points as OpenMC does is checked
for both new problems in test_geometry.py.
"""
import numpy as np
import openmc
import pytest

from core import geometry as G
from core import openmc_import, transport
from core import surfaces as S
from core.openmc_import import Unsupported
from problems import PROBLEMS, common

RNG = np.random.default_rng(41)


def _surf(surf):
    kind, coef, _ = openmc_import._surface(surf)
    return kind, coef


TORI = [
    openmc.ZTorus(x0=0.5, y0=-0.3, z0=0.2, a=4.0, b=1.5, c=1.0),
    openmc.XTorus(x0=0.0, y0=1.0, z0=0.0, a=3.0, b=1.2, c=1.2),
    openmc.YTorus(x0=1.0, y0=0.0, z0=-1.0, a=5.0, b=0.8, c=2.0),
]
QUADRICS = [
    openmc.Quadric(a=1 / 9, b=1 / 4, c=1 / 16, d=0.05, e=-0.03, f=0.02,
                   g=0.1, k=-1.0),                         # ellipsoid
    openmc.Quadric(a=1.0, b=1.0, c=-0.5, g=0.3, k=-4.0),   # hyperboloid
]
SHAPES = TORI + QUADRICS


@pytest.mark.parametrize("surf", SHAPES)
def test_value_is_openmcs(surf):
    kind, c = _surf(surf)
    for p in RNG.uniform(-8, 8, (500, 3)):
        assert S.value(kind, c, *p) == pytest.approx(surf.evaluate(p),
                                                     rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("surf", SHAPES)
def test_ray_lands_on_the_surface_and_skips_nothing(surf):
    kind, c = _surf(surf)
    hits = 0
    for _ in range(400):
        p = RNG.uniform(-8, 8, 3)
        d = RNG.normal(size=3)
        d /= np.linalg.norm(d)
        t = S.ray(kind, c, *p, *d)
        ts = np.linspace(0, min(t, 40.0), 600)[1:-1]
        f0 = S.value(kind, c, *p)
        fs = np.array([S.value(kind, c, *(p + s * d)) for s in ts])
        assert np.all(np.sign(fs) == np.sign(f0))
        if t < 1e299:
            hits += 1
            q = p + t * d
            g = np.linalg.norm(S.normal(kind, c, *q))
            assert abs(S.value(kind, c, *q)) <= 1e-8 * max(
                1.0, g * np.linalg.norm(q))
    assert hits > 20


@pytest.mark.parametrize("surf", TORI)
def test_torus_nearest_is_the_true_distance(surf):
    """Exact: never larger than the distance to a fine grid of points on
    the torus, and within the grid spacing of it."""
    kind, c = _surf(surf)
    ax = {S.X_TORUS: 0, S.Y_TORUS: 1, S.Z_TORUS: 2}[kind]
    a, b, cc = c[3], c[4], c[5]
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 721)[:-1],
                         np.linspace(0, 2 * np.pi, 721)[:-1], indexing="ij")
    rho, h = a + cc * np.cos(ph), b * np.sin(ph)
    pts = np.empty(th.shape + (3,))
    o = [k for k in range(3) if k != ax]
    pts[..., o[0]] = rho * np.cos(th)
    pts[..., o[1]] = rho * np.sin(th)
    pts[..., ax] = h
    pts = pts.reshape(-1, 3) + c[:3]
    for p in RNG.uniform(-8, 8, (150, 3)):
        near = S.nearest(kind, c, *p)
        brute = np.sqrt(np.min(np.sum((pts - p) ** 2, axis=1)))
        assert near <= brute + 1e-9
        assert brute - near < 0.06


@pytest.mark.parametrize("surf", QUADRICS)
def test_quadric_nearest_is_a_lower_bound(surf):
    """No formula exists, so a lower bound: never larger than the distance
    to any point found on the surface (by rays from all over)."""
    kind, c = _surf(surf)
    pts = []
    while len(pts) < 20000:
        p = RNG.uniform(-12, 12, 3)
        d = RNG.normal(size=3)
        d /= np.linalg.norm(d)
        t = S.ray(kind, c, *p, *d)
        if t < 60.0:
            pts.append(p + t * d)
    pts = np.array(pts)
    for p in RNG.uniform(-6, 6, (300, 3)):
        near = S.nearest(kind, c, *p)
        assert 0.0 <= near <= np.min(np.linalg.norm(pts - p, axis=1)) + 1e-9


def test_quadric_nearest_is_exact_for_a_plane_and_a_sphere():
    plane = openmc.Quadric(g=1.0, h=2.0, j=-2.0, k=-3.0)      # |n| = 3
    sphere = openmc.Quadric(a=1.0, b=1.0, c=1.0, k=-4.0)      # r = 2
    for p in RNG.uniform(-5, 5, (100, 3)):
        kp, cp = _surf(plane)
        assert S.nearest(kp, cp, *p) == pytest.approx(
            abs(p @ [1, 2, -2] - 3) / 3, rel=1e-12)
        ks, cs = _surf(sphere)
        d = abs(np.linalg.norm(p) - 2.0)
        # the bound is tight near the sphere and safe away from it
        assert S.nearest(ks, cs, *p) <= d + 1e-12
        if d < 0.05:
            assert S.nearest(ks, cs, *p) > 0.95 * d


@pytest.mark.parametrize("surf", SHAPES)
def test_normal_is_the_gradient(surf):
    kind, c = _surf(surf)
    h = 1e-6
    for p in RNG.uniform(-6, 6, (50, 3)):
        g = np.array(S.normal(kind, c, *p))
        fd = np.array([(S.value(kind, c, *(p + h * e)) -
                        S.value(kind, c, *(p - h * e))) / (2 * h)
                       for e in np.eye(3)])
        assert np.allclose(g, fd, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------- lattices
def _find_cell(model, p):
    found = [x for x in model.geometry.find(p.copy())
             if isinstance(x, openmc.Cell)]
    return found[-1].id if found else None


def test_rect_lattice_outer_universe_is_tiled_like_openmc():
    """Outside a lattice's array OpenMC fills each would-be element with
    the outer universe in that element's own coordinates (before v1.4.1 we
    used one frame for all of them: right only for a uniform outer)."""
    mats, m = common.one_group_materials("outer", {"a": (1.0, 0.1),
                                                   "b": (1.0, 0.5)})
    dot = openmc.Sphere(r=0.4)
    outer = openmc.Universe(cells=[openmc.Cell(fill=m["b"], region=-dot),
                                   openmc.Cell(fill=m["a"], region=+dot)])
    inner = openmc.Universe(cells=[openmc.Cell(fill=m["a"])])
    lat = openmc.RectLattice()
    lat.lower_left, lat.pitch = (-1.0, -1.0), (1.0, 1.0)
    lat.universes = [[inner, inner], [inner, inner]]
    lat.outer = outer
    box = openmc.model.RectangularParallelepiped(-5, 5, -5, 5, -1, 1,
                                                 boundary_type="vacuum")
    model = openmc.Model(openmc.Geometry([openmc.Cell(fill=lat,
                                                      region=-box)]), mats,
                         common.settings(common.point_source((0, 0, 0))))
    pr = openmc_import.load(model)
    pts = RNG.uniform([-5, -5, -1], [5, 5, 1], (3000, 3))
    cells, _ = G.locate_many(pr.geom, pts)
    assert np.array_equal(pr.cell_ids[cells],
                          [_find_cell(model, p) for p in pts])
    # and the distance field sees the dot at a would-be element's centre
    d = G.nearest_many(pr.geom, np.array([[3.5, 3.5, 0.0]]))[0]
    assert d == pytest.approx(0.4)


# ------------------------------------------------- distributed materials
def _distrib_model():
    """A distributed cell reached three ways: through a universe cell, a
    2 x 2 lattice, and a translated universe cell (6 instances)."""
    mats, m = common.one_group_materials(
        "distrib", {f"m{i}": (1.0, 0.1 * (i + 1)) for i in range(4)})
    s = openmc.ZCylinder(r=0.3)
    pin = openmc.Cell(name="pin", region=-s)
    pin.fill = [m["m0"], m["m1"], None, m["m2"], m["m3"], m["m1"]]
    u = openmc.Universe(cells=[pin, openmc.Cell(fill=m["m0"], region=+s)])
    lat = openmc.RectLattice()
    lat.lower_left, lat.pitch = (0.0, 0.0), (1.0, 1.0)
    lat.universes = [[u, u], [u, u]]
    xa, xb, xc = (openmc.XPlane(v) for v in (-1.0, 0.0, 2.0))
    xd = openmc.XPlane(3.0)
    y0, y1 = openmc.YPlane(-0.5), openmc.YPlane(2.0)
    z0 = openmc.ZPlane(-1.0, boundary_type="vacuum")
    z1 = openmc.ZPlane(1.0, boundary_type="vacuum")
    for p in (xa, xd, y0, y1):
        p.boundary_type = "vacuum"
    slab = +y0 & -y1 & +z0 & -z1
    a = openmc.Cell(fill=u, region=+xa & -xb & slab)
    b = openmc.Cell(fill=lat, region=+xb & -xc & slab)
    c = openmc.Cell(fill=u, region=+xc & -xd & slab)
    a.translation = (-0.5, 0.5, 0.0)
    c.translation = (2.5, 1.0, 0.0)
    model = openmc.Model(openmc.Geometry([a, b, c]), mats,
                         common.settings(common.point_source((0.5, 0.5, 0))))
    return model, pin


def test_distributed_materials_follow_openmcs_instance_order():
    model, pin = _distrib_model()
    pr = openmc_import.load(model)
    model.geometry.determine_paths()
    mat_index = {mid: i for i, mid in enumerate(pr.mats.ids)}
    pts = RNG.uniform([-1, -0.5, -1], [3, 2, 1], (4000, 3))
    _, mats = G.locate_many(pr.geom, pts)
    checked = set()
    for p, ours in zip(pts, mats):
        path = model.geometry.find(p.copy())
        if path[-1] is not pin:
            continue
        s = ""
        for x in path:
            if isinstance(x, openmc.Universe):
                s += f"u{x.id}"
            elif isinstance(x, openmc.Cell):
                s += f"->c{x.id}->"
            else:
                lat, idx = x
                s += f"l{lat.id}({','.join(str(i) for i in idx)})->"
        j = pin.paths.index(s.rstrip("->"))
        want = pin.fill[j]
        assert ours == (G.VOID if want is None else mat_index[want.id])
        checked.add(j)
    assert checked == set(range(6))


def test_wrong_number_of_distributed_materials_is_refused():
    model, pin = _distrib_model()
    pin.fill = pin.fill[:5]
    with pytest.raises(Unsupported, match="5 materials for 6 instances"):
        openmc_import.load(model)


# --------------------------------------------------- rotational periodic
def _cylinder(quarter):
    mats, m = common.one_group_materials("rot", {"a": (1.0, 0.1)})
    wall = openmc.ZCylinder(r=5.0, boundary_type="vacuum")
    zb = openmc.ZPlane(-3.0, boundary_type="vacuum")
    zt = openmc.ZPlane(3.0, boundary_type="vacuum")
    region = -wall & +zb & -zt
    if quarter:
        x0 = openmc.XPlane(0.0, boundary_type="periodic")
        y0 = openmc.YPlane(0.0, boundary_type="periodic")
        x0.periodic_surface = y0
        region &= +x0 & +y0
        src = common.box_source((0.0, 0.0, -1.0), (2.0, 2.0, 1.0))
    else:
        src = common.box_source((-2.0, -2.0, -1.0), (2.0, 2.0, 1.0))
    return openmc.Model(openmc.Geometry([openmc.Cell(fill=m["a"],
                                                     region=region)]),
                        mats, common.settings(src))


def test_rotational_quarter_equals_the_full_cylinder():
    """A quarter with rotational walls is the whole problem when the
    problem has the quarter's symmetry: same leakage within error bars."""
    full = transport.run(openmc_import.load(_cylinder(False)), 50_000, 4,
                         seed=1)
    quarter = transport.run(openmc_import.load(_cylinder(True)), 50_000, 4,
                            seed=2)
    z = (full.leakage - quarter.leakage) / np.hypot(full.leakage_se,
                                                    quarter.leakage_se)
    assert abs(z) < 4


def test_rotation_angle_matches_openmcs_rule():
    pr = openmc_import.load(_cylinder(True))
    a = pr.geom.surf_angle[pr.geom.surf_bc == G.BC_PERIODIC]
    assert sorted(a) == pytest.approx([-np.pi / 2, np.pi / 2])


# --------------------------------------------------------------- refusals
def test_rotational_wall_off_the_z_axis_is_refused():
    model = _cylinder(True)
    for s in model.geometry.get_all_surfaces().values():
        if isinstance(s, openmc.YPlane):
            s.y0 = 0.5
    with pytest.raises(Unsupported, match="z axis"):
        openmc_import.load(model)


def test_horn_torus_is_accepted():
    """Refused before v1.6.1 (surfaces.py: the self-intersecting tube's
    inner roots are skipped; checks/gapcheck.py runs one vs OpenMC)."""
    openmc_import._surface(openmc.ZTorus(a=1.0, b=1.0, c=1.5))
