"""v1.4.0: cylinders and cones on any axis, white and periodic boundaries.

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import openmc
import pytest
from numba import njit
from scipy import stats

from core import geometry as G
from core import openmc_import, transport
from core import surfaces as S
from core.openmc_import import Unsupported
from problems import PROBLEMS, common

RNG = np.random.default_rng(4)


def _coef(surf):
    _, coef, _ = openmc_import._surface(surf)
    return coef


SHAPES = [
    (S.CYLINDER, openmc.Cylinder(x0=1.0, y0=-2.0, z0=0.5, r=2.5,
                                 dx=1.0, dy=2.0, dz=-1.0)),
    (S.CONE, openmc.Cone(x0=0.5, y0=1.0, z0=-1.0, r2=0.3,
                         dx=-1.0, dy=0.5, dz=2.0)),
]


@pytest.mark.parametrize("kind, surf", SHAPES)
def test_value_has_openmcs_sign_and_shape(kind, surf):
    """Same zero set and same sides as OpenMC's own (quadric) form: the
    two functions are positive multiples of each other."""
    c = _coef(surf)
    pts = RNG.uniform(-8, 8, (2000, 3))
    ours = np.array([S.value(kind, c, *p) for p in pts])
    ref = np.array([surf.evaluate(p) for p in pts])
    ratio = ours / ref
    assert np.all(ratio > 0)
    assert np.allclose(ratio, ratio[0], rtol=1e-9)


@pytest.mark.parametrize("kind, surf", SHAPES)
def test_ray_lands_on_the_surface_and_skips_nothing(kind, surf):
    c = _coef(surf)
    for _ in range(500):
        p = RNG.uniform(-8, 8, 3)
        d = RNG.normal(size=3)
        d /= np.linalg.norm(d)
        t = S.ray(kind, c, *p, *d)
        # no sign change of f before t (sampled finely)
        ts = np.linspace(0, min(t, 40.0), 400)[1:-1]
        f0 = S.value(kind, c, *p)
        fs = np.array([S.value(kind, c, *(p + s * d)) for s in ts])
        assert np.all(np.sign(fs) == np.sign(f0)) or abs(f0) < 1e-9
        if t < 1e299:
            q = p + t * d
            g = np.array(S.normal(kind, c, *q))
            assert abs(S.value(kind, c, *q)) <= 1e-8 * max(
                1.0, np.linalg.norm(g) * np.linalg.norm(q))


@pytest.mark.parametrize("kind, surf", SHAPES)
def test_nearest_is_the_true_distance(kind, surf):
    """Never larger than the distance to any surface point (the balls
    depend on that), and reached: some surface point is that far away."""
    c = _coef(surf)
    # surface points on a fine grid along the axis and around it
    a, d = c[0:3], c[4:7]
    e1 = np.cross(d, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    h, phi = np.meshgrid(np.linspace(-30, 30, 3001),
                         np.linspace(0, 2 * np.pi, 361)[:-1], indexing="ij")
    rad = np.abs(h) * np.sqrt(c[3]) if kind == S.CONE else np.full_like(h, c[3])
    surf_pts = (a + h[..., None] * d + rad[..., None] *
                (np.cos(phi)[..., None] * e1 + np.sin(phi)[..., None] * e2)
                ).reshape(-1, 3)
    for p in RNG.uniform(-6, 6, (200, 3)):
        near = S.nearest(kind, c, *p)
        brute = np.min(np.linalg.norm(surf_pts - p, axis=1))
        assert near <= brute + 1e-9
        assert brute - near < 0.05                    # grid spacing


@pytest.mark.parametrize("kind, surf", SHAPES)
def test_normal_is_the_gradient(kind, surf):
    c = _coef(surf)
    h = 1e-6
    for p in RNG.uniform(-6, 6, (50, 3)):
        g = np.array(S.normal(kind, c, *p))
        fd = np.array([(S.value(kind, c, *(p + h * e)) -
                        S.value(kind, c, *(p - h * e))) / (2 * h)
                       for e in np.eye(3)])
        assert np.allclose(g, fd, rtol=1e-5, atol=1e-5)


def test_tilted_problem_locates_like_openmc():
    model = PROBLEMS["tilted"]()
    pr = openmc_import.load(model)
    pts = RNG.uniform([-10, -10, -15], [10, 10, 15], (3000, 3))
    cells, _ = G.locate_many(pr.geom, pts)
    ref = [model.geometry.find(p)[-1].id for p in pts]
    assert np.array_equal(pr.cell_ids[cells], ref)


# ------------------------------------------------------------ boundaries
@njit
def _white_draws(g, k, P, n):
    out = np.empty((n, 3))
    for j in range(n):
        _, _, _, u, v, w = transport._boundary(g, k, P, 0, 0.0, 10.0, 0.0,
                                               0.0, 1.0, 0.0)
        out[j] = u, v, w
    return out


def test_white_reflection_follows_the_cosine_law():
    """Leaving through y = +10: the new direction points back in (-y),
    with mu = -u distributed as P(mu < m) = m^2, and isotropic azimuth."""
    model = PROBLEMS["tilted"]()
    pr = openmc_import.load(model)
    g = pr.geom
    k = [j for j in np.flatnonzero(g.surf_bc == G.BC_WHITE)
         if g.surf_coef[j, 0] > 0][0]                       # y = +10
    P = transport._particles(1)
    P.seed[0] = 12345
    d = _white_draws(g, k, P, 50_000)
    mu = -d[:, 1]
    assert np.all(mu > 0)
    assert stats.kstest(mu ** 2, "uniform").pvalue > 1e-3
    phi = np.arctan2(d[:, 0], d[:, 2])
    assert stats.kstest((phi + np.pi) / (2 * np.pi), "uniform").pvalue > 1e-3


def _slab_with_walls(kind):
    model = PROBLEMS["slab"]()
    for s in model.geometry.get_all_surfaces().values():
        if s.boundary_type == "reflective":
            s.boundary_type = kind
    return model


def test_periodic_walls_equal_reflective_walls_in_a_uniform_slab():
    """The slab is uniform in y and z and its source fills y and z, so
    periodic side walls and mirror side walls give the same answer (an
    infinite slab either way); checked within the error bars."""
    a = transport.run(openmc_import.load(_slab_with_walls("reflective")),
                      50_000, 4, seed=1)
    b = transport.run(openmc_import.load(_slab_with_walls("periodic")),
                      50_000, 4, seed=2)
    z = (a.leakage - b.leakage) / np.hypot(a.leakage_se, b.leakage_se)
    assert abs(z) < 4


def test_a_closed_white_box_absorbs_everything():
    """No vacuum anywhere: every source particle ends absorbed, so the
    absorption tally is 1 per source particle (to rounding)."""
    mats, m = common.one_group_materials("whitebox", {"a": (0.5, 0.05)})
    s = openmc.Sphere(r=5.0, boundary_type="white")
    cell = openmc.Cell(fill=m["a"], region=-s)
    t = openmc.Tally(name="total")
    t.scores = ["absorption"]
    src = common.point_source((1.0, 0.0, 0.0))
    model = openmc.Model(openmc.Geometry([cell]), mats,
                         common.settings(src), openmc.Tallies([t]))
    r = transport.run(openmc_import.load(model), 5000, 2, seed=1)
    assert r.leakage == 0.0
    # roulette keeps absorption unbiased, not exact: within 5 se of 1
    m_, se = r.tallies["total"]["mean"][0, 1], r.tallies["total"]["se"][0, 1]
    assert abs(m_ - 1.0) < 5 * se + 1e-9


# --------------------------------------------------------------- refusals
def _refuse(model, words):
    with pytest.raises(Unsupported, match=words):
        openmc_import.load(model)


def test_rotational_periodic_is_refused():
    model = PROBLEMS["slab"]()
    walls = [s for s in model.geometry.get_all_surfaces().values()
             if isinstance(s, openmc.YPlane)]
    p = openmc.Plane(a=1.0, b=1.0, c=0.0, d=0.0)
    for s in walls:
        s.boundary_type = "periodic"
    walls[0].periodic_surface = p
    p.boundary_type = "periodic"
    _refuse(model, "not parallel")


def test_albedo_is_kept():
    """Refused before v1.6.1; now the weight sent back is multiplied by
    it, the rest counted as leakage (as OpenMC)."""
    model = PROBLEMS["slab"]()
    for s in model.geometry.get_all_surfaces().values():
        if s.boundary_type == "reflective":
            s.albedo = 0.5
    pr = openmc_import.load(model)
    assert np.any(pr.geom.surf_shift[:, 4] == 0.5)


def test_current_on_a_white_surface_is_zero():
    """Refused in v1.4.0; since v1.5 scored as OpenMC does: exactly 0."""
    model = PROBLEMS["tilted"]()
    y = [s for s in model.geometry.get_all_surfaces().values()
         if s.boundary_type == "white"]
    t = openmc.Tally(name="w")
    t.filters, t.scores = [openmc.SurfaceFilter(y)], ["current"]
    model.tallies.append(t)
    r = transport.run(openmc_import.load(model), 5000, 2, seed=1)
    assert (r.tallies["w"]["mean"][:, 0] == 0.0).all()
