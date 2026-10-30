"""v1.6 edge cases (SRS V1-ROB-5): inputs at the edge of what makes sense.

Each must either run with a sane answer (no crash, no NaN, no lost
particles, plain MC and exact balls agreeing, exact values where they are
known) or be refused cleanly while the model is read.

    empty tallies              a cell no particle reaches, a mesh outside
                               the geometry, an energy bin outside the group
    source on a surface        a point source exactly on an inner sphere
    balls touching a boundary  a gap of 1e-6 cm between two spheres
    zero-volume cells          an empty region, a shell of zero thickness
    very thin and very thick   sigma 1e-9 /cm (a near-void) and 100 /cm
    materials
    no way out                 nothing absorbs and nothing leaks: refused

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import openmc
import pytest

from ball.oracle import Oracle
from core import openmc_import, transport
from core.openmc_import import Unsupported
from problems import common


def _run(model, balls=False, n=20_000, batches=8, seed=3):
    pr = openmc_import.load(model)
    r = transport.run(pr, n, batches, seed=seed,
                      backend=Oracle() if balls else None, r_star=1.0)
    assert r.counters["lost"] == 0
    for t in r.tallies.values():
        assert np.all(np.isfinite(t["mean"])) and np.all(np.isfinite(t["se"]))
    return r


def _z(a, sa, b, sb):
    return (a - b) / max(np.hypot(sa, sb), 1e-300)


def _model(cells, mats, src, tallies=()):
    return openmc.Model(openmc.Geometry(cells), mats, common.settings(src),
                        openmc.Tallies(list(tallies)))


# ---------------------------------------------------------- empty tallies
def test_empty_tallies_are_zero_not_nan():
    mats, m = common.one_group_materials("edge_empty", {"a": (1.0, 0.1)})
    s1 = openmc.Sphere(r=5.0, boundary_type="vacuum")
    s2 = openmc.Sphere(x0=20.0, r=3.0, boundary_type="vacuum")
    lit = openmc.Cell(fill=m["a"], region=-s1)
    dark = openmc.Cell(fill=m["a"], region=-s2)    # never reached
    far = common.flux_tally("far", common.mesh_filter((50, 50, 50),
                                                      (60, 60, 60), (2, 2, 2)))
    cells = common.flux_tally("cells", openmc.CellFilter([lit, dark]))
    en = common.flux_tally("energy", openmc.EnergyFilter([1e8, 2e8]))
    for balls in (False, True):
        r = _run(_model([lit, dark], mats, common.point_source((0, 0, 0)),
                        [far, cells, en]), balls)
        assert np.all(r.tallies["far"]["mean"] == 0.0)
        assert np.all(r.tallies["far"]["se"] == 0.0)
        assert r.tallies["cells"]["mean"][1, 0] == 0.0
        assert r.tallies["cells"]["mean"][0, 0] > 0.0
        assert np.all(r.tallies["energy"]["mean"] == 0.0)


# --------------------------------------------------- source on a surface
@pytest.mark.parametrize("balls", [False, True])
def test_point_source_exactly_on_an_inner_surface(balls):
    """Born exactly on the sphere between two materials: the answer must
    be continuous, so it matches a source 1e-7 cm to either side."""
    def build(x):
        mats, m = common.one_group_materials("edge_onsurf",
                                             {"a": (1.0, 0.2),
                                              "b": (0.5, 0.05)})
        inner = openmc.Sphere(r=3.0)
        outer = openmc.Sphere(r=8.0, boundary_type="vacuum")
        return _model([openmc.Cell(fill=m["a"], region=-inner),
                       openmc.Cell(fill=m["b"], region=+inner & -outer)],
                      mats, common.point_source((x, 0.0, 0.0)),
                      [common.flux_tally("total")])
    on = _run(build(3.0), balls, seed=5)
    for x in (3.0 - 1e-7, 3.0 + 1e-7):
        off = _run(build(x), balls, seed=6)
        assert abs(_z(on.leakage, on.leakage_se, off.leakage,
                      off.leakage_se)) < 4


# ------------------------------------------- balls touching a boundary
def test_balls_in_a_hair_thin_gap():
    """Two shells 1e-6 cm apart: balls must stop short of both and nothing
    may be lost, and the answer equals the same geometry without the
    gap's surface."""
    mats, m = common.one_group_materials("edge_gap", {"a": (2.0, 0.05)})
    s1, s2 = openmc.Sphere(r=4.0), openmc.Sphere(r=4.0 + 1e-6)
    out = openmc.Sphere(r=9.0, boundary_type="vacuum")
    split = _model([openmc.Cell(fill=m["a"], region=-s1),
                    openmc.Cell(fill=m["a"], region=+s1 & -s2),
                    openmc.Cell(fill=m["a"], region=+s2 & -out)],
                   mats, common.point_source((0.0, 0.0, 0.0)),
                   [common.flux_tally("total")])
    whole = _model([openmc.Cell(fill=m["a"], region=-openmc.Sphere(
        r=9.0, boundary_type="vacuum"))], mats,
        common.point_source((0.0, 0.0, 0.0)), [common.flux_tally("total")])
    a = _run(split, balls=True, seed=7)
    b = _run(whole, balls=False, seed=8)
    assert a.counters["balls"] > 0
    assert abs(_z(a.leakage, a.leakage_se, b.leakage, b.leakage_se)) < 4


# ----------------------------------------------------- zero-volume cells
def test_zero_volume_cells_score_nothing_and_change_nothing():
    mats, m = common.one_group_materials("edge_zero", {"a": (1.0, 0.1)})
    r5, r5b = openmc.Sphere(r=5.0), openmc.Sphere(r=5.0)
    out = openmc.Sphere(r=8.0, boundary_type="vacuum")
    p = openmc.XPlane(1.0)
    inside = openmc.Cell(fill=m["a"], region=-r5)
    shell = openmc.Cell(fill=m["a"], region=+r5 & -r5b)   # thickness 0
    empty = openmc.Cell(fill=m["a"], region=+p & -p & -out)  # no points
    rest = openmc.Cell(fill=m["a"], region=+r5b & -out)
    t = common.flux_tally("cells", openmc.CellFilter([inside, shell, empty,
                                                      rest]))
    ref = _model([openmc.Cell(fill=m["a"], region=-openmc.Sphere(
        r=8.0, boundary_type="vacuum"))], mats,
        common.point_source((0, 0, 0)), [common.flux_tally("total")])
    for balls in (False, True):
        r = _run(_model([inside, shell, empty, rest], mats,
                        common.point_source((0, 0, 0)), [t]), balls, seed=9)
        f = r.tallies["cells"]["mean"][:, 0]
        assert f[1] == 0.0 and f[2] == 0.0
        b = _run(ref, seed=10)
        assert abs(_z(r.leakage, r.leakage_se, b.leakage, b.leakage_se)) < 4


# ------------------------------------------- very thin, very thick media
def test_near_void_is_exact():
    """sigma_t 1e-9 /cm over 10 cm: leakage exp(-1e-8), flux = path."""
    mats, m = common.one_group_materials("edge_thin", {"a": (0.0, 1e-9)})
    cell = openmc.Cell(fill=m["a"], region=-openmc.Sphere(
        r=10.0, boundary_type="vacuum"))
    r = _run(_model([cell], mats, common.point_source((0, 0, 0)),
                    [common.flux_tally("total")]), n=2000, batches=2)
    assert abs(r.leakage - np.exp(-1e-8)) < 1e-14
    assert abs(r.tallies["total"]["mean"][0, 0] - 10.0) < 1e-6


def test_near_void_scatterer_runs_without_balls_misbehaving():
    """sigma_s 1e-9: almost no scatters, balls almost never fit (R tiny);
    plain MC and balls give leakage 1 to 1e-7."""
    mats, m = common.one_group_materials("edge_thin2", {"a": (1e-9, 0.0)})
    cell = openmc.Cell(fill=m["a"], region=-openmc.Sphere(
        r=10.0, boundary_type="vacuum"))
    for balls in (False, True):
        r = _run(_model([cell], mats, common.point_source((0, 0, 0)),
                        [common.flux_tally("total")]), balls, n=4000)
        assert abs(r.leakage - 1.0) < 1e-7


@pytest.mark.parametrize("sig", [(99.9, 0.1), (0.0, 100.0)])
def test_very_thick_material(sig):
    """sigma_t 100 /cm in a 2 cm sphere (200 mfp of radius): a strong
    scatterer (balls take over) and a strong absorber (exactly
    exp(-200) leaks).  Leakage + absorption = 1 within error."""
    mats, m = common.one_group_materials("edge_thick", {"a": sig})
    cell = openmc.Cell(fill=m["a"], region=-openmc.Sphere(
        r=2.0, boundary_type="vacuum"))
    model = _model([cell], mats, common.point_source((0, 0, 0)),
                   [common.flux_tally("total")])
    if sig[0] == 0.0:
        r = _run(model, n=2000, batches=2)
        assert r.leakage == pytest.approx(np.exp(-200.0), rel=1e-9)
        return
    a = _run(model, n=4000, batches=4, seed=11)
    b = _run(model, balls=True, n=4000, batches=4, seed=12)
    assert b.counters["balls"] > 0
    assert abs(_z(a.leakage, a.leakage_se, b.leakage, b.leakage_se)) < 4
    for r in (a, b):
        t = r.tallies["total"]
        bal = r.leakage + t["mean"][0, 1]
        assert abs(bal - 1.0) < 4 * np.hypot(r.leakage_se, t["se"][0, 1]) \
            + 1e-9


# ------------------------------------------------------------ no way out
def test_no_absorption_and_no_leakage_is_refused():
    """A pure scatterer in a reflective box: no history ever ends (OpenMC
    would run forever too), so it is refused while the model is read."""
    mats, m = common.one_group_materials("edge_trap", {"a": (1.0, 0.0)})
    planes = [openmc.XPlane(-1), openmc.XPlane(1), openmc.YPlane(-1),
              openmc.YPlane(1), openmc.ZPlane(-1), openmc.ZPlane(1)]
    for p in planes:
        p.boundary_type = "reflective"
    cell = openmc.Cell(fill=m["a"], region=+planes[0] & -planes[1] &
                       +planes[2] & -planes[3] & +planes[4] & -planes[5])
    with pytest.raises(Unsupported, match="never end"):
        openmc_import.load(_model([cell], mats,
                                  common.point_source((0, 0, 0))))
