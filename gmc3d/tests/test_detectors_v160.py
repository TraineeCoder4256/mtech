"""v1.6: point detectors (core/detectors.py) and `run.py check`.

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import openmc
import openmc.stats as st
import pytest

from ball.oracle import Oracle
from checks import detcheck
from core import model_check, openmc_import, transport
from core.openmc_import import Unsupported
from problems import PROBLEMS, common


def test_exact_uncollided_flux_through_two_shells():
    model, pts = detcheck.exact_model()
    r = transport.run(openmc_import.load(model), 1000, 2, seed=1)
    for name, p in pts.items():
        assert r.tallies[name]["mean"][0, 0] == pytest.approx(
            detcheck.exact_value(p), rel=1e-10)
        assert r.tallies[name]["se"][0, 0] < 1e-8 * \
            r.tallies[name]["mean"][0, 0]


def test_a_line_through_a_vacuum_gap_scores_nothing():
    """Two spheres apart, each with a vacuum boundary, the source in one
    and the detector in the other: no straight line stays inside."""
    mats, m = common.one_group_materials("det_gap", {"a": (0.5, 0.1)})
    s1 = openmc.Sphere(r=3.0, boundary_type="vacuum")
    s2 = openmc.Sphere(x0=10.0, r=3.0, boundary_type="vacuum")
    model = openmc.Model(openmc.Geometry([
        openmc.Cell(fill=m["a"], region=-s1),
        openmc.Cell(fill=m["a"], region=-s2)]), mats,
        common.settings(common.point_source((0, 0, 0))),
        openmc.Tallies([common.flux_tally("total")]))
    common.point_detector(model, "far", (10.0, 0.0, 0.0), r0=0.2)
    common.point_detector(model, "near", (1.0, 0.0, 0.0), r0=0.2)
    r = transport.run(openmc_import.load(model), 2000, 4, seed=2)
    assert r.tallies["far"]["mean"][0, 0] == 0.0
    assert r.tallies["near"]["mean"][0, 0] > 0.0


def test_balls_and_plain_mc_agree():
    """The method's claim, on a small run (the full check is
    `run.py detcheck`): a detector in a void cavity of a scattering
    sphere, with and without balls."""
    model = detcheck.scatter_model()
    pr = openmc_import.load(model)
    a = transport.run(pr, 20_000, 8, seed=3)
    b = transport.run(pr, 20_000, 8, seed=4, backend=Oracle(), r_star=1.0)
    assert b.counters["balls"] > 0
    for name in ("cavity", "outside", "medium"):
        x, sx = a.tallies[name]["mean"][0, 0], a.tallies[name]["se"][0, 0]
        y, sy = b.tallies[name]["mean"][0, 0], b.tallies[name]["se"][0, 0]
        assert abs(x - y) < 4 * np.hypot(sx, sy), name


def test_r0_bounds_a_score_inside_the_radius():
    """A source right at the detector: with r0 the score is the averaged
    kernel 3 (1 - exp(-s r0)) / (4 pi s r0^3), finite."""
    model, _ = detcheck.exact_model()
    model.gmc3d_detectors = [("at_source", (0.0, 0.0, 0.0), 0.5)]
    r = transport.run(openmc_import.load(model), 1000, 2, seed=1)
    s, r0 = 0.2, 0.5
    assert r.tallies["at_source"]["mean"][0, 0] == pytest.approx(
        3 * (1 - np.exp(-s * r0)) / (4 * np.pi * s * r0 ** 3), rel=1e-10)


def test_runs_without_detectors_are_unchanged():
    """Bit for bit: a detector-free run takes no detector code path."""
    pr = openmc_import.load(PROBLEMS["sphere"](mesh=4))
    a = transport.run(pr, 5000, 2, seed=9)
    b = transport.run(pr, 5000, 2, seed=9)
    assert a.leakage == b.leakage
    assert "detector" not in {t["filter"] for t in a.tallies.values()}


@pytest.mark.parametrize("what", ["reflective", "mono", "outside", "twice"])
def test_detector_refusals(what):
    model = PROBLEMS["sphere"](box=(what == "reflective"), mesh=4)
    if what == "mono":
        model.settings.source = common.point_source((0, 0, 0), (1, 0, 0))
    pos = (50.0, 0.0, 0.0) if what == "outside" else (1.0, 0.0, 0.0)
    common.point_detector(model, "d", pos)
    if what == "twice":
        common.point_detector(model, "d", (2.0, 0.0, 0.0))
    with pytest.raises(Unsupported):
        openmc_import.load(model)


# ------------------------------------------------------------- run.py check
def _bad_model():
    mats, m = common.one_group_materials("chk_bad", {"a": (0.5, 0.1)})
    s1 = openmc.Sphere(r=10, boundary_type="vacuum")
    s1.albedo = 0.5
    t = openmc.ZTorus(a=1.0, b=2.0, c=2.0)
    st_ = common.settings(common.point_source((0, 0, 0)))
    st_.survival_biasing = True
    t1 = openmc.Tally(name="coll")
    t1.estimator, t1.scores = "collision", ["flux"]
    t2 = openmc.Tally(name="fiss")
    t2.scores = ["fission"]
    return openmc.Model(openmc.Geometry([
        openmc.Cell(fill=m["a"], region=-s1 & +t),
        openmc.Cell(fill=m["a"], region=-t)]), mats, st_,
        openmc.Tallies([t1, t2]))


def test_check_lists_every_refusal_at_once():
    rep = model_check.check(_bad_model())
    text = "\n".join(rep.lines())
    assert not rep.loads
    for bit in ("albedo 0.5", "torus with c", "'collision' estimator",
                "['fission']"):
        assert bit in text, bit
    assert any("survival_biasing" in w for w in rep.warned)


def test_check_accepts_every_validation_problem():
    for name in PROBLEMS:
        rep = model_check.check(PROBLEMS[name]())
        assert rep.loads and not rep.refused, name


def test_check_reads_xml_and_python_files(tmp_path):
    PROBLEMS["slab"]().export_to_model_xml(tmp_path / "model.xml")
    assert model_check.check(model_check.load_model(
        str(tmp_path / "model.xml"))).loads
    f = tmp_path / "m.py"
    f.write_text("from problems import PROBLEMS\nmodel = PROBLEMS['cask']()\n")
    assert model_check.check(model_check.load_model(str(f))).loads
