"""v1.1.3: no silent gaps in settings and tallies; scatter and total
scores; 1D, 2D and rectilinear meshes.

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import warnings

import numpy as np
import openmc
import pytest

from ball.oracle import Oracle
from core import openmc_import, transport
from core.openmc_import import IgnoredSetting, Unsupported
from problems import PROBLEMS


# ---------------------------------------------------------------- settings
def test_settings_that_change_only_noise_are_named_in_a_warning():
    model = PROBLEMS["slab"]()
    model.settings.survival_biasing = True
    model.settings.cutoff = {"weight": 0.3, "weight_avg": 0.9}
    model.settings.temperature = {"default": 600.0}
    model.settings.max_lost_particles = 5
    with pytest.warns(IgnoredSetting) as rec:
        pr = openmc_import.load(model)
    said = " ".join(str(w.message) for w in rec)
    for tag in ("survival_biasing", "cutoff", "temperature_default",
                "max_lost_particles"):
        assert tag in said
    assert len(pr.warnings) == len(rec)


def test_plain_problems_load_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", IgnoredSetting)
        for name, build in PROBLEMS.items():
            assert openmc_import.load(build()).warnings == [], name


@pytest.mark.parametrize("change, words", [
    (lambda s: setattr(s, "run_mode", "eigenvalue"), "run_mode"),
    (lambda s: setattr(s, "photon_transport", True), "photon"),
])
def test_settings_that_change_the_problem_are_refused(change, words):
    model = PROBLEMS["slab"]()
    change(model.settings)
    with pytest.raises(Unsupported, match=words):
        openmc_import.load(model)


# ----------------------------------------------------------------- tallies
def _slab_tally(**attrs):
    model = PROBLEMS["slab"]()
    t = openmc.Tally(name="extra")
    t.scores = ["flux"]
    for k, v in attrs.items():
        setattr(t, k, v)
    model.tallies.append(t)
    return model


@pytest.mark.parametrize("attrs, words", [
    ({"nuclides": ["U235"]}, "nuclides"),
    ({"derivative": openmc.TallyDerivative(variable="density",
                                           material=1)}, "derivative"),
    ({"scores": ["events"]}, "events"),
    ({"scores": ["heating"]}, "heating"),
])
def test_tally_settings_that_change_the_answer_are_refused(attrs, words):
    with pytest.raises(Unsupported, match=words):
        openmc_import.load(_slab_tally(**attrs))


@pytest.mark.parametrize("attrs", [{"estimator": "collision"},
                                   {"estimator": "analog"}])
def test_other_estimators_are_named_in_a_warning(attrs):
    """Same expected value, other noise: since v1.6.1 a warning, not a
    refusal."""
    with pytest.warns(IgnoredSetting, match="estimator"):
        openmc_import.load(_slab_tally(**attrs))


def test_fission_scores_are_zero():
    pr = openmc_import.load(_slab_tally(scores=["flux", "fission"]))
    r = transport.run(pr, 2000, 2, seed=1)
    from core.tallies import ALL_SCORES
    assert r.tallies["extra"]["mean"][0, ALL_SCORES.index("fission")] == 0


def test_tally_triggers_are_named_in_a_warning():
    trig = openmc.Trigger("rel_err", 0.01)
    trig.scores = ["flux"]
    with pytest.warns(IgnoredSetting, match="triggers"):
        openmc_import.load(_slab_tally(triggers=[trig]))


def test_track_length_estimator_and_total_nuclide_are_accepted():
    openmc_import.load(_slab_tally(estimator="tracklength",
                                   nuclides=["total"]))


def test_mesh_born_filter_is_not_a_mesh_filter():
    """Before v1.1.3 it passed the isinstance(MeshFilter) check and was
    scored as an ordinary mesh: silently wrong.  Refused from v1.1.3,
    scored by birth element since v1.6.1."""
    m = openmc.RegularMesh()
    m.lower_left, m.upper_right, m.dimension = (0, -1, -1), (60, 1, 1), \
        (6, 1, 1)
    pr = openmc_import.load(_slab_tally(filters=[openmc.MeshBornFilter(m)]))
    assert [f.kind for f in pr.tallies[-1].filters] == ["meshborn"]
    assert pr.tallies[-1].ext


def test_mesh_surface_filter_on_a_cylindrical_mesh_is_accepted():
    m = openmc.CylindricalMesh(r_grid=[0, 1, 2], z_grid=[0, 1],
                               phi_grid=[0, 2 * np.pi])
    model = _slab_tally(filters=[openmc.MeshSurfaceFilter(m)],
                        scores=["current"])
    assert openmc_import.load(model).tallies[-1].ext


# ------------------------------------------------- scores and mesh shapes
@pytest.fixture(scope="module")
def cask_runs():
    pr = openmc_import.load(PROBLEMS["cask"]())
    return [transport.run(pr, 20000, 4, seed=7, backend=be, r_star=1.0)
            for be in (None, Oracle())]


def test_total_is_absorption_plus_scatter_in_every_bin(cask_runs):
    for r in cask_runs:
        t = r.tallies["material"]["mean"]
        assert np.allclose(t[:, 3], t[:, 1] + t[:, 2], rtol=1e-12,
                           atol=1e-300)
        assert np.all(t[:, 2] <= t[:, 3])


def test_2d_and_rectilinear_meshes_hold_all_the_flux(cask_runs):
    """The 2D x-y mesh is unbounded in z and the rectilinear grid covers
    the same box as the 3D mesh, and both contain the whole geometry, so
    their flux sums to the total, with balls or without."""
    for r in cask_runs:
        total = r.tallies["total"]["mean"][0, 0]
        for name in ("mesh", "plan2d", "duct_grid"):
            s = r.tallies[name]["mean"][:, 0].sum()
            assert abs(s - total) < 1e-9 * total, name


def test_1d_mesh_matches_the_3d_depth_mesh_bin_by_bin():
    """The slab's 1D mesh (30 bins) against its 3D mesh (60 bins, 1 x 1
    across): merging pairs of 3D bins gives the 1D bins exactly."""
    pr = openmc_import.load(PROBLEMS["slab"]())
    r = transport.run(pr, 20000, 4, seed=8, backend=Oracle(), r_star=1.0)
    one = r.tallies["depth1d"]["mean"]
    three = r.tallies["depth"]["mean"][:, :2]
    assert np.allclose(one[:, :2], three.reshape(30, 2, 2).sum(axis=1),
                       rtol=1e-9)


def test_rectilinear_grid_equals_regular_grid_with_the_same_edges():
    """A rectilinear mesh whose edges are a regular grid's, nudged by
    1e-9 cm so the binary-search path runs, scores the same flux."""
    model = PROBLEMS["slab"]()
    rm = openmc.RectilinearMesh()
    x = np.linspace(0.0, 60.0, 61)
    x[1:-1] += 1e-9 * np.cos(np.arange(59))
    rm.x_grid, rm.y_grid, rm.z_grid = x, [-100.0, 100.0], [-100.0, 100.0]
    t = openmc.Tally(name="rect")
    t.filters, t.scores = [openmc.MeshFilter(rm)], ["flux"]
    model.tallies.append(t)
    pr = openmc_import.load(model)
    r = transport.run(pr, 20000, 4, seed=9, backend=Oracle(), r_star=1.0)
    a, b = r.tallies["rect"]["mean"][:, 0], r.tallies["depth"]["mean"][:, 0]
    assert np.allclose(a, b, rtol=1e-5)
