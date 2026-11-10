"""v2.0: anisotropic scattering laws (core/scattering.py) as OpenMC samples
them, their import, and that balls stay out of anisotropic materials.

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import openmc
import pytest
from numba import njit

from ball.oracle import Oracle
from core import openmc_import, rng, scattering as SC, transport
from core.openmc_import import Unsupported
from problems import PROBLEMS, common


@njit(cache=True)
def _draw(ang, m, n, seed):
    state = rng.streams(np.uint64(seed), 0, 1)
    out = np.empty(n)
    for k in range(n):
        out[k] = SC.sample_mu(ang, m, state, 0)
    return out


LAWS = {
    "tabular_p3": ("legendre", [1.0, 0.6, 0.3, 0.1], {}),
    "tabular_clipped": ("legendre", [1.0, 0.9, 0.5, -0.4], {}),
    "tabular_9_points": ("legendre", [1.0, 0.9, 0.5, -0.4],
                         {"tab_points": 9}),
    "rejection_p3": ("legendre", [1.0, 0.6, 0.3, 0.1],
                     {"tab_enable": False}),
    "rejection_clipped": ("legendre", [1.0, 0.9, 0.5, -0.4],
                          {"tab_enable": False}),
    "histogram": ("histogram", [1, 1, 2, 3, 5, 8, 13, 20], {}),
    "tabular_format": ("tabular", [0.2, 0.3, 0.6, 1.2, 2.0], {}),
}


@pytest.mark.parametrize("name", list(LAWS))
def test_sampled_cosines_follow_the_density(name):
    """400k draws: Legendre moments 1..4 within 4 sigma of the density's,
    and a 40-bin histogram consistent with it (chi2 p > 1e-4)."""
    fmt, data, kw = LAWS[name]
    kind, v = SC.law(fmt, np.asarray(data, float), **kw)
    assert kind != SC.ISO
    ang = SC.pack([(kind, v)])
    mu = _draw(ang, 0, 400_000, 12345)
    assert mu.min() >= -1.0 and mu.max() <= 1.0
    mom = SC.moments(kind, v, 4)
    for l in range(1, 5):
        p = np.polynomial.legendre.legval(mu, np.eye(5)[l])
        assert abs(p.mean() - mom[l]) <= 4 * p.std() / np.sqrt(mu.size), l
    edges = np.linspace(-1, 1, 41)
    counts, _ = np.histogram(mu, edges)
    xs, ws = np.polynomial.legendre.leggauss(12)
    expect = []
    for a, b in zip(edges[:-1], edges[1:]):
        x = 0.5 * (b - a) * xs + 0.5 * (a + b)
        expect.append(np.sum(SC.density(kind, v, x) * ws) * 0.5 * (b - a))
    expect = np.array(expect) * mu.size
    keep = expect > 5
    from scipy import stats
    chi2 = np.sum((counts[keep] - expect[keep]) ** 2 / expect[keep])
    assert stats.chi2.sf(chi2, keep.sum() - 1) > 1e-4


@pytest.mark.parametrize("name", list(LAWS))
def test_pdf_integrates_to_one_and_matches_density(name):
    fmt, data, kw = LAWS[name]
    kind, v = SC.law(fmt, np.asarray(data, float), **kw)
    ang = SC.pack([(kind, v)])
    edges = np.linspace(-1, 1, 65)
    xs, ws = np.polynomial.legendre.leggauss(16)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        x = 0.5 * (b - a) * xs + 0.5 * (a + b)
        f = np.array([SC.pdf(ang, 0, t) for t in x])
        assert np.allclose(f, SC.density(kind, v, x), rtol=1e-12,
                           atol=1e-14)
        total += np.sum(f * ws) * 0.5 * (b - a)
    assert total == pytest.approx(1.0, abs=2e-3 if "rejection" in name
                                  else 1e-9)


def test_isotropic_laws_are_iso():
    assert SC.law("legendre", [2.0])[0] == SC.ISO
    assert SC.law("legendre", [2.0, 0.0, 0.0])[0] == SC.ISO
    assert SC.law("histogram", [3.0, 3.0, 3.0])[0] == SC.ISO
    assert SC.law("tabular", [1.0, 1.0])[0] == SC.ISO
    assert SC.law("legendre", [1.0, 0.5], max_order=0)[0] == SC.ISO


def test_tabular_conversion_clips_negative_density_as_openmc():
    """OpenMC sets the negative part of the Legendre density to 0 before
    renormalising; the mean cosine moves accordingly."""
    kind, v = SC.law("legendre", [1.0, 0.9, 0.5, -0.4])
    assert kind == SC.TAB and v[0].min() == 0.0
    assert SC.mean_cosine(kind, v) != pytest.approx(0.9, abs=1e-3)


def test_histogram_with_negative_bin_is_refused():
    with pytest.raises(ValueError):
        SC.law("histogram", [1.0, -0.1, 2.0])


def _one_sphere(law, tag, **settings):
    with common.scattering_law(law, tag):
        model = PROBLEMS["sphere"]()
    for k, val in settings.items():
        setattr(model.settings, k, val)
    return model


def test_import_reads_law_and_settings():
    pr = openmc_import.load(_one_sphere([1.0, 0.6, 0.3, 0.1], "p3"))
    assert pr.mats.laws[0][0] == SC.TAB and pr.mats.laws[0][1][0].size == 33
    pr = openmc_import.load(_one_sphere(
        [1.0, 0.6, 0.3, 0.1], "p3",
        tabular_legendre={"enable": True, "num_points": 9}))
    assert pr.mats.laws[0][1][0].size == 9
    pr = openmc_import.load(_one_sphere(
        [1.0, 0.6, 0.3, 0.1], "p3", tabular_legendre={"enable": False}))
    assert pr.mats.laws[0][0] == SC.LEG
    pr = openmc_import.load(_one_sphere([1.0, 0.6, 0.3, 0.1], "p3",
                                        max_order=0))
    assert pr.mats.laws[0][0] == SC.ISO


def test_angle_representation_is_refused():
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    lib = openmc.MGXSLibrary(groups)
    d = openmc.XSdata("m", groups, representation="angle")
    d.num_azimuthal, d.num_polar = 2, 2
    d.order = 0
    d.set_total(np.full((2, 2, 1), 1.0))
    d.set_absorption(np.full((2, 2, 1), 0.1))
    d.set_scatter_matrix(np.full((2, 2, 1, 1, 1), 0.9))
    lib.add_xsdata(d)
    path = common.LIBRARY_DIR / "angle_rep.h5"
    lib.export_to_hdf5(str(path))
    mat = openmc.Material()
    mat.set_density("macro", 1.0)
    mat.add_macroscopic("m")
    mats = openmc.Materials([mat])
    mats.cross_sections = str(path)
    s = openmc.Sphere(r=5, boundary_type="vacuum")
    model = openmc.Model(openmc.Geometry([openmc.Cell(fill=mat,
                                                      region=-s)]), mats,
                         common.settings(common.point_source((0, 0, 0))))
    with pytest.raises(Unsupported):
        openmc_import.load(model)


class _IsoOnly(Oracle):
    """A backend without anisotropic walks (as every trained model until
    interface v2; since v2.1 the Oracle itself walks them)."""
    aniso = False


def test_balls_only_in_isotropic_materials():
    """Cask with only the concrete anisotropic, and a backend that walks
    isotropic balls only: balls still run (in fuel and steel), none start
    in concrete."""
    with common.scattering_law({"concrete": [1.0, 0.6, 0.3, 0.1]}, "mix"):
        model = PROBLEMS["cask"]()
    pr = openmc_import.load(model)
    kinds = [k for k, _ in pr.mats.laws]
    assert sorted(set(kinds)) == [SC.ISO, SC.TAB]
    r = transport.run(pr, 2000, 2, seed=1, backend=_IsoOnly(), r_star=1.0)
    assert r.counters["balls"] > 0
    with common.scattering_law([1.0, 0.6, 0.3, 0.1], "p3"):
        model = PROBLEMS["cask"]()
    r = transport.run(openmc_import.load(model), 2000, 2, seed=1,
                      backend=_IsoOnly(), r_star=1.0)
    assert r.counters["balls"] == 0


def test_isotropic_problem_unchanged_by_the_law_machinery():
    """A law of [1] (P0) is the same as no law: bit-identical tallies."""
    a = transport.run(openmc_import.load(PROBLEMS["sphere"]()), 2000, 2,
                      seed=4)
    b = transport.run(openmc_import.load(_one_sphere([1.0], "p0")), 2000,
                      2, seed=4)
    for name in a.tallies:
        assert np.array_equal(a.tallies[name]["mean"],
                              b.tallies[name]["mean"])


def _sph_scatter_model(law, tag):
    model = _one_sphere(law, tag)
    f = openmc.SphericalHarmonicsFilter(2)
    f.cosine = "scatter"
    t = openmc.Tally(name="sphs")
    t.filters, t.scores = [f], ["scatter"]
    s = openmc.Tally(name="sc")
    s.scores = ["scatter"]
    model.tallies = openmc.Tallies([t, s])
    return model


def test_scatter_harmonics_keep_balls_when_all_isotropic():
    """As in v1.6: every law isotropic -> only the (0, 0) term, equal to
    the scatter score, and balls stay on."""
    pr = openmc_import.load(_sph_scatter_model([1.0], "p0"))
    assert not any("cannot be scored inside a ball" in w
                   for w in pr.warnings)
    r = transport.run(pr, 2000, 2, seed=1, backend=Oracle(), r_star=1.0)
    assert r.counters["balls"] > 0
    m = r.tallies["sphs"]["mean"]
    assert m[0, 2] == pytest.approx(r.tallies["sc"]["mean"][0, 2],
                                    rel=1e-12)
    assert np.all(m[1:, 2] == 0.0)


def test_scatter_harmonics_with_a_law_run_plain():
    pr = openmc_import.load(_sph_scatter_model([1.0, 0.6, 0.3], "p2"))
    assert any("cannot be scored inside a ball" in w for w in pr.warnings)
