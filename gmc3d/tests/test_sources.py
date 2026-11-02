"""v1.2: every OpenMC source kind, sampled the way OpenMC samples it.

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import openmc
import openmc.stats as st
import pytest
from scipy import stats

from core import geometry as G
from core import openmc_import, transport
from core.openmc_import import IgnoredSetting, Unsupported
from problems import PROBLEMS

N = 200_000
P_MIN = 1e-3          # a fair test fails one time in a thousand


def _load(src, problem="cask"):
    model = PROBLEMS[problem]()
    model.settings.source = src
    return openmc_import.load(model), model


def _draw(src, n=N, problem="cask", seed=3):
    pr, model = _load(src, problem)
    return pr.source.sample(n, np.random.default_rng(seed)), pr, model


# ----------------------------------------------------------------- spaces
def test_cartesian_matches_each_distributions_own_samples():
    x, y, z = st.Normal(0.0, 5.0), st.PowerLaw(1.0, 14.0, 2), \
        st.Tabular([-50.0, 0.0, 50.0], [1.0, 3.0, 1.0],
                   interpolation="linear-linear")
    s, _, _ = _draw(openmc.IndependentSource(
        space=st.CartesianIndependent(x, y, z)))
    for k, d in enumerate((x, y, z)):
        ref, _ = d.sample(N, seed=99)
        assert stats.ks_2samp(s.xyz[:, k], ref).pvalue > P_MIN


def test_spherical_power_law_two_is_uniform_in_a_ball():
    s, _, _ = _draw(openmc.IndependentSource(space=st.SphericalIndependent(
        st.PowerLaw(0.0, 15.0, 2), st.Uniform(-1.0, 1.0),
        st.Uniform(0.0, 2 * np.pi), origin=(1.0, 2.0, 3.0))))
    d = s.xyz - (1.0, 2.0, 3.0)
    r = np.linalg.norm(d, axis=1)
    assert stats.kstest((r / 15.0) ** 3, "uniform").pvalue > P_MIN
    assert np.all(np.abs(d.mean(axis=0)) < 5 * 15.0 / np.sqrt(N))


def test_cylindrical_power_law_one_is_uniform_in_a_disc():
    s, _, _ = _draw(openmc.IndependentSource(space=st.CylindricalIndependent(
        st.PowerLaw(0.0, 18.0, 1), st.Uniform(0.0, 2 * np.pi),
        st.Uniform(-60.0, 60.0))))
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    assert stats.kstest((r / 18.0) ** 2, "uniform").pvalue > P_MIN


def test_pointcloud_bias_gives_weights_that_restore_the_strengths():
    pts = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0)]
    s, _, _ = _draw(openmc.IndependentSource(space=st.PointCloud(
        pts, strengths=[1.0, 2.0, 7.0], bias=[1.0, 1.0, 1.0])))
    which = np.round(s.xyz[:, 0] / 5.0) + 2 * np.round(s.xyz[:, 1] / 5.0)
    share = np.array([s.wgt[which == k].sum() for k in range(3)]) / N
    assert np.allclose(share, [0.1, 0.2, 0.7], atol=0.01)
    assert np.allclose(np.bincount(which.astype(int)) / N, 1 / 3, atol=0.01)


def test_meshspatial_volume_normalised_on_a_cylindrical_mesh():
    mesh = openmc.CylindricalMesh(r_grid=[0.0, 6.0, 12.0, 18.0],
                                  phi_grid=[0.0, 2 * np.pi],
                                  z_grid=[-60.0, 60.0])
    s, _, _ = _draw(openmc.IndependentSource(space=st.MeshSpatial(
        mesh, [1.0, 2.0, 4.0], volume_normalized=True)))
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    got = np.histogram(r, [0, 6, 12, 18])[0] / N
    vol = np.array([36.0, 144.0 - 36.0, 324.0 - 144.0])
    want = vol * [1.0, 2.0, 4.0] / np.sum(vol * [1.0, 2.0, 4.0])
    assert np.allclose(got, want, atol=0.005)
    # uniform in volume inside an element: r^2 uniform in the outer ring
    o = r[r > 12.0]
    assert stats.kstest((o ** 2 - 144.0) / 180.0, "uniform").pvalue > P_MIN


# ----------------------------------------------------------------- angles
def test_polar_azimuthal_follows_the_reference_frame():
    """mu = 0.5 about u = +x, phi = 0 about v = +y: every direction is
    0.5 x + sqrt(0.75) y (OpenMC's frame: mu u + sin(theta)(cos(phi) v +
    sin(phi) u x v))."""
    s, _, _ = _draw(openmc.IndependentSource(angle=st.PolarAzimuthal(
        st.Discrete([0.5], [1.0]), st.Discrete([0.0], [1.0]),
        reference_uvw=(1.0, 0.0, 0.0), reference_vwu=(0.0, 1.0, 0.0))),
        n=100)
    assert np.allclose(s.uvw, [0.5, np.sqrt(0.75), 0.0])
    s, _, _ = _draw(openmc.IndependentSource(angle=st.PolarAzimuthal(
        st.Discrete([0.0], [1.0]), st.Discrete([0.5 * np.pi], [1.0]),
        reference_uvw=(1.0, 0.0, 0.0), reference_vwu=(0.0, 1.0, 0.0))),
        n=100)
    assert np.allclose(s.uvw, [0.0, 0.0, 1.0])       # u x v = +z
    assert not s.iso.any()


def test_isotropic_directions_are_isotropic_and_marked():
    s, _, _ = _draw(openmc.IndependentSource())
    assert s.iso.all()
    assert stats.kstest((s.uvw[:, 2] + 1) / 2, "uniform").pvalue > P_MIN
    assert np.allclose(np.linalg.norm(s.uvw, axis=1), 1.0)


# -------------------------------------------- mesh, file, several sources
def test_mesh_source_picks_elements_by_strength_with_their_angles():
    mesh = openmc.RegularMesh()
    mesh.lower_left, mesh.upper_right, mesh.dimension = \
        (-10.0, -10.0, -10.0), (10.0, 10.0, 10.0), (2, 1, 1)
    parts = [openmc.IndependentSource(strength=1.0),
             openmc.IndependentSource(
                 angle=st.Monodirectional((0.0, 0.0, 1.0)), strength=3.0)]
    s, _, _ = _draw(openmc.MeshSource(mesh, parts))
    right = s.xyz[:, 0] > 0
    assert abs(right.mean() - 0.75) < 0.01
    assert np.allclose(s.uvw[right], [0.0, 0.0, 1.0])
    assert s.iso[~right].all() and not s.iso[right].any()
    assert np.allclose(s.wgt, 4.0)                      # total strength


def test_strengths_choose_the_source_and_set_the_weight():
    src = [openmc.IndependentSource(space=st.Point((1.0, 0.0, 0.0)),
                                    strength=1.0),
           openmc.IndependentSource(space=st.Point((2.0, 0.0, 0.0)),
                                    strength=3.0)]
    s, _, _ = _draw(src)
    assert abs((s.xyz[:, 0] == 2.0).mean() - 0.75) < 0.01
    assert np.allclose(s.wgt, 4.0)


def _file_source(tmp_path, wgt):
    rng = np.random.default_rng(5)
    sites = [openmc.SourceParticle(r=(x, 0.0, 0.0), u=(1.0, 0.0, 0.0),
                                   wgt=wgt) for x in rng.uniform(0, 30, 500)]
    path = tmp_path / f"w{wgt}.h5"
    openmc.write_source_file(sites, str(path))
    return openmc.FileSource(str(path))


def test_file_source_weights_scale_tallies_not_leakage(tmp_path):
    """OpenMC (measured): tallies per source particle, leakage per unit of
    source weight.  Same sites, weights 1 and 2, same random numbers."""
    out = []
    for w in (1.0, 2.0):
        pr, _ = _load(_file_source(tmp_path, w), "slab")
        out.append(transport.run(pr, 5000, 2, seed=3))
    a, b = out
    assert b.leakage == pytest.approx(a.leakage, rel=1e-12)
    assert b.tallies["total"]["mean"][0, 0] == \
        pytest.approx(2 * a.tallies["total"]["mean"][0, 0], rel=1e-12)


# ------------------------------------------------------------ constraints
@pytest.mark.parametrize("kind", ["universe", "cell", "material"])
def test_domain_constraints_hold_at_any_level(kind):
    model = PROBLEMS["nested"]()
    univs = {u.name: u for u in model.geometry.get_all_universes().values()}
    cells = {c.name: c for c in model.geometry.get_all_cells().values()}
    mats = {m.name: m for m in model.materials}
    dom = {"universe": univs["offset"], "cell": cells["tilted-pin"],
           "material": mats["pin"]}[kind]
    model.settings.source = openmc.IndependentSource(
        space=st.Box((-4.0, -4.0, -3.0), (4.0, 4.0, 3.0)),
        constraints={"domains": [dom]})
    pr = openmc_import.load(model)
    s = pr.source.sample(20_000, np.random.default_rng(1))
    assert s.redrawn > 0
    c, u, m = G.paths_many(pr.geom, s.xyz)
    ids_c = pr.cell_ids
    if kind == "material":
        assert np.all(np.array(pr.mats.ids)[m] == dom.id)
    elif kind == "cell":
        assert np.all((np.where(c >= 0, ids_c[c], -1) == dom.id).any(axis=1))
    else:
        uid = sorted(model.geometry.get_all_universes())
        assert np.all((np.where(u >= 0, np.array(uid)[u], -1)
                       == dom.id).any(axis=1))


def test_bounds_and_kill_are_warned_as_unused():
    with pytest.warns(IgnoredSetting) as rec:
        _load(openmc.IndependentSource(constraints={
            "energy_bounds": [0.0, 1e6], "time_bounds": [0.0, 1.0],
            "rejection_strategy": "kill"}))
    said = " ".join(str(w.message) for w in rec)
    for word in ("energy_bounds", "time_bounds", "kill"):
        assert word in said


# --------------------------------------------------------------- refusals
@pytest.mark.parametrize("src, words", [
    (lambda: openmc.CompiledSource("libsource.so"), "C\\+\\+"),
    (lambda: openmc.IndependentSource(particle="photon"), "photon"),
    (lambda: openmc.IndependentSource(constraints={"fissionable": True}),
     "fissionable"),
    (lambda: openmc.IndependentSource(space=st.Box(
        (0, 0, 0), (1, 1, 1), only_fissionable=True)), "fissionable"),
    (lambda: openmc.FileSource("sites.txt"), "sites.txt"),
])
def test_unsupported_sources_are_refused(src, words):
    with pytest.raises(Unsupported, match=words):
        _load(src())
