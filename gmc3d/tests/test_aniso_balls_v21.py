"""v2.1: balls in anisotropic materials, walked exactly (ball/walk.py
walk_aniso, the Oracle's sample_aniso).

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import openmc
import pytest
from scipy import stats

from ball import walk
from ball.oracle import Oracle
from checks import compare, validate
from core import openmc_import, scattering as SC, transport
from problems import PROBLEMS, common

P3 = [1.0, 0.6, 0.3, 0.1]


def _flat_tab():
    """A flat law packed as TAB (law() would call it ISO), so that the
    anisotropic walk runs with isotropic scattering."""
    return SC.pack([(SC.TAB, (np.full(3, 0.5), np.array([0.0, 0.5, 1.0])))])


@pytest.mark.parametrize("R", [1.0, 6.0])
def test_isotropic_law_gives_v1_exit(R):
    """With isotropic scattering the anisotropic walk's (mu, s, k) are v1's
    (KS p > 1e-3), and the exit point leans towards the first direction
    (its first flight runs along it)."""
    cn, mu, phi, s, k = walk.sample_aniso(R, _flat_tab(), 0, 100_000, seed=3)
    mu0, s0, k0 = walk.sample(R, 100_000, seed=4)
    assert stats.ks_2samp(mu, mu0).pvalue > 1e-3
    assert stats.ks_2samp(s, s0).pvalue > 1e-3
    assert stats.ks_2samp(k, k0).pvalue > 1e-3
    assert cn.mean() > 0.05
    assert np.all((cn >= -1) & (cn <= 1) & (mu >= 0) & (mu <= 1)
                  & (phi >= 0) & (phi <= np.pi) & (s > R))


def test_forward_law_moves_the_exit_forward():
    ang = SC.pack([SC.law("legendre", np.array(P3))])
    cn, mu, phi, s, k = walk.sample_aniso(5.0, ang, 0, 50_000, seed=3)
    cn0, *_ = walk.sample_aniso(5.0, _flat_tab(), 0, 50_000, seed=3)
    assert cn.mean() > cn0.mean() + 0.1
    assert k.mean() < walk.sample(5.0, 50_000, seed=4)[2].mean()


def test_exit_frame_is_orthonormal():
    rs = np.random.default_rng(1)
    for _ in range(200):
        n = rs.normal(size=3)
        n /= np.linalg.norm(n)
        a = rs.normal(size=3)
        a /= np.linalg.norm(a)
        if _ == 0:
            a = n.copy()                         # degenerate: a along n
        e = walk.exit_frame(*n, *a)
        e1, e2 = np.array(e[:3]), np.array(e[3:])
        for x, y in ((e1, e2), (e1, n), (e2, n)):
            assert abs(x @ y) < 1e-12
        assert np.linalg.norm(e1) == pytest.approx(1.0)
        assert np.allclose(np.cross(n, e1), e2)
        if _:
            assert e1 @ a >= 0.0                 # on a's side


def _model(law, tag, meshes=False):
    with common.scattering_law(law, tag):
        model = PROBLEMS["sphere"]()
    if not meshes:
        model.tallies = openmc.Tallies([
            t for t in model.tallies
            if not any(isinstance(f, openmc.MeshFilter) for f in t.filters)])
    return model


def test_anisotropic_balls_agree_with_plain_mc():
    pr = openmc_import.load(_model(P3, "p3"))
    mc = transport.run(pr, 20_000, 10, seed=5)
    orc = Oracle()
    ob = transport.run(pr, 20_000, 10, seed=6, backend=orc, r_star=1.0)
    assert ob.counters["balls"] > 10_000 and orc.scatters > 0
    ok, bad = validate.verdict(compare.compare(ob, mc))
    assert ok, bad


def test_mesh_holds_anisotropic_balls_in_one_voxel():
    """With a mesh the balls shrink to one voxel; the sphere's voxels are
    under 1 mfp, so at R* = 1 there are none."""
    pr = openmc_import.load(_model(P3, "p3", meshes=True))
    ob = transport.run(pr, 2_000, 2, seed=6, backend=Oracle(), r_star=1.0)
    assert ob.counters["balls"] == 0
    ob = transport.run(pr, 2_000, 2, seed=6, backend=Oracle(), r_star=0.2,
                       mesh_rule="cap")
    assert ob.counters["balls"] > 0


def test_no_anisotropic_balls_with_detectors_or_without_aniso_backend():
    model = _model(P3, "p3")
    common.point_detector(model, "d", (5.0, 0.0, 0.0), r0=0.3)
    pr = openmc_import.load(model)
    ob = transport.run(pr, 2_000, 2, seed=6, backend=Oracle(), r_star=1.0)
    assert ob.counters["balls"] == 0

    class IsoOnly(Oracle):
        aniso = False
    pr = openmc_import.load(_model(P3, "p3"))
    ob = transport.run(pr, 2_000, 2, seed=6, backend=IsoOnly(), r_star=1.0)
    assert ob.counters["balls"] == 0
