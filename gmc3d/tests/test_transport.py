"""The solver: random streams, exact answers, ball steps, reproducibility.

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import pytest

from ball import walk
from ball.oracle import Oracle
from ball.table import Table
from core import openmc_import, rng, transport
from problems import PROBLEMS


def test_skip_ahead_matches_stepping():
    st = np.array([12345], np.uint64)
    for _ in range(1000):
        rng.rand(st, 0)
    assert st[0] == rng.skip_ahead(1000, 12345)


def test_pure_absorber_is_exact():
    """sigma_s = 0: every particle flies straight out; zero variance."""
    pr = openmc_import.load(PROBLEMS["sphere"](radius=10, sig_s=0.0,
                                               sig_a=0.1))
    r = transport.run(pr, 2000, 3)
    assert abs(r.leakage - np.exp(-1.0)) < 1e-12
    assert abs(r.tallies["total"]["mean"][0, 0] - (1 - np.exp(-1)) / 0.1) \
        < 1e-10


@pytest.mark.parametrize("use_balls", [False, True])
def test_infinite_medium(use_balls):
    """Reflective cube: flux 1/sigma_a, absorption 1, no leakage."""
    pr = openmc_import.load(PROBLEMS["sphere"](radius=5, sig_s=1.0,
                                               sig_a=0.1, box=True))
    r = transport.run(pr, 20000, 10, backend=Oracle() if use_balls else None,
                      r_star=1.0, mesh_rule="centre")
    t = r.tallies["total"]
    assert r.leakage == 0.0
    assert abs(t["mean"][0, 0] - 10.0) < 4 * t["se"][0, 0]
    assert r.counters["lost"] == 0
    # the mesh covers the whole cube, so its voxels add up to the total
    assert abs(r.tallies["mesh"]["mean"][:, 0].sum() - t["mean"][0, 0]) \
        < 1e-9 * t["mean"][0, 0]
    if use_balls:
        assert r.counters["balls"] > 0


def test_balls_match_plain_mc_on_slab():
    pr = openmc_import.load(PROBLEMS["slab"]())
    a = transport.run(pr, 20000, 10, seed=2)
    b = transport.run(pr, 20000, 10, seed=3, backend=Oracle(), r_star=1.0,
                      mesh_rule="centre")
    assert b.counters["balls"] > 0
    z = (a.leakage - b.leakage) / np.hypot(a.leakage_se, b.leakage_se)
    assert abs(z) < 4


def test_reproducible_and_thread_independent():
    """Same seed -> same answer; the chunking changes only the order of
    the final sums (rounding), never the random numbers."""
    pr = openmc_import.load(PROBLEMS["curved"]())
    a = transport.run(pr, 5000, 2, seed=4, backend=Oracle(), r_star=2.0)
    b = transport.run(pr, 5000, 2, seed=4, backend=Oracle(), r_star=2.0)
    c = transport.run(pr, 5000, 2, seed=4, backend=Oracle(), r_star=2.0,
                      nchunk=7)
    assert a.leakage == b.leakage
    assert a.counters == b.counters
    assert abs(a.leakage - c.leakage) < 1e-12
    assert a.counters["flights"] == c.counters["flights"]


def test_walk_outputs_are_physical():
    mu, s, k = walk.sample(5.0, 5000, seed=2)
    assert np.all((mu > 0) & (mu <= 1))
    assert np.all(s > 5.0)
    assert np.all(k >= 1)


def test_table_mean_path_close_to_exact():
    t = Table.build(n_grid=16, n_per=4096)
    R = 7.3                                     # between grid points
    st = rng.streams(np.uint64(3), 0, 50000)
    _, s1 = t.sample(np.full(50000, R), st, np.arange(50000))
    _, s0, _ = walk.sample(R, 50000, seed=8)
    se = np.hypot(s0.std() / np.sqrt(s0.size), s1.std() / np.sqrt(s1.size))
    assert abs(s1.mean() - s0.mean()) < 5 * se
