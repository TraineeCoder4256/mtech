"""v2.1.3: walk.first_scatter_aniso is walk_aniso up to its second
collision, walk for walk on the same streams."""
import numpy as np
import pytest

from ball import laws as LW, walk
from core import rng, scattering as SC


@pytest.mark.parametrize("name", list(LW.held_out()))
def test_same_walks_as_walk_aniso(name):
    ang = SC.pack([LW.held_out()[name]])
    n = 20_000
    R = np.exp(np.random.default_rng(5).uniform(0, np.log(30), n))
    mats = np.zeros(n, np.int64)
    cn, mu, phi, s, k = walk.walk_many_aniso(
        R, mats, ang, rng.streams(np.uint64(7), 0, n), np.arange(n))
    ex, c1, m1, p1, s1 = walk.first_scatter_many_aniso(
        R, mats, ang, rng.streams(np.uint64(7), 0, n), np.arange(n))
    assert np.array_equal(ex, k == 1) and ex.sum() > 500
    for a, b in ((cn, c1), (mu, m1), (phi, p1), (s, s1)):
        assert np.array_equal(a[ex], b[ex])
    assert np.all(np.isnan(s1[~ex]))
    assert np.all(np.abs(p1[ex] - np.pi) < 1e-9)       # in-plane, far side
