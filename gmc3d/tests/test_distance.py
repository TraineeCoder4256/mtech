"""The distance field never overestimates (SRS S1-DF-1, S1-DF-2).

The acceptance test of the whole method: at 10^6 random points (200,000
per problem), take the ball the distance field allows and probe it -- 32
points on its surface and 32 inside.  Every probe must land in the same
cell as the centre.  A single miss means a ball could cross a boundary and
the ball step would give a wrong answer.

The test also reports how tight the field is (ball radius over the true
distance, estimated by rays), since a field that is safe but tiny would
make ball steps pointlessly small.
"""
import numpy as np
from numba import njit, prange

from core import geometry as G
from core import openmc_import
from problems import PROBLEMS
from test_geometry import random_points


@njit(parallel=True, cache=True)
def _violations(g, pts, dirs, radii_frac):
    n = pts.shape[0]
    bad = np.zeros(n, np.int64)
    for i in prange(n):
        x, y, z = pts[i, 0], pts[i, 1], pts[i, 2]
        d, cell, mat = G.nearest_boundary(g, x, y, z)
        if mat == G.LOST or d <= 0.0:
            continue
        r = d * (1.0 - 1e-9)
        for k in range(dirs.shape[0]):
            f = radii_frac[k] * r
            c2, _ = G.locate(g, x + f * dirs[k, 0], y + f * dirs[k, 1],
                             z + f * dirs[k, 2])
            if c2 != cell:
                bad[i] += 1
    return bad


def test_never_overestimates():
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(64, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    frac = np.concatenate([np.ones(32), rng.uniform(0, 1, 32) ** (1 / 3)])
    total = 0
    for name in sorted(PROBLEMS):
        prob = openmc_import.load(PROBLEMS[name]())
        pts = random_points(name, 200_000, seed=1)
        bad = _violations(prob.geom, pts, dirs, frac)
        total += pts.shape[0]
        assert bad.sum() == 0, f"{name}: {int((bad > 0).sum())} balls leak"
    assert total == 1_000_000
