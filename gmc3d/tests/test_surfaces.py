"""Each surface kernel against brute force (SRS S1-GEO-2, S1-DF-3).

For every surface kind, at random points:
  * ray: the point reached is on the surface, and nothing closer is;
  * nearest: no ray in any direction reaches the surface sooner (it is a
    lower bound), and the best of many rays nearly reaches it (it is tight).
"""
import numpy as np
import pytest

from core import surfaces as S

CASES = {
    S.X_PLANE: [1.5], S.Y_PLANE: [-2.0], S.Z_PLANE: [0.7],
    S.PLANE: [1.0, -2.0, 0.5, 1.2],
    S.SPHERE: [0.5, -1.0, 0.3, 2.5],
    S.X_CYLINDER: [0.4, -0.3, 1.7], S.Y_CYLINDER: [1.0, 0.2, 2.2],
    S.Z_CYLINDER: [-0.5, 0.5, 1.4],
    S.X_CONE: [0.2, 0.1, -0.3, 0.6], S.Y_CONE: [0.0, 0.5, 0.0, 1.3],
    S.Z_CONE: [-0.4, 0.3, 0.2, 0.25],
}


def _dirs(rng, n):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@pytest.mark.parametrize("kind", sorted(CASES))
def test_ray_lands_on_surface(kind):
    c = np.zeros(4)
    c[:len(CASES[kind])] = CASES[kind]
    rng = np.random.default_rng(kind)
    for p, d in zip(rng.uniform(-4, 4, (2000, 3)), _dirs(rng, 2000)):
        t = S.ray(kind, c, *p, *d)
        if t >= S.INF:
            continue
        q = p + t * d
        # on the surface: |f| small relative to its gradient
        gx, gy, gz = S.normal(kind, c, *q)
        g = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
        assert abs(S.value(kind, c, *q)) / g < 1e-8
        # nothing earlier: f keeps its sign along the way
        f0 = S.value(kind, c, *p)
        for s in np.linspace(0.0, t, 50)[1:-1]:
            f = S.value(kind, c, *(p + s * d))
            assert np.sign(f) == np.sign(f0) or abs(f) < 1e-9


@pytest.mark.parametrize("kind", sorted(CASES))
def test_nearest_is_tight_lower_bound(kind):
    c = np.zeros(4)
    c[:len(CASES[kind])] = CASES[kind]
    rng = np.random.default_rng(100 + kind)
    dirs = _dirs(rng, 20000)
    for p in rng.uniform(-4, 4, (40, 3)):
        d = S.nearest(kind, c, *p)
        rays = np.array([S.ray(kind, c, *p, *u) for u in dirs])
        best = rays.min()
        assert d <= best * (1 + 1e-9) + 1e-12          # never overestimates
        assert best <= d + 0.05 * max(d, 0.1) + 0.02   # and is nearly exact
