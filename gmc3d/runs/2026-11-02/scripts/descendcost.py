"""Nanoseconds per `_descend`, the geometry walk every flight makes.

Version-agnostic: `_descend` gained a `path` buffer at v1.3, so the driver
loop is picked from the signature.  Same points and directions in every
version (fixed seed), both questions asked (want_near, want_ray), so the
number is comparable across versions.  Nothing in the repo is changed.
"""
import inspect, sys, time
sys.path.insert(0, ".")
import numpy as np
from numba import njit
from core import geometry as G, openmc_import
from problems import PROBLEMS

TAG = sys.argv[1]
NARG = len(inspect.signature(G._descend.py_func).parameters)
HAS_PATH = NARG >= 10
_d = G._descend
ML = G.MAX_LEVELS


@njit(cache=True)
def bench_old(g, xs, ys, zs, us, vs, ws, reps):
    acc = 0.0
    for _ in range(reps):
        for i in range(xs.size):
            c, m, near, t, s, at = _d(g, xs[i], ys[i], zs[i],
                                      us[i], vs[i], ws[i], True, True)
            acc += near + t + c
    return acc


@njit(cache=True)
def bench_new(g, xs, ys, zs, us, vs, ws, reps, path):
    acc = 0.0
    for _ in range(reps):
        for i in range(xs.size):
            out = _d(g, xs[i], ys[i], zs[i],
                     us[i], vs[i], ws[i], True, True, path)
            acc += out[2] + out[3] + out[0]
    return acc


for name in sys.argv[2].split(","):
    pr = openmc_import.load(PROBLEMS[name]())
    gen = np.random.default_rng(7)
    xs, ys, zs = [], [], []
    for half in (2.0, 10.0, 50.0, 200.0):
        q = gen.uniform(-half, half, size=(3, 40000))
        for j in range(q.shape[1]):
            if G.locate(pr.geom, q[0, j], q[1, j], q[2, j])[1] >= 0:
                xs.append(q[0, j]); ys.append(q[1, j]); zs.append(q[2, j])
        if len(xs) >= 5000:
            break
    n = min(5000, len(xs))
    xs = np.array(xs[:n]); ys = np.array(ys[:n]); zs = np.array(zs[:n])
    d = gen.normal(size=(3, n)); d /= np.linalg.norm(d, axis=0)
    us, vs, ws = d[0].copy(), d[1].copy(), d[2].copy()
    path = np.empty(ML, np.int64)

    def one(reps):
        if HAS_PATH:
            return bench_new(pr.geom, xs, ys, zs, us, vs, ws, reps, path)
        return bench_old(pr.geom, xs, ys, zs, us, vs, ws, reps)

    one(1)                                    # compile
    best = 1e30
    for _ in range(7):
        t = time.perf_counter()
        one(40)
        best = min(best, 1e9 * (time.perf_counter() - t) / (40 * n))
    print(f"DESC {TAG} {name:7s} {best:7.0f} ns per _descend "
          f"({n} points, want_near+want_ray)", flush=True)
