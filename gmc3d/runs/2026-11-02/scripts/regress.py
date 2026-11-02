"""v1.2 vs HEAD, same container, same session: where did plain MC lose 5-16%?

Run from inside a worktree's gmc3d/.  Splits each problem's plain-MC time
into the part that happens with no tallies at all (geometry + flights) and
the part the tallies add, then times the geometry lookup on its own.
Version-agnostic: it never calls _descend directly, because its signature
changed in v1.3.
"""
import json, sys, time
sys.path.insert(0, ".")
import numpy as np
from numba import njit
from core import geometry as G, openmc_import, transport as TR
from problems import PROBLEMS

TAG = sys.argv[1]
PROBS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sphere", "cask", "slab"]
N, B, TRIES = 100_000, 5, 3


@njit(cache=True)
def bench_locate(g, xs, ys, zs, reps):
    acc = 0
    for _ in range(reps):
        for i in range(xs.size):
            c, m = G.locate(g, xs[i], ys[i], zs[i])
            acc += c + m
    return acc


def best_run(pr, **kw):
    TR.run(pr, 1000, 1, seed=1, **kw)                 # compile
    out, res = 1e30, None
    for k in range(TRIES):
        t = time.perf_counter()
        res = TR.run(pr, N, B, seed=9 + k, **kw)
        out = min(out, time.perf_counter() - t)
    return out, res


rows = {}
for name in PROBS:
    pr = openmc_import.load(PROBLEMS[name]())
    tallies = list(pr.tallies)
    with_t, res = best_run(pr)
    pr.tallies = []
    no_t, _ = best_run(pr)
    pr.tallies = tallies

    # the geometry lookup on its own, on points inside the geometry.  The
    # source object changed shape in v1.2, so points are found by search
    # instead: random points in a wide box, kept where a material is there.
    gen = np.random.default_rng(0)
    keep = [[], [], []]
    for half in (2.0, 10.0, 50.0, 200.0):
        q = gen.uniform(-half, half, size=(3, 20000))
        for j in range(q.shape[1]):
            c, m = G.locate(pr.geom, q[0, j], q[1, j], q[2, j])
            if m >= 0:
                keep[0].append(q[0, j]); keep[1].append(q[1, j])
                keep[2].append(q[2, j])
        if len(keep[0]) >= 5000:
            break
    xs = np.array(keep[0][:5000]); ys = np.array(keep[1][:5000])
    zs = np.array(keep[2][:5000])
    bench_locate(pr.geom, xs, ys, zs, 1)
    src_ns = 1e30
    for _ in range(5):
        t = time.perf_counter()
        bench_locate(pr.geom, xs, ys, zs, 40)
        src_ns = min(src_ns, 1e9 * (time.perf_counter() - t) / (40 * xs.size))

    fl = res.counters["flights"]
    rows[name] = {"with_tallies_s": with_t, "no_tallies_s": no_t,
                  "tally_cost_s": with_t - no_t, "locate_ns": src_ns, "locate_points": int(xs.size),
                  "flights": fl, "leakage": res.leakage,
                  "ns_per_flight_no_tallies": 1e9 * no_t / max(fl, 1)}
    print(f"{TAG:6s} {name:9s} with tallies {with_t:7.2f} s   "
          f"no tallies {no_t:7.2f} s   tallies cost {with_t - no_t:6.2f} s   "
          f"locate {src_ns:6.0f} ns   flights {fl:,}   "
          f"leakage {res.leakage:.6f}", flush=True)

json.dump(rows, open(f"/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad/regress_{TAG}.json", "w"), indent=1)
