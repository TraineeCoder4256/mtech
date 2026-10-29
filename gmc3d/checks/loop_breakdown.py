"""Where does the transport loop's time go?

Used by: run by hand.  From gmc3d/:

    python -m checks.loop_breakdown --problem slab --r-star 2
    python -m checks.loop_breakdown --problem sphere --backend table

`checks/net_profile.py` took one network draw apart operation by operation,
but `core/transport.py` reports its own time as a single number,
`transport_s`, so the ceiling a free backend would hit has never been broken
down.  A free backend is what gmc3d is up against once the network is cheap
(see runs/2026-10-06/SPEED.md), so this is the thing worth knowing next.

TWO PARTS.

Part A re-runs the wave loop here, phase by phase, with the SAME njit
kernels imported from core/transport.py.  Nothing in core/ is touched or
changed; only the Python driver is duplicated, and the answer is checked
against `transport.run` before any timing is reported.  It shows what
`transport_s` and `backend_s` leave out: the per-batch particle allocation,
the WAITING gather and scatter, and the tally reduction, none of which is
inside either number today -- so a race's `seconds` understates the real
wall time.

Part B times the two kernels that dominate one flight, on positions
harvested from a real run (the scatter sites where particles asked for a
ball, which is exactly where the geometry gets its hardest call):

  * `geometry._descend` with want_near False and True.  The difference is
    the tax the ball step puts on every fresh flight, because a ball needs
    the distance to the NEAREST surface in any direction, not just the
    distance along the ray.
  * `tallies.score_segment`, the scoring done on every flight.

Timing njit code one call at a time would measure Python, so each is driven
from its own njit loop and the result is accumulated into a return value so
nothing is optimised away.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from ball import table as BT
from ball.oracle import Oracle
from core import geometry as G
from core import openmc_import, rng
from core import tallies as T
from core import transport as TR
from problems import PROBLEMS

HERE = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- Part B
@njit(cache=True)
def _bench_descend(g, xs, ys, zs, us, vs, ws, want_near, reps):
    acc = 0.0
    path = np.full(G.MAX_LEVELS, -1, np.int64)
    for _ in range(reps):
        for i in range(xs.size):
            cell, mat, near, t_ray, surf, at_root = G._descend(
                g, xs[i], ys[i], zs[i], us[i], vs[i], ws[i], want_near, True,
                path)
            acc += t_ray + near + cell + mat
    return acc


@njit(cache=True)
def _bench_score(row, L, cells, mats, xs, ys, zs, us, vs, ws, steps, reps):
    acc = 0.0
    path = np.full(G.MAX_LEVELS, -1, np.int64)
    scr = T.scratch(G.MAX_LEVELS)
    for _ in range(reps):
        for i in range(xs.size):
            path[0] = cells[i]          # one level: enough for the timing
            T.score_segment(row, L, cells[i], path, mats[i], 0.1, 0.9, 1.0,
                            xs[i], ys[i], zs[i], us[i], vs[i], ws[i],
                            steps[i], scr)
            acc += row[0]
    return acc


@njit(cache=True)
def _bench_rng(seeds, reps):
    acc = 0.0
    for _ in range(reps):
        for i in range(seeds.size):
            acc += rng.rand(seeds, i)
            u, v, w = rng.isotropic(seeds, i)
            acc += u + v + w
    return acc


def _best(fn, args, reps, n, tries=5):
    out = 1e30
    for _ in range(tries):
        t = time.perf_counter()
        fn(*args, reps)
        out = min(out, (time.perf_counter() - t) / (reps * n))
    return out


# --------------------------------------------------------------- Part A
def run_split(problem, particles, batches, seed, backend, r_star,
              mesh_rule="profile", nchunk=32):
    """core/transport.run, with a stopwatch on every phase.

    Kept deliberately close to the original so the two can be compared
    line by line; the phases are the only addition.
    """
    g, m = problem.geom, problem.mats
    ph, L, _ = TR.physics(problem, backend, r_star, None, mesh_rule)
    accum = T.Accumulator(L)
    cnt = np.zeros((nchunk, len(TR.COUNTERS)), np.int64)
    ph_t = dict(allocate=0.0, birth=0.0, advance=0.0, gather=0.0,
                backend=0.0, reduce=0.0)
    calls = 0
    sites = None

    wall0 = time.perf_counter()
    for b in range(batches):
        t = time.perf_counter()
        P = TR._particles(particles)
        P.seed[:] = rng.streams(np.uint64(seed), b * particles, particles)
        acc = np.zeros((nchunk, L.n_entries))
        ph_t["allocate"] += time.perf_counter() - t

        t = time.perf_counter()
        st = problem.source.sample(particles, TR.source_rng(seed, b))
        TR._birth(st.xyz, st.uvw, st.wgt, st.iso, P, cnt, nchunk)
        ph_t["birth"] += time.perf_counter() - t

        while True:
            t = time.perf_counter()
            TR._advance(g, ph, L, P, acc, cnt, nchunk)
            ph_t["advance"] += time.perf_counter() - t

            t = time.perf_counter()
            idx = np.flatnonzero(P.status == TR.WAITING)
            ph_t["gather"] += time.perf_counter() - t
            if idx.size == 0:
                break
            if sites is None and idx.size > 1000:      # harvest for Part B
                k = idx[:20000]
                sites = dict(x=P.x[k].copy(), y=P.y[k].copy(),
                             z=P.z[k].copy(), cell=P.cell[k].copy(),
                             mat=P.mat[k].copy(), R=P.ball_R[k].copy(),
                             d=P.ball_d[k].copy(), seed=P.seed[k].copy())

            t = time.perf_counter()
            mu, s_mfp = backend.sample(P.ball_R[idx], P.seed, idx)
            ph_t["backend"] += time.perf_counter() - t

            t = time.perf_counter()
            P.mu[idx], P.s[idx] = mu, s_mfp
            P.status[idx] = TR.READY
            ph_t["gather"] += time.perf_counter() - t
            calls += 1

        t = time.perf_counter()
        accum.add_batch(acc, particles)
        ph_t["reduce"] += time.perf_counter() - t
    wall = time.perf_counter() - wall0

    counters = dict(zip(TR.COUNTERS, cnt.sum(axis=0).tolist()))
    counters["backend_calls"] = calls
    timing = {"transport_s": ph_t["birth"] + ph_t["advance"],
              "backend_s": ph_t["backend"]}
    res = accum.results(problem.tallies, counters, timing,
                        dict(particles=particles, batches=batches))
    return res, ph_t, wall, sites


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--problem", default="slab")
    p.add_argument("--backend", default="table", choices=["table", "oracle"])
    p.add_argument("--r-star", type=float, default=2.0)
    p.add_argument("--particles", type=int, default=100_000)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    be = (BT.Table.load(HERE / "data" / "ball" / "table.npz")
          if a.backend == "table" else Oracle())
    pr = openmc_import.load(PROBLEMS[a.problem]())

    print(f"\n== {a.problem}, backend {a.backend}, R* = {a.r_star:g}, "
          f"{a.particles:,} x {a.batches} = "
          f"{a.particles * a.batches:,} histories")

    TR.run(pr, 1000, 1, seed=1, backend=be, r_star=a.r_star,
           mesh_rule="profile")                       # compile the kernels
    ref = TR.run(pr, a.particles, a.batches, seed=1, backend=be,
                 r_star=a.r_star, mesh_rule="profile")
    res, ph_t, wall, sites = run_split(pr, a.particles, a.batches, 1, be,
                                       a.r_star)

    ok = (abs(res.leakage - ref.leakage) < 1e-12
          and res.counters["flights"] == ref.counters["flights"])
    print(f"\n  answer check: ours {res.leakage:.9f}, transport.run "
          f"{ref.leakage:.9f}; flights {res.counters['flights']:,} vs "
          f"{ref.counters['flights']:,} -> "
          f"{'IDENTICAL' if ok else 'DIFFERENT, timings below are void'}")
    if not ok:
        raise SystemExit(1)

    print("\n-- Part A: where the wall time goes")
    named = sum(ph_t.values())
    for k in ("allocate", "birth", "advance", "gather", "backend", "reduce"):
        print(f"  {k:10s} {ph_t[k]:8.3f} s  {100 * ph_t[k] / wall:5.1f}%")
    print(f"  {'(other)':10s} {wall - named:8.3f} s  "
          f"{100 * (wall - named) / wall:5.1f}%")
    print(f"  {'WALL':10s} {wall:8.3f} s")
    # What transport.run would report for THIS execution: its t_track is
    # birth + advance and its t_backend covers the gather as well as the
    # backend call, so only `allocate` and `reduce` fall outside.  Taking
    # both from the same execution is the point: comparing the reported
    # seconds of one run against the wall clock of another mixes the
    # structural gap with this container's run-to-run timing noise.
    rep = ph_t["birth"] + ph_t["advance"] + ph_t["gather"] + ph_t["backend"]
    print(f"\n  what a race calls `seconds` is transport_s + backend_s, "
          f"which for this run is\n  {rep:.3f} s against a {wall:.3f} s "
          f"wall: {100 * (1 - rep / wall):.1f}% low, all of it the "
          f"per-batch\n  particle allocation. transport.run reported "
          f"{ref.timing['transport_s'] + ref.timing['backend_s']:.3f} s "
          f"for its own\n  execution of the same thing, so run-to-run "
          f"noise is larger than the gap.")

    print("\n-- Part B: the two kernels inside one flight")
    n = sites["x"].size
    gen = np.random.default_rng(0)
    d = gen.normal(size=(3, n))
    d /= np.linalg.norm(d, axis=0)
    us, vs, ws = d
    L = T.compile_layout(pr.tallies, len(pr.cell_ids), len(pr.mats.ids),
                         len(pr.geom.surf_kind),
                         leaf=pr.geom.cell_fill_type <= G.FILL_VOID)
    row = np.zeros(L.n_entries)
    steps = np.full(n, 0.5)
    args = (pr.geom, sites["x"], sites["y"], sites["z"], us, vs, ws)
    _bench_descend(*args, True, 1)
    _bench_score(row, L, sites["cell"], sites["mat"], sites["x"],
                 sites["y"], sites["z"], us, vs, ws, steps, 1)
    _bench_rng(sites["seed"], 1)

    far = _best(lambda *x: _bench_descend(*args, False, x[-1]), (), 20, n)
    near = _best(lambda *x: _bench_descend(*args, True, x[-1]), (), 20, n)
    sc = _best(lambda *x: _bench_score(row, L, sites["cell"], sites["mat"],
                                       sites["x"], sites["y"], sites["z"],
                                       us, vs, ws, steps, x[-1]), (), 20, n)
    rn = _best(lambda *x: _bench_rng(sites["seed"], x[-1]), (), 20, n)

    print(f"  measured on {n:,} real scatter sites from the run above, "
          f"single-threaded, nanoseconds per call")
    print(f"  {'_descend, want_near=False':32s} {1e9 * far:8.0f}")
    print(f"  {'_descend, want_near=True':32s} {1e9 * near:8.0f}"
          f"   <- what a ball step needs")
    print(f"  {'  the ball tax (the difference)':32s} {1e9 * (near - far):8.0f}"
          f"   {near / far:5.2f}x")
    print(f"  {'score_segment':32s} {1e9 * sc:8.0f}")
    print(f"  {'rand + isotropic':32s} {1e9 * rn:8.0f}")

    fl = res.counters["flights"]
    adv = ph_t["advance"]
    import os
    nth = len(os.sched_getaffinity(0))
    print(f"\n  {fl:,} flights took {adv:.3f} s of _advance, "
          f"{1e9 * adv / fl:.0f} ns each on {nth} threads.")
    per = far + sc + rn
    acct = fl * per
    print(f"  one _descend(False) + score_segment + rng is "
          f"{1e9 * per:.0f} ns, of which geometry is "
          f"{100 * far / per:.0f}%.")
    print(f"  that accounts for {100 * acct / (nth * adv):.0f}% of the "
          f"{nth}-thread budget (assuming perfect scaling, so read it as "
          f"a lower\n  bound on geometry's share).")

    sc_n = res.counters.get("scatters", 0)
    cr_n = res.counters.get("crossings", 0)
    hist = res.counters.get("histories", 0)
    print(f"\n  every flight pays a full root-down _descend, and every "
          f"birth pays one too:\n  {hist:,} births + {fl:,} flights. "
          f"Of those flights {sc_n:,} ended in a scatter\n  "
          f"({100 * sc_n / max(fl, 1):.0f}%) and {cr_n:,} in a surface "
          f"crossing ({100 * cr_n / max(fl, 1):.0f}%).")
    print(f"  A scatter does not change the cell, so a cached cell would "
          f"let {100 * sc_n / max(fl, 1):.0f}% of\n  the descends start "
          f"where the particle already is instead of at the root.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"problem": a.problem, "backend": a.backend,
             "r_star": a.r_star, "phases": ph_t, "wall_s": wall,
             "reported_seconds": rep, "counters": res.counters,
             "ns_per_call": {"descend_far": 1e9 * far,
                             "descend_near": 1e9 * near,
                             "score_segment": 1e9 * sc,
                             "rng": 1e9 * rn},
             "advance_s": adv, "sites": int(n)}, indent=1))
        print(f"\n  written {a.json}")


if __name__ == "__main__":
    main()
