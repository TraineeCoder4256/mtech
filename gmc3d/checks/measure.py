"""How much time can ball steps save, and from what ball size? (S1-VAL-5)

Used by: run.py (`python run.py measure`).

Four measurements per problem, each answering one question:

 1. COST OF A FLIGHT.  Plain Monte Carlo: transport seconds / flights.  A
    flight here includes the geometry work (finding the cell and the
    distance to the next surface), so it is dearer in complex geometry.

 2. BIG-BALL SHARE.  Run with the exact walk (oracle) as the backend and
    the smallest threshold, R* = 1.  The oracle counts, for every range of
    R, the balls it was asked for and the scatters inside them, printed
    relative to plain Monte Carlo's scatters.  CAVEAT: inside a ball the
    walk always runs to the edge, while plain Monte Carlo often stops a
    low-weight walk early by Russian roulette, so in absorbing media this
    ratio overstates the work saved and can pass 100% (the sphere: 310%).
    The honest measure of work removed is the drop in flights per history
    in the timed runs below, which `study` prints for each R*.

 3. COST OF A DRAW.  Seconds per (mu, s) draw for each backend, at the
    batch size the run actually produced (requests per backend call).

 4. BREAK-EVEN R*.  A ball of radius R replaces on average k(R) + 1
    flights (k(R) scatters inside, plus the flight out) at a price of one
    draw plus one extra geometry query (the distance field), about one
    flight.  So it pays when
            draw cost + flight cost  <  (k(R) + 1) x flight cost,
    i.e. k(R) > draw cost / flight cost.  k(R) is measured with the exact
    walk.  This is an estimate; research item R1 (choosing R*) is Phani's,
    and the timed runs in `study` give the direct answer.

`study` then times real runs with each backend at several R*, against plain
Monte Carlo with the same number of histories.  Speed-up is the ratio of
wall times.  The leakage FOM ratio (figure of merit, 1 / (variance x time))
also counts any change in noise per history; with 10 batches its own
uncertainty is roughly +-40%, so read it as a rough check.
"""

import time

import numpy as np

from ball import walk
from ball.oracle import HIST_EDGES, Oracle
from core import openmc_import, rng, transport
from problems import PROBLEMS
from . import compare


def flight_cost(problem, particles, batches, seed=3):
    r = transport.run(problem, particles, batches, seed=seed)
    return (r.timing["transport_s"] / max(r.counters["flights"], 1),
            r)


def big_ball_share(problem, particles, batches, mc_scatters, seed=3):
    """Share of all scatters inside balls with R >= each histogram edge."""
    orc = Oracle()
    r = transport.run(problem, particles, batches, seed=seed, backend=orc,
                      r_star=1.0, mesh_rule="centre")
    inside = orc.hist_scatters[::-1].cumsum()[::-1]       # R >= edge
    balls = orc.hist_balls[::-1].cumsum()[::-1]
    total = max(mc_scatters, 1)
    return {"edges": HIST_EDGES[:-1].tolist(),
            "share_of_scatters": (inside / total).tolist(),
            "balls_per_history": (balls / (particles * batches)).tolist(),
            "mean_batch": orc.walks / max(r.counters["backend_calls"], 1)}


def draw_cost(backend, batch, R=5.0, repeats=5):
    """Seconds per draw at a given batch size (best of `repeats`)."""
    Rs = np.full(int(batch), float(R))
    state = rng.streams(np.uint64(9), 0, Rs.size)
    idx = np.arange(Rs.size)
    backend.sample(Rs[:16], state, idx[:16])                # warm up
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        backend.sample(Rs, state, idx)
        best = min(best, time.perf_counter() - t)
    return best / Rs.size


def scatters_per_ball(radii=(1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30), n=20000):
    """k(R): mean scatters inside a collided ball walk."""
    return {float(R): float(walk.sample(R, n, seed=5)[2].mean())
            for R in radii}


def break_even(k_of_R, draw_s, flight_s):
    """Smallest tabulated R where (k(R) + 1) flights cost more than one
    draw plus one flight."""
    for R in sorted(k_of_R):
        if k_of_R[R] * flight_s > draw_s:
            return R
    return float("inf")


def study(names, backends, r_stars=(2.0, 3.0, 5.0, 10.0), particles=100_000,
          batches=10, seed=3, log=print):
    """All four measurements, then timed runs with each backend at each R*,
    compared with plain Monte Carlo at the same number of histories.

    backends: {name: backend object}.  Returns a dict of everything."""
    k_of_R = scatters_per_ball()
    log("k(R), mean scatters inside a collided ball walk:")
    log("  " + "  ".join(f"R={R:g}: {k:.1f}" for R, k in k_of_R.items()))
    out = {"k_of_R": k_of_R, "problems": {}}
    for name in names:
        pr = openmc_import.load(PROBLEMS[name]())
        transport.run(pr, 1000, 1, seed=seed)                # compile
        t_fl, mc = flight_cost(pr, particles, batches, seed)
        share = big_ball_share(pr, particles, batches,
                               mc.counters["scatters"], seed)
        batch = max(int(share["mean_batch"]), 1)
        costs = {b: draw_cost(be, batch) for b, be in backends.items()}
        log(f"\n== {name}")
        log(f"  plain MC: {mc.counters['flights']:,} flights in "
            f"{mc.timing['transport_s']:.2f} s = {1e9 * t_fl:.0f} ns per "
            f"flight; {mc.counters['scatters'] / (particles * batches):.1f}"
            f" scatters per history")
        log("  scatters inside balls with R >= X, relative to plain MC's "
            "scatters (see caveat in measure.py):")
        log("    " + "  ".join(
            f"X={e:g}: {100 * f:.1f}%" for e, f in
            zip(share["edges"], share["share_of_scatters"])
            if e <= 30))
        log(f"  mean requests per backend call: {share['mean_batch']:.0f}")
        rows = {}
        for b, c in costs.items():
            be_R = break_even(k_of_R, c, t_fl)
            log(f"  {b}: {1e9 * c:.0f} ns per draw = "
                f"{c / t_fl:.2f} flights -> break-even R* about {be_R:g}")
            rows[b] = {"draw_s": c, "break_even_R": be_R, "runs": []}
            for rs in r_stars:
                be = backends[b]
                if rs < be.r_min:
                    continue
                r = transport.run(pr, particles, batches, seed=seed + 1,
                                  backend=be, r_star=rs, mesh_rule="centre")
                wall = r.timing["transport_s"] + r.timing["backend_s"]
                cmp = compare.compare(r, mc)
                tot = cmp.get("total/flux", {})
                z_tot = ((tot["ours"] - tot["ref"]) /
                         np.hypot(tot["ours_se"], tot["ref_se"])) \
                    if tot else float("nan")
                # figure of merit 1/(variance x time), relative to plain MC
                fom = (mc.leakage_se ** 2 * mc.timing["transport_s"]) / \
                    max(r.leakage_se ** 2 * wall, 1e-300)
                log(f"    R* = {rs:4g}: {wall:6.2f} s vs plain "
                    f"{mc.timing['transport_s']:.2f} s (speed-up "
                    f"{mc.timing['transport_s'] / wall:.2f}x, leakage FOM "
                    f"x{fom:.2f}); "
                    f"{r.counters['balls'] / (particles * batches):.2f} "
                    f"balls and {r.counters['flights'] / (particles * batches):.1f}"
                    f" flights per history (plain "
                    f"{mc.counters['flights'] / (particles * batches):.1f}); "
                    f"backend {r.timing['backend_s']:.2f}"
                    f" s; leakage z {cmp['leakage']['z']:+.2f}, total flux "
                    f"z {z_tot:+.2f}")
                rows[b]["runs"].append({"r_star": rs, "wall_s": wall,
                                        "leakage_fom_ratio": fom,
                                        "timing": r.timing,
                                        "counters": r.counters,
                                        "vs_plain_mc": cmp})
        out["problems"][name] = {"flight_s": t_fl,
                                 "mc_timing": mc.timing,
                                 "mc_counters": mc.counters,
                                 "big_ball_share": share, "backends": rows}
    return out
