"""The driver's call shape: how big the backend's batches are, and why.

Used by: run by hand (needs OpenMC, like the rest of checks/).  From gmc3d/:

    python -m checks.loop_profile                       # every problem
    python -m checks.loop_profile --problems slab cask
    python -m checks.loop_profile --gpu-costs costs.json

Uses: core/transport.py (unchanged), ball/table.py and ball/oracle.py,
problems/.

WHY THIS FILE.  A GPU's cost per draw is a steep function of how many draws
one call asks for: on a T4 the same network costs 3.6 us per draw at 1,000
requests and 0.30 us at 100,000, because below ~10,000 the call is all
kernel launches.  checks/race.py showed the problems produce only
2,700-12,700 requests per call -- right in the steep part.  So before any
GPU tuning, the question is: WHAT SETS that number, and can the driver be
made to ask for more at once?

core/transport.py runs one statistical batch at a time.  Inside a batch it
advances every live particle until each dies or wants a ball, then sends
all the waiting radii in ONE call ("a wave").  So

    requests per call  =  balls per history x particles per batch / waves,

and `waves` is how many ball steps a history takes one after another --
a property of the problem, not of the batch size.  Raising `particles` per
batch therefore raises the backend's batch size in proportion, for free.
This file measures that, for each problem:

 1. IN-FLIGHT SWEEP.  The same number of histories split as few big
    batches or many small ones.  Prints balls, calls, requests per call,
    waves per batch, and the time, so the proportionality can be checked
    rather than assumed.  CAVEAT: the error BAR is estimated from the
    scatter between batches, so fewer batches means a noisier estimate of
    the error (not a worse answer).  The printed leakage and its error show
    what that costs.

 2. THE CEILING.  The lookup table costs about 10 ns per draw, so a table
    run is what a FREE backend would do, with the right physics.  Its time
    is the floor any network run must approach, and
        (10% of the table run's transport time) / balls
    is the budget per draw for a network run to land within 10% of that
    floor.  That is the number a GPU has to hit.

 3. HOW THE TIME SPLITS.  Transport seconds against backend seconds, and
    max(transport, backend) -- what the run would cost if the CPU advanced
    the next wave while the GPU drew the current one.  The gap between the
    sum and the max is what overlapping the two would be worth.

 4. PROJECTION (with --gpu-costs).  Given a measured cost-per-draw curve
    from a real device (the JSON written by checks/net_profile.py or a
    {batch: seconds} file), the run time at each in-flight setting, from
        table transport time + balls x cost(requests per call).
    Checked against the measured CPU network run at the same R*, so the
    model is validated before it is trusted.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ball import oracle, table
from core import openmc_import, rng, transport
from problems import PROBLEMS

HISTORIES = 2_000_000
SPLITS = ((20, "many small"), (5, "fewer, bigger"), (2, "two"))
TABLE_FILE = Path(__file__).resolve().parent.parent / "data" / "ball" / \
    "table.npz"


def _row(pr, be, particles, batches, r_star, seed=7):
    t = time.perf_counter()
    r = transport.run(pr, particles, batches, seed=seed, backend=be,
                      r_star=r_star, mesh_rule="centre")
    wall = time.perf_counter() - t
    c, tm = r.counters, r.timing
    calls = max(c["backend_calls"], 1)
    return {"particles": particles, "batches": batches,
            "balls": c["balls"], "flights": c["flights"], "calls": calls,
            "per_call": c["balls"] / calls, "waves_per_batch": calls / batches,
            "transport_s": tm["transport_s"], "backend_s": tm["backend_s"],
            "wall_s": wall, "leakage": r.leakage, "leakage_se": r.leakage_se,
            "balls_per_history": c["balls"] / (particles * batches)}


def cost_curve(path):
    """{requests: seconds per draw} -> a log-log interpolator."""
    d = json.loads(Path(path).read_text())
    d = d.get("by_batch", d)
    b = np.log([float(k) for k in sorted(d, key=float)])
    c = np.log([float(d[k]) for k in sorted(d, key=float)])
    return lambda n: float(np.exp(np.interp(np.log(max(n, 1.0)), b, c)))


def study(names, r_star=2.0, histories=HISTORIES, splits=SPLITS,
          gpu_costs=None, log=print):
    tab = table.Table.load(TABLE_FILE)
    cost_at = cost_curve(gpu_costs) if gpu_costs else None
    out = {}
    for name in names:
        pr = openmc_import.load(PROBLEMS[name]())
        transport.run(pr, 1000, 1, seed=7)                   # compile
        tab.sample(np.full(16, 5.0), rng.streams(np.uint64(9), 0, 16),
                   np.arange(16))
        plain = _row(pr, None, histories // splits[0][0], splits[0][0],
                     r_star)
        log(f"\n== {name}: {histories:,} histories, R* = {r_star:g}")
        log(f"  plain MC (no balls): {plain['transport_s']:.2f} s, "
            f"{plain['flights'] / histories:.1f} flights per history")
        log(f"  {'split':>22s} {'balls/hist':>10s} {'calls':>7s} "
            f"{'per call':>9s} {'waves/batch':>12s} {'transport':>10s} "
            f"{'backend':>8s} {'leakage':>9s} {'+- se':>8s}")
        rows = []
        for batches, label in splits:
            n = histories // batches
            r = _row(pr, tab, n, batches, r_star)
            rows.append(r)
            log(f"  {label + f' ({batches} x {n:,})':>22s} "
                f"{r['balls_per_history']:10.2f} {r['calls']:7,} "
                f"{r['per_call']:9,.0f} {r['waves_per_batch']:12.1f} "
                f"{r['transport_s']:9.2f}s {r['backend_s']:7.2f}s "
                f"{r['leakage']:9.5f} {r['leakage_se']:8.5f}")
        base = rows[0]
        if base["balls"]:
            budget = 0.1 * base["transport_s"] / base["balls"]
            log(f"  CEILING: with a free backend this problem runs in "
                f"{base['transport_s']:.2f} s "
                f"({plain['transport_s'] / base['transport_s']:.2f}x plain "
                f"MC). To land within 10% of it the backend has "
                f"{1e9 * budget:.0f} ns per draw.")
            log(f"  OVERLAP: transport {base['transport_s']:.2f} s + backend "
                f"{base['backend_s']:.2f} s; if they ran at the same time "
                f"the run would be max() = "
                f"{max(base['transport_s'], base['backend_s']):.2f} s. "
                f"{base['calls']:,} calls = {base['calls']:,} syncs.")
            if cost_at:
                log(f"  PROJECTION at the measured cost curve:")
                for r in rows:
                    per = cost_at(r["per_call"])
                    tot = r["transport_s"] + r["balls"] * per
                    log(f"    {r['batches']:3d} x {r['particles']:>9,}: "
                        f"{r['per_call']:>8,.0f} per call -> "
                        f"{1e9 * per:6.0f} ns per draw -> {tot:7.2f} s "
                        f"({plain['transport_s'] / tot:5.2f}x plain MC, "
                        f"{tot / r['transport_s']:4.2f}x the ceiling)")
        else:
            log("  no ball of this size fits: nothing to accelerate")
        out[name] = {"plain": plain, "rows": rows, "r_star": r_star}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--problems", nargs="*", default=None)
    p.add_argument("--r-star", type=float, default=2.0)
    p.add_argument("--histories", type=int, default=HISTORIES)
    p.add_argument("--gpu-costs", default=None,
                   help="JSON from checks/net_profile.py --json")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    res = study(a.problems or list(PROBLEMS), a.r_star, a.histories,
                gpu_costs=a.gpu_costs)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))
        print(f"\nwritten to {a.json}")


if __name__ == "__main__":
    main()
