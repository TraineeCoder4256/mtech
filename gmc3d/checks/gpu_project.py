"""What a measured GPU cost per draw would do to a whole run.

Used by: run by hand (NumPy only; no OpenMC, no PyTorch).  From gmc3d/:

    python -m checks.gpu_project --costs t4_compiled.json
    python -m checks.gpu_project --costs runs/2026-10-06/net_profile_cpu.json

Uses: the JSON that checks/loop_profile.py wrote (our runs, with the
ball counts, the requests per call and the error bars at each in-flight
setting), the race JSON (OpenMC's own numbers on the same problems) and a
cost curve {requests per call: seconds per draw} measured on a device by
checks/net_profile.py or checks/net_variants.py.

The model is one line:

    run seconds  =  transport seconds  +  balls x cost(requests per call)

with "transport seconds" taken from the lookup-table run at the same R*,
because the table costs about 10 ns per draw and so stands in for a free
backend with the right physics.  The model is printed against the measured
CPU network run so its error is visible before it is trusted.

Time alone is not the verdict: our solver and OpenMC have different noise
per history, so the row that matters is the figure of merit,
FOM = 1 / (relative error^2 x seconds), which is how many times faster a
contender reaches the same error bar.  FOM does not depend on how many
histories were run, so our 2M-history rows and the race's 4M-history
OpenMC rows can be compared directly.

With --overlap the table also shows what the run would cost if the CPU
advanced the next wave while the GPU drew the current one, i.e.
max(transport, balls x cost) instead of the sum -- the value of making the
backend call asynchronous.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
LOOP = HERE / "runs" / "2026-10-06" / "loop_profile.json"
RACE = HERE / "runs" / "2026-10-05" / "race.json"


def curve(path):
    d = json.loads(Path(path).read_text())
    d = d.get("by_batch", d.get("ns_per_draw", d))
    keys = sorted(d, key=float)
    xs = np.log([float(k) for k in keys])
    vals = [d[k] for k in keys]
    if isinstance(vals[0], dict):                 # net_variants format
        raise SystemExit("pick one variant: pass {batch: seconds} JSON")
    ys = np.log([float(v) for v in vals])
    scale = 1e-9 if max(float(v) for v in vals) > 1e-3 else 1.0
    return lambda n: float(np.exp(np.interp(np.log(max(n, 1.0)), xs, ys))) \
        * scale


def from_race(race, cost_at, multipliers, label, ceiling="ours + table",
              log=print):
    """Project onto the 5 Oct race, whose 40 batches give trustworthy error
    bars (ours here give only 20, and the error on an error bar is about
    sqrt(2/N)).  The lookup-table row stands in for a free backend: it has
    the same ball counts and the same physics at about 10 ns per draw.

    `multipliers` are how many times more particles are kept in flight per
    batch than the race's 100,000; the backend's batch grows with it while
    the answer does not (see checks/loop_profile.py).
    """
    log(f"\nProjected onto the race, network cost from {label}; the "
        f"ceiling row is {ceiling!r}.")
    log("Figure of merit = 1 / (relative error^2 x seconds), as a multiple "
        "of OpenMC analog's.")
    log(f"  {'problem':10s} {'OpenMC':>7s} {'ours MC':>8s} "
        f"{'free':>6s}" + "".join(f"{'x' + str(m) + ' flight':>13s}"
                                  for m in multipliers))
    out = {}
    for name, rows in race.items():
        om = rows[0]["fom"]["leakage"]
        mc = next(r for r in rows if r["label"] == "ours, plain MC")
        tab = next((r for r in rows if r["label"].startswith(ceiling)),
                   None)
        if tab is None or not tab["counters"]["balls"]:
            log(f"  {name:10s} {1.0:6.2f}x {mc['fom']['leakage'] / om:7.2f}x"
                f"    no ball fits at R* = 2")
            continue
        balls = tab["counters"]["balls"]
        calls = tab["counters"]["backend_calls"]
        per_call = balls / calls
        base_s = tab["seconds"] - tab["backend_s"]
        m, se = tab["quantities"]["leakage"]
        rel = se / m
        free = 1.0 / (rel ** 2 * base_s) / om
        cells, vals = [], {}
        for k in multipliers:
            c = cost_at(per_call * k)
            t = base_s + balls * c
            f = 1.0 / (rel ** 2 * t) / om
            cells.append(f"{f:8.2f}x ({1e9 * c:.0f}ns)")
            vals[k] = {"per_call": per_call * k, "ns_per_draw": 1e9 * c,
                       "seconds": t, "fom_x_openmc": f}
        log(f"  {name:10s} {1.0:6.2f}x {mc['fom']['leakage'] / om:7.2f}x "
            f"{free:5.2f}x " + " ".join(cells))
        out[name] = {"free_fom": free, "plain_mc_fom": mc["fom"]["leakage"] / om,
                     "per_call_at_race": per_call, "balls": balls,
                     "ceiling_seconds": base_s, "rel_err": rel,
                     "projected": vals}
    return out


def openmc_fom(race, name):
    """OpenMC analog's leakage FOM on this problem, and its seconds."""
    rows = race.get(name)
    if not rows:
        return None
    om = rows[0]
    return om["fom"]["leakage"], om["seconds"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--costs", required=True,
                   help="JSON {requests: seconds or ns per draw}")
    p.add_argument("--loop", default=str(LOOP))
    p.add_argument("--race", default=str(RACE))
    p.add_argument("--label", default="this device")
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--from-race", action="store_true",
                   help="project onto the 5 Oct race instead (better error "
                        "bars); --flight sets the in-flight multipliers")
    p.add_argument("--flight", type=int, nargs="*", default=[1, 4, 10, 40])
    p.add_argument("--ceiling", default="ours + table",
                   help="label prefix of the row that stands in for a free "
                        "backend (a race with several R* has one per R*)")
    a = p.parse_args()

    cost_at = curve(a.costs)
    loop = json.loads(Path(a.loop).read_text())
    race = json.loads(Path(a.race).read_text())

    print(f"cost curve: {a.costs} ({a.label})")
    print("  requests per call ->  ns per draw: " + ", ".join(
        f"{n:,}: {1e9 * cost_at(n):.0f}"
        for n in (1_000, 3_000, 10_000, 30_000, 100_000)))

    if a.from_race:
        from_race(race, cost_at, a.flight, a.label, a.ceiling)
        return

    for name, d in loop.items():
        rows, plain = d["rows"], d["plain"]
        if not rows[0]["balls"]:
            print(f"\n== {name}: no ball fits at R* = {d['r_star']:g}; "
                  "nothing to accelerate")
            continue
        om = openmc_fom(race, name)
        print(f"\n== {name}, R* = {d['r_star']:g}, "
              f"{rows[0]['particles'] * rows[0]['batches']:,} histories; "
              f"plain MC {plain['transport_s']:.2f} s, ceiling "
              f"{rows[0]['transport_s']:.2f} s")
        # The relative error of the ANSWER does not depend on how the
        # histories are split into batches (loop_profile.log shows the same
        # leakage to five figures), and since core/tallies.py takes its
        # error bar from the thread chunks, every row now has hundreds of
        # groups behind it and can be used as it stands.  JSON written
        # before that change has only `batches` groups, which is far too
        # noisy on a 2- or 5-batch row, so there the 20-batch row's error is
        # used for every row instead.
        fine = all(r.get("groups", 0) >= 64 for r in rows)
        rel20 = rows[0]["leakage_se"] / rows[0]["leakage"]
        head = (f"  {'in flight':>11s} {'per call':>9s} {'ns/draw':>8s} "
                f"{'run s':>7s} {'x plain MC':>11s} {'x ceiling':>10s}")
        if om:
            head += f" {'FOM x OpenMC':>13s}"
        if a.overlap:
            head += f" {'overlapped':>11s}"
        print(head)
        for r in rows:
            per = cost_at(r["per_call"])
            draw = r["balls"] * per
            tot = r["transport_s"] + draw
            line = (f"  {r['particles']:>11,} {r['per_call']:>9,.0f} "
                    f"{1e9 * per:>8.0f} {tot:>7.2f} "
                    f"{plain['transport_s'] / tot:>10.2f}x "
                    f"{tot / r['transport_s']:>9.2f}x")
            if om:
                rel = r["leakage_se"] / r["leakage"] if fine else rel20
                fom = 1.0 / (rel ** 2 * tot)
                line += f" {fom / om[0]:>12.2f}x"
            if a.overlap:
                ov = max(r["transport_s"], draw)
                line += f" {ov:>10.2f}s"
            print(line)
        if om:
            best = rows[0]
            print(f"  for reference: a free backend would be "
                  f"{(1.0 / (rel20 ** 2 * best['transport_s'])) / om[0]:.2f}x "
                  f"OpenMC analog, and plain MC "
                  f"{(1.0 / ((plain['leakage_se'] / plain['leakage']) ** 2 * plain['transport_s'])) / om[0]:.2f}x")


if __name__ == "__main__":
    main()
