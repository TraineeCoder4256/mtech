"""How much does a race FOM still wander, now that both error bars are fine?

Same race, three seeds.  Each seed is an independent realisation of BOTH
codes and an independent timing, so the spread of the FOM ratio is the real
uncertainty on a published multiplier.  Reports, per contender, the three
FOMs, their spread, and the spread of the seconds alone -- which separates
statistical noise from this container's timing noise.
"""
import json, sys, statistics as st
from pathlib import Path
HERE = Path("/home/claude/mtech/gmc3d"); sys.path.insert(0, str(HERE))
from ball import network, table
from checks import race
from core import output

bes = {"table": table.Table.load(HERE / "data" / "ball" / "table.npz"),
       "network": network.Network.load(network.MODEL_DIR / "flow_seed0")}
CONT = (("ours, plain MC", None, None),
        ("ours + table", "table", 2.0),
        ("ours + network", "network", 2.0))
SEEDS = [11, 2011, 4011]
PROBS = ["sphere", "slab"]

runs = {}
for s in SEEDS:
    print(f"\n######## seed {s}", flush=True)
    runs[s] = race.race(PROBS, bes, particles=100_000, batches=40, seed=s,
                        workdir=str(HERE / "data" / "openmc"),
                        contenders=CONT, log=lambda *p: print(*p, flush=True))

out = {}
def spread(v):
    return 100 * (max(v) - min(v)) / st.mean(v)

print("\n\n==== spread over the three seeds "
      "(ratio = FOM as a multiple of OpenMC analog's)")
for name in PROBS:
    print(f"\n== {name}")
    labels = [r["label"] for r in runs[SEEDS[0]][name]]
    out[name] = {}
    print(f"  {'contender':26s} {'seconds (3 seeds)':>26s} {'spread':>7s}   "
          f"{'FOM x analog':>22s} {'spread':>7s}   {'rel err %':>20s}")
    for i, lab in enumerate(labels):
        secs = [runs[s][name][i]["seconds"] for s in SEEDS]
        foms = [runs[s][name][i]["fom"]["leakage"]
                / runs[s][name][0]["fom"]["leakage"] for s in SEEDS]
        rels = [100 * runs[s][name][i]["quantities"]["leakage"][1]
                / runs[s][name][i]["quantities"]["leakage"][0] for s in SEEDS]
        out[name][lab] = {"seconds": secs, "fom_x_analog": foms,
                          "rel_err_pct": rels,
                          "seconds_spread_pct": spread(secs),
                          "fom_spread_pct": spread(foms)}
        print(f"  {lab:26s} " + " ".join(f"{x:7.2f}" for x in secs)
              + f" {spread(secs):6.1f}%   "
              + " ".join(f"{x:6.2f}" for x in foms)
              + f" {spread(foms):6.1f}%   "
              + " ".join(f"{x:6.4f}" for x in rels))
json.dump(out, open(HERE / "runs/2026-10-06/race_repeat.json", "w"), indent=1)
print("\nwritten runs/2026-10-06/race_repeat.json")
