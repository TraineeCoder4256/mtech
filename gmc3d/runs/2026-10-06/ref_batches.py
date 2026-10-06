"""Does splitting the OpenMC reference into more batches cost it time?

Same histories (4M), several splits, on the fast problems where per-batch
overhead would show up most.  Reports transport seconds (what race.py
times), the answer and the reported error bar.
"""
import json, sys
sys.path.insert(0, ".")
from checks import openmc_ref
from checks.race import PROBLEMS

N = 4_000_000
SPLITS = [40, 200, 1000]
out = {}
for name in ("sphere", "slab", "cask"):
    out[name] = []
    for b in SPLITS:
        model = PROBLEMS[name]()
        model.settings.survival_biasing = False
        r = openmc_ref.run(model, f"/tmp/refb_{name}_{b}", max(N // b, 1), b,
                           seed=101, reuse=False)
        row = {"batches": b, "particles": max(N // b, 1),
               "transport_s": r.timing["openmc_transport_s"],
               "wall_s": r.timing["wall_s"],
               "leakage": r.leakage, "leakage_se": r.leakage_se}
        out[name].append(row)
        print(f"{name:7s} {b:5d} x {row['particles']:>8,}  "
              f"transport {row['transport_s']:7.2f} s  wall {row['wall_s']:7.2f} s  "
              f"leakage {r.leakage:.6g} +- {r.leakage_se:.3g} "
              f"(rel {r.leakage_se/r.leakage*100:.4f}%)", flush=True)
    print(flush=True)
json.dump(out, open("runs/2026-10-06/ref_batches.json", "w"), indent=1)
