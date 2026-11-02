"""OpenMC with survival biasing (implicit capture), the fair reference.

500 x 1000 = 500,000 histories, finely batched so its error bar is as well
determined as ours.  Three seeds; the fastest transport time is kept.
"""
import json, sys
sys.path.insert(0, ".")
from checks import openmc_ref
from problems import PROBLEMS

out = {}
for name in sys.argv[1].split(","):
    best = None
    for k in range(3):
        mdl = PROBLEMS[name]()
        mdl.settings.survival_biasing = True
        mdl.settings.cutoff = {"weight": 0.25, "weight_avg": 1.0}
        ref = openmc_ref.run(mdl, f"/tmp/oref_{name}_{k}", 500, 1000,
                             seed=201 + k, reuse=False)
        s = ref.timing["openmc_transport_s"]
        if best is None or s < best[0]:
            best = (s, ref.leakage, ref.leakage_se)
    out[name] = {"implicit_s": best[0], "leakage": best[1], "se": best[2]}
    print(f"OREF {name:9s} {best[0]:7.2f} s  leakage {best[1]:.6f} "
          f"+- {best[2]:.2g}", flush=True)
json.dump(out, open(f"{sys.argv[2]}/openref.json", "w"), indent=1)
print("OREF_DONE", flush=True)
