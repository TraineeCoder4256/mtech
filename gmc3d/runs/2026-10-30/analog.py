"""OpenMC in its DEFAULT analog mode, the same problems and histories."""
import json, sys
sys.path.insert(0, ".")
from checks import openmc_ref
from problems import PROBLEMS

out = {}
for name in sys.argv[1].split(","):
    mdl = PROBLEMS[name]()
    mdl.settings.survival_biasing = False
    ref = openmc_ref.run(mdl, f"/tmp/vsa_{name}", 1250, 400, seed=101,
                         reuse=False)
    out[name] = {"analog_s": ref.timing["openmc_transport_s"],
                 "leakage": ref.leakage, "se": ref.leakage_se}
    print(f"{name:9s} OpenMC analog {out[name]['analog_s']:7.2f} s   "
          f"leakage {ref.leakage:.6f} +- {ref.leakage_se:.2g}", flush=True)
json.dump(out, open("/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad/analog.json", "w"), indent=1)
