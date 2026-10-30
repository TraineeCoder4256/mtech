"""Our plain MC against OpenMC, both in this session, on the same problems.

The point is ns per flight: it separates "we do more work per flight"
(geometry) from "we take more flights" (physics or bookkeeping).
"""
import json, sys, time
sys.path.insert(0, ".")
from checks import openmc_ref
from core import openmc_import, transport as TR
from problems import PROBLEMS

PROBS = sys.argv[1].split(",")
N, B = 100_000, 5
REF_B = 400
out = {}

for name in PROBS:
    mdl = PROBLEMS[name]()
    pr = openmc_import.load(PROBLEMS[name]())
    TR.run(pr, 1000, 1, seed=1)
    ours, res = 1e30, None
    for k in range(2):
        t = time.perf_counter()
        res = TR.run(pr, N, B, seed=9 + k)
        ours = min(ours, time.perf_counter() - t)

    mdl.settings.survival_biasing = True
    mdl.settings.cutoff = {"weight": 0.25, "weight_avg": 1.0}
    ref = openmc_ref.run(mdl, f"/tmp/vso_{name}", max(N * B // REF_B, 1),
                         REF_B, seed=101, reuse=False)
    om = ref.timing["openmc_transport_s"]
    fl = res.counters["flights"]
    hist = N * B
    out[name] = {"ours_s": ours, "openmc_implicit_s": om,
                 "ratio": ours / om, "flights": fl,
                 "flights_per_history": fl / hist,
                 "ns_per_flight": 1e9 * ours / max(fl, 1),
                 "ours_leakage": res.leakage, "openmc_leakage": ref.leakage}
    print(f"{name:9s} ours {ours:7.2f} s   OpenMC implicit {om:7.2f} s   "
          f"{ours / om:5.2f}x   flights/history {fl / hist:7.2f}   "
          f"{1e9 * ours / max(fl, 1):5.0f} ns/flight   "
          f"leakage {res.leakage:.6f} vs {ref.leakage:.6f}", flush=True)

json.dump(out, open("/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad/vsopenmc.json", "w"), indent=1)
