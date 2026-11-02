"""One tally at a time, ours against OpenMC, same session, same histories.

For each problem: run with no tallies, then with each of the problem's own
tallies ALONE, then with all of them.  Both codes see exactly the same
tally objects, so the incremental cost of a tally is comparable, and a
tally type we are slow at shows up on its own row.

OpenMC runs with survival biasing and the same weight cutoff, so both
sides play the same (implicit capture) game.
"""
import json, sys, time
sys.path.insert(0, ".")
import openmc
from checks import openmc_ref
from core import openmc_import, transport as TR
from problems import PROBLEMS

N, B = 100_000, 5          # 500,000 histories
REF_B = 400                # OpenMC: 1,250 x 400
PROBS = sys.argv[1].split(",")
out = {}


def ours(model, reps=2):
    pr = openmc_import.load(model)
    TR.run(pr, 1000, 1, seed=1)                       # warm the cache
    best, res = 1e30, None
    for k in range(reps):
        t = time.perf_counter()
        res = TR.run(pr, N, B, seed=9)
        best = min(best, time.perf_counter() - t)
    return best, res


def theirs(model, tag):
    m = model
    m.settings.survival_biasing = True
    m.settings.cutoff = {"weight": 0.25, "weight_avg": 1.0}
    ref = openmc_ref.run(m, f"/tmp/tc_{tag}", max(N * B // REF_B, 1),
                         REF_B, seed=101, reuse=False)
    return ref.timing["openmc_transport_s"]


for name in PROBS:
    base = PROBLEMS[name]()
    names = [t.name or f"t{i}" for i, t in enumerate(base.tallies)]
    configs = [("none", [])] + [(nm, [nm]) for nm in names] + \
              [("all", names)]
    rows = {}
    for tag, keep in configs:
        model = PROBLEMS[name]()
        model.tallies = openmc.Tallies(
            [t for t in model.tallies if (t.name or "") in keep])
        o, res = ours(model)
        model2 = PROBLEMS[name]()
        model2.tallies = openmc.Tallies(
            [t for t in model2.tallies if (t.name or "") in keep])
        p = theirs(model2, f"{name}_{tag}")
        nbins = sum(int(t.num_bins) for t in model.tallies) \
            if len(model.tallies) else 0
        rows[tag] = {"ours_s": o, "openmc_s": p, "bins": nbins,
                     "flights": res.counters["flights"]}
        print(f"TC {name:7s} {tag:10s} ours {o:7.2f} s   openmc {p:7.2f} s"
              f"   bins {nbins:7d}", flush=True)
    out[name] = rows
    json.dump(out, open("/tmp/claude-0/-home-claude-mtech/"
                        "59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad/"
                        "tallycost.json", "w"), indent=1)
