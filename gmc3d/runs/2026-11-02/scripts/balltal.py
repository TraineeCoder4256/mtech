"""Which tallies switch the ball step off, or cap it?

Run from inside a worktree's gmc3d/.  For each problem: one timed pair
(plain MC, exact-walk balls) with no tallies, then one pair per tally of
the problem's own set, each tally ALONE.  flights and balls per history
say whether the ball step survived that tally; seconds say what it cost.
Nothing in core/ or ball/ is touched.
"""
import json, sys, time
sys.path.insert(0, ".")
from core import openmc_import, transport as TR
from problems import PROBLEMS
from ball.oracle import Oracle

PROBS = sys.argv[1].split(",")
N, B, TRIES, RSTAR = 100_000, 5, 2, 2.0
be = Oracle()
rows = []
for name in PROBS:
    pr = openmc_import.load(PROBLEMS[name]())
    tallies = list(pr.tallies)
    sets = [("(none)", [])] + [(getattr(t, "name", f"t{i}") or f"t{i}", [t])
                               for i, t in enumerate(tallies)]
    for label, ts in sets:
        pr.tallies = ts
        for lab, bk in (("mc", None), ("oracle", be)):
            kw = dict(backend=bk, r_star=RSTAR, mesh_rule="profile")
            TR.run(pr, 1000, 1, seed=1, **kw)
            best, res = 1e30, None
            for k in range(TRIES):
                t0 = time.perf_counter()
                r = TR.run(pr, N, B, seed=9 + k, **kw)
                s = time.perf_counter() - t0
                if s < best:
                    best, res = s, r
            n = N * B
            row = dict(prob=name, tally=label, be=lab, sec=best,
                       bes=res.timing.get("backend_s", 0.0),
                       fl=res.counters["flights"] / n,
                       balls=res.counters.get("balls", 0) / n,
                       leak=res.leakage, se=res.leakage_se)
            rows.append(row)
            print("BT " + json.dumps(row), flush=True)
    pr.tallies = tallies
json.dump(rows, open(f"{sys.argv[2]}/balltal.json", "w"), indent=1)
print("BT_DONE", flush=True)
