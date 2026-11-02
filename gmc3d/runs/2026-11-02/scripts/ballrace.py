"""Does the ball step's speed-up survive the v1.6.1 driver overhead?

Run from inside a worktree's gmc3d/.  For each problem, four timed runs in
one process: plain MC and exact-walk balls, each with no tallies and with
the problem's own tallies.  Reports seconds (min of TRIES), flights and
balls per history, the backend's own seconds, and leakage with its error
bar so a FOM can be formed.  Nothing in core/ or ball/ is touched.
"""
import json, sys, time
sys.path.insert(0, ".")
from core import openmc_import, transport as TR
from problems import PROBLEMS
from ball.oracle import Oracle

TAG, RND = sys.argv[1], sys.argv[2]
PROBS = sys.argv[3].split(",")
MESH = sys.argv[4] if len(sys.argv) > 4 else "cap"
N, B, TRIES, RSTAR = 100_000, 5, 3, 2.0

be = Oracle()
rows = []
for name in PROBS:
    pr = openmc_import.load(PROBLEMS[name]())
    tallies = list(pr.tallies)
    for tal in (False, True):
        pr.tallies = tallies if tal else []
        for lab, bk in (("mc", None), ("oracle", be)):
            kw = dict(backend=bk, r_star=RSTAR, mesh_rule=MESH)
            TR.run(pr, 1000, 1, seed=1, **kw)             # compile / warm
            best, res = 1e30, None
            for k in range(TRIES):
                t0 = time.perf_counter()
                r = TR.run(pr, N, B, seed=9 + k, **kw)
                s = time.perf_counter() - t0
                if s < best:
                    best, res = s, r
            n = N * B
            row = dict(tag=TAG, rnd=RND, prob=name, tal=int(tal), be=lab,
                       sec=best, adv=res.timing.get("transport_s", 0.0),
                       bes=res.timing.get("backend_s", 0.0),
                       fl=res.counters["flights"] / n,
                       balls=res.counters.get("balls", 0) / n,
                       leak=res.leakage, se=res.leakage_se,
                       ntal=len(tallies))
            rows.append(row)
            print("BR " + json.dumps(row), flush=True)
    pr.tallies = tallies
json.dump(rows, open(f"/tmp/ballrace_{TAG}_{RND}.json", "w"))
print("BR_DONE", TAG, RND, flush=True)
