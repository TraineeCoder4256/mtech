"""The problem's OWN full tally set, ours only: no tallies vs all."""
import sys, time
sys.path.insert(0, ".")
from core import openmc_import, transport as TR
from problems import PROBLEMS
N, B = 100_000, 5
TAG = sys.argv[1]
for name in sys.argv[2].split(","):
    pr = openmc_import.load(PROBLEMS[name]())
    keep = list(pr.tallies)
    TR.run(pr, 1000, 1, seed=1)
    best = 1e30
    for k in range(2):
        t = time.perf_counter(); TR.run(pr, N, B, seed=9)
        best = min(best, time.perf_counter() - t)
    pr.tallies = []
    TR.run(pr, 1000, 1, seed=1)
    bare = 1e30
    for k in range(2):
        t = time.perf_counter(); TR.run(pr, N, B, seed=9)
        bare = min(bare, time.perf_counter() - t)
    pr.tallies = keep
    print(f"OWN {TAG} {name:7s} all {best:7.2f} s   none {bare:7.2f} s   "
          f"tallies {best - bare:7.2f} s   ntally {len(keep)}", flush=True)
