"""One timing round for one version, with a FIXED tally set.

Run from inside a worktree's gmc3d/.  The problems' own tally sets grew
over the versions (cask went from 3 tallies to 6 at v1.1.3), so the
"with tallies" column of regress.py is not comparable across versions.
Here the tally set is cut back to v1.0's in every version: the tallies
v1.0 defined, by name, each scoring flux and absorption only.

One rep per problem per config; the driver interleaves rounds across
versions so machine drift hits every version equally.
"""
import json, sys, time
sys.path.insert(0, ".")
import openmc
from core import openmc_import, transport as TR
from problems import PROBLEMS

TAG, RND = sys.argv[1], sys.argv[2]
PROBS = sys.argv[3].split(",")
N, B = 100_000, 5

KEEP = {"sphere": ("total", "cell", "mesh"),
        "slab": ("total", "depth"),
        "cask": ("total", "cell", "mesh")}

for name in PROBS:
    model = PROBLEMS[name]()
    model.tallies = openmc.Tallies(
        [t for t in model.tallies if t.name in KEEP[name]])
    for t in model.tallies:
        t.scores = ["flux", "absorption"]
    pr = openmc_import.load(model)
    tallies = list(pr.tallies)

    TR.run(pr, 500, 1, seed=1)                      # warm the cache
    t0 = time.perf_counter()
    res = TR.run(pr, N, B, seed=9)
    with_t = time.perf_counter() - t0

    pr.tallies = []
    TR.run(pr, 500, 1, seed=1)
    t0 = time.perf_counter()
    res0 = TR.run(pr, N, B, seed=9)
    no_t = time.perf_counter() - t0
    pr.tallies = tallies

    print(f"ROW {TAG} {RND} {name} no_t {no_t:.3f} with_t {with_t:.3f} "
          f"adv {res0.timing.get('transport_s', float('nan')):.3f} "
          f"flights {res0.counters['flights']} "
          f"leak {res0.leakage:.6f} ntally {len(tallies)}", flush=True)
