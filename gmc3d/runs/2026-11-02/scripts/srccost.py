"""How long the source sampler takes per batch (v1.2 and later).

Before v1.2 the source was sampled inside the parallel njit `_birth`;
since v1.2 Python/numpy draws the batch's sites first and the geometry
check behind it (`G.paths_many`) is a SERIAL njit loop.  This times that
call on its own, no code changed.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from core import openmc_import, transport as TR
from problems import PROBLEMS

TAG = sys.argv[1]
for name in sys.argv[2].split(","):
    pr = openmc_import.load(PROBLEMS[name]())
    src = getattr(pr, "source", None)
    if src is None or not hasattr(src, "sample"):
        print(f"SRC {TAG} {name:7s} n/a (pre-v1.2 source)", flush=True)
        continue
    src.sample(1000, np.random.default_rng(99))          # warm
    best = 1e30
    for k in range(5):
        g = np.random.default_rng(k)
        t = time.perf_counter()
        s = src.sample(100_000, g)
        best = min(best, time.perf_counter() - t)
    print(f"SRC {TAG} {name:7s} {best * 1000:7.1f} ms per batch of 100k"
          f"   -> {best * 5:5.2f} s of a 5-batch run"
          f"   redrawn {s.redrawn}", flush=True)
