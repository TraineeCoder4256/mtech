"""v1.6.1: point detectors with the trained network backend (the v1.6
checks used the exact walk only).  Same scattering-sphere model as
checks/detcheck.py; network balls against plain Monte Carlo."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np
import run as R
from checks import detcheck
from core import openmc_import, transport

pr = openmc_import.load(detcheck.scatter_model())
t = time.perf_counter()
mc = transport.run(pr, 100_000, 20, seed=7)
print(f"plain MC {time.perf_counter() - t:.0f} s")
for rs in (1.0, 3.0):
    t = time.perf_counter()
    nb = transport.run(pr, 100_000, 20, seed=8, backend=R.backend("network"),
                       r_star=rs)
    print(f"network r* {rs}: {time.perf_counter() - t:.0f} s, "
          f"{nb.counters['balls']:,} balls")
    for name, _, r0 in pr.detectors:
        a, sa = mc.tallies[name]["mean"][0, 0], mc.tallies[name]["se"][0, 0]
        b, sb = nb.tallies[name]["mean"][0, 0], nb.tallies[name]["se"][0, 0]
        print(f"  {name:12s} plain {a:.6g} +- {sa:.2g}  network {b:.6g} +- "
              f"{sb:.2g}  z {(b - a) / np.hypot(sa, sb):+.2f}")
