"""v1.6: why the slab depth tally's chi2/dof jumps between seeds.

40 independent plain MC runs (seeds 100-139, 100k histories each) of the
slab; for the 60-bin depth flux:
  * the correlation between bins across runs (adjacent, 5 apart, 20 apart)
  * each run's chi2/dof against the 20M OpenMC reference (slab_ref.py)
    and how widely it spreads, against what 60 INDEPENDENT bins would give
    (chi2/dof has standard deviation sqrt(2/60) = 0.18 then)
Run from gmc3d/ after slab_ref.py:  python runs/2026-10-30/v1.6/slab_corr.py
"""
import sys
sys.path.insert(0, "/home/claude/mtech/gmc3d")
import numpy as np
from scipy import stats
from checks import openmc_ref
from core import openmc_import, transport
from problems import PROBLEMS

pr = openmc_import.load(PROBLEMS["slab"]())
ref = openmc_ref.read(sys.argv[1] + "/openmc_slab_20M/statepoint.2000.h5")
rd = ref.tallies["depth"]
runs, chi = [], []
for s in range(100, 140):
    r = transport.run(pr, 20_000, 5, seed=s)
    t = r.tallies["depth"]
    runs.append(t["mean"][:, 0])
    z = (t["mean"][:, 0] - rd["mean"][:, 0]) / np.hypot(t["se"][:, 0],
                                                        rd["se"][:, 0])
    chi.append(np.mean(z ** 2))
X = np.array(runs)
C = np.corrcoef(X.T)
for lag in (1, 2, 5, 10, 20):
    print(f"correlation of depth bins {lag:2d} apart: mean "
          f"{np.mean(np.diag(C, lag)):+.3f}")
chi = np.array(chi)
print(f"\nchi2/dof of 40 runs vs OpenMC 20M: mean {chi.mean():.3f}, sd "
      f"{chi.std(ddof=1):.3f}, min {chi.min():.3f}, max {chi.max():.3f}")
print(f"for 60 independent bins: mean 1.0, sd {np.sqrt(2 / 60):.3f}")
lim = stats.chi2.isf(1e-3, 60) / 60
print(f"runs above the p = 0.001 line ({lim:.3f}): {(chi > lim).sum()} of 40"
      f" (independent bins: 0.04 expected)")
# the same test on the decorrelated (principal component) scale
d = X - X.mean(axis=0)
w, V = np.linalg.eigh(np.cov(X.T))
print(f"\neffective number of independent directions (participation "
      f"ratio of the covariance): {w.sum() ** 2 / (w ** 2).sum():.1f} of 60")

# the v1.5 validate run itself (seed 1, 2M) against the 20M reference
from checks import compare, validate
r = transport.run(pr, 100_000, 20, seed=1)
c = compare.compare(r, ref)
ok, bad = validate.verdict(c)
print(f"\nv1.5's failing run (plain MC, seed 1, 2M) vs OpenMC 20M: depth "
      f"chi2/dof {c['depth/flux']['chi2_dof']:.3f}, depth1d "
      f"{c['depth1d/flux']['chi2_dof']:.3f}: {'PASS' if ok else 'FAIL ' + str(bad)}")
