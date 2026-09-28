"""Calibration of the per-cell chi2 in metrics.lattice_accuracy (28 Sept 2026).

Scores EXACT Monte Carlo as if it were a model: 12 groups of 6 fresh
mc.solve runs per scale, seeds 1000 apart (mc.solve seeds its 64 blocks
seed, seed+1, ..., so seeds closer than 64 share random numbers).  An exact
method must score ~1.  The "old" column is the pre-28-Sept statistic
(floor spread, every cell, 2 runs) on the same runs.
Run from gmc2d/:  PYTHONPATH=. python runs/2026-09-28/checks/chi2_calibration.py
"""
import numpy as np, benchmark as B, metrics as M, mc
G_PER, GROUPS = 6, 12
for sc in B.SCALES:
    ref = B.load_reference(sc); prob = B.problem(sc); R = ref["mc_cell"]
    new, old, cells = [], [], []
    for g in range(GROUPS):
        runs = [B.coarsen(mc.solve(prob, B.N, seed=700_000_000 + 1000*(G_PER*g + k) + sc)[0], prob["per_cm"])
                for k in range(G_PER)]
        a = M.lattice_accuracy(runs, ref); new.append(a["cell_chi2"]); cells.append(a["cell_chi2_cells"])
        # old statistic: floor spread, all cells, first 2 runs (as the old pipeline)
        G2 = np.mean(runs[:2], 0); sd = ref["floor_cells"].std(0, ddof=1)
        se = np.sqrt(sd**2/2 + ref["mc_cell_se"]**2); z = (G2-R)/np.where(se > 0, se, np.inf)
        old.append(np.mean(z**2))
    print(f"scale {sc:>2}: new chi2 mean {np.mean(new):.2f} (min {min(new):.2f}, max {max(new):.2f}), "
          f"{cells[0]} of 49 cells scored | old chi2 mean {np.mean(old):.1f} (max {max(old):.1f})", flush=True)
