import numpy as np, torch
torch.set_num_threads(1)
import benchmark as B, metrics as M
from data import load_dataset
from generators.oracle import Oracle
from sampler import CellSampler
from solve import gmc_solve
ds = load_dataset("data/cells.npz", 0.05, 0)
s = CellSampler(Oracle(ds["ynorm"]), ds["ynorm"], ds["cnorm"])
ref = B.load_reference(1); R = ref["mc_cell"]
F = [gmc_solve(B.problem(1), 20000, s, seed=500 + k)[0] for k in range(8)]
e = [M.rel_l2(f, R) for f in F]
print("oracle-GMC vs ref, 8 runs:", " ".join(f"{x*100:.2f}" for x in e), f"mean {np.mean(e)*100:.3f}")
fl = [M.rel_l2(f, R) for f in ref["floor_cells"]]
print("mc.py      vs ref, 6 runs:", " ".join(f"{x*100:.2f}" for x in fl), f"mean {np.mean(fl)*100:.3f}")
G = np.array(F); sdg = G.std(0, ddof=1); sdm = ref["floor_cells"].std(0, ddof=1)
print("mean of 8 oracle vs ref:", f"{M.rel_l2(G.mean(0), R)*100:.3f}%  (pure noise would give ~{np.mean(e)/np.sqrt(8)*100:.3f}%)")
print("per-cell sd ratio oracle/mc, source cell and 4 neighbours:",
      [f"{sdg[y,x]/sdm[y,x]:.2f}" for y,x in [(3,3),(3,2),(3,4),(2,3),(4,3)]])
print("flux mean in those cells, oracle/ref:", [f"{G.mean(0)[y,x]/R[y,x]:.4f}" for y,x in [(3,3),(3,2),(3,4),(2,3),(4,3)]])
