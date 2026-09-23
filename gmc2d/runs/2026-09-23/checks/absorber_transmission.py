"""Throwaway: does CFM get the surviving weight through an absorber cell right?
In solve.py an absorber cell of scattering size W = sig_s*L multiplies the weight
by exp(-(sig_a/sig_s) * s), with s the optical scattering path; sig_a/sig_s = 19."""
import numpy as np, torch
import mc
from data import load_dataset
from generators.cfm import CFM
from sampler import CellSampler
ds = load_dataset("data/cells.npz", 0.05, 0)
s = CellSampler(CFM.load("models/cfm"), ds["ynorm"], ds["cnorm"])
n = 40000
for scale in (1, 4, 10, 20):
    W = 0.5 * scale
    for xi, ox, oy in ((0.5, 1.0, 0.0), (0.3, 0.6, 0.5)):
        a = mc.sample_cell(n, W, W, xi=xi, direction=(ox, oy), seed=11)
        g = s.sample(np.full(n, W), np.full(n, W), np.full(n, xi), np.full(n, ox), np.full(n, oy), seed=12)
        tm, tg = np.exp(-19 * a["s"]), np.exp(-19 * g["s"])
        se = np.hypot(tm.std(), tg.std()) / np.sqrt(n)
        print(f"scale {scale:>2}  W {W:>4}  entry ({xi},{ox},{oy})  surviving weight MC {tm.mean():.4e}  "
              f"CFM {tg.mean():.4e}  ratio {tg.mean()/tm.mean():.3f}  z {(tg.mean()-tm.mean())/se:+.1f}")
