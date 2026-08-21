#!/usr/bin/env python3
"""Fig. 4b (MC part): wall-clock cost of standard MC cell transmission vs
cell optical thickness, for N in {5e3, 2e4, 8e4, 3.2e5} particles.

Square pure-scattering cells of side L mean free paths, L from 1e-2 to
1e3; particles enter the left face (cosine-law incidence, uniform along
the face).  Standard MC must resolve every scattering event, so its cost
grows with optical thickness, while GMC (not reproduced here -- it is the
neural sampler) stays O(1).  The mean number of scattering events per
particle is also recorded: it is the hardware-independent cost measure.

Usage: python scripts/fig4b_scaling.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import time_single_cell  # noqa: E402

N_LIST = [5_000, 20_000, 80_000, 320_000]
L_LIST = np.logspace(-2, 3, 11)


def main():
    time_single_cell(1000, 1.0)  # JIT warm-up, excluded from timings

    times = np.zeros((len(N_LIST), len(L_LIST)))
    scats = np.zeros_like(times)
    for i, n in enumerate(N_LIST):
        for j, L in enumerate(L_LIST):
            t, k = time_single_cell(n, L, seed=1 + j, repeats=2)
            times[i, j], scats[i, j] = t, k
            print(f"N={n:>7d}  L={L:9.3g} mfp  t={t:8.4f} s  "
                  f"<scatters>={k:10.1f}")
    np.savez(ROOT / "data" / "fig4b_scaling.npz",
             n=np.array(N_LIST), L=L_LIST, t=times, k=scats)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, n in enumerate(N_LIST):
        axes[0].loglog(L_LIST, times[i], "o-", ms=3, label=f"MC, N={n}")
    axes[0].set_xlabel("Cell size (mean free paths)")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("MC cost vs optical thickness")

    axes[1].loglog(L_LIST, scats[-1], "o-", ms=3, color="crimson")
    axes[1].set_xlabel("Cell size (mean free paths)")
    axes[1].set_ylabel("Mean scattering events / particle")
    axes[1].set_title("Hardware-independent cost")
    fig.suptitle("Fig. 4b (MC part): computational scaling")
    fig.tight_layout()
    out = ROOT / "figures" / "fig4b_scaling.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
