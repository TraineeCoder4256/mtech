#!/usr/bin/env python3
"""Fig. 4a (MC part): statistical convergence of the cell-averaged standard
deviation of the scalar flux, Eq. (12), for both benchmarks.

Exactly as in the paper: N in {1e5, 2e5, 4e5, 8e5, 1.6e6, 3.2e6, 6.4e6,
1.28e7}, K = 5 independent runs per count.  Use --quick for a reduced
sweep (drops the two largest N).

Usage: python scripts/fig4a_convergence.py [--quick]
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import run_transport, lattice_problem, hohlraum_problem  # noqa: E402

N_LIST = [100_000, 200_000, 400_000, 800_000,
          1_600_000, 3_200_000, 6_400_000, 12_800_000]
K = 5


def sigma_bar(problem, n, k=K, seed0=100):
    runs = np.stack([run_transport(problem, n, seed=seed0 + 1000 * j)
                     for j in range(k)])
    # Eq. (12): per-cell std over the K runs (ddof=1), averaged over cells
    return np.std(runs, axis=0, ddof=1).mean()


# sigma_bar anchors read off the paper's Fig. 4a MC curves at N = 1e5
# (both paper curves follow 1/sqrt(N) exactly).  Our implementation gives
# the same slope at a ~2x lower level (implementation-dependent variance;
# see README).
PAPER_ANCHOR = {"Lattice": 3.3e-3, "Hohlraum": 1.6e-2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="replot from data/fig4a_convergence.npz")
    args = ap.parse_args()
    n_list = N_LIST[:-2] if args.quick else N_LIST

    if args.replot:
        d = np.load(ROOT / "data" / "fig4a_convergence.npz")
        n_list = d["n"].tolist()
        results = {k: d[k] for k in ("Lattice", "Hohlraum")}
    else:
        results = {}
        for name, prob in [("Lattice", lattice_problem()),
                           ("Hohlraum", hohlraum_problem())]:
            sig = []
            for n in n_list:
                t0 = time.time()
                s = sigma_bar(prob, n)
                sig.append(s)
                print(f"{name} N={n:.1e}: sigma_bar={s:.3e} "
                      f"({time.time()-t0:.0f} s)")
            results[name] = np.array(sig)
        np.savez(ROOT / "data" / "fig4a_convergence.npz",
                 n=np.array(n_list), **results)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    narr = np.array(n_list, dtype=float)
    for name, marker in [("Lattice", "o"), ("Hohlraum", "s")]:
        ax.loglog(n_list, results[name], marker + "-",
                  label=f"ours, MC ({name})")
        ax.loglog(n_list, PAPER_ANCHOR[name] * np.sqrt(1e5 / narr), ":",
                  color="0.5", lw=1.2, label=f"paper MC ({name})")
    guide = results["Lattice"][0] * np.sqrt(n_list[0] / narr)
    ax.loglog(n_list, guide, "k--", lw=1, label=r"$1/\sqrt{N}$")
    ax.set_xlabel("Number of particles")
    ax.set_ylabel(r"$\bar\sigma_\phi(N)$")
    ax.legend()
    ax.set_title("Statistical convergence (Eq. 12, K=5 runs)")
    fig.tight_layout()
    out = ROOT / "figures" / "fig4a_convergence.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
