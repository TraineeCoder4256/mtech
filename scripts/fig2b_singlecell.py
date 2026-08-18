#!/usr/bin/env python3
"""MC part of Fig. 2b: joint distribution of exit angle u_exit vs path
length for single-cell transmissions (80,000 samples), hexbin + marginals.

NOTE: the exact conditioning (cell optical size, entry state, sigma_s) is
specified only in the paper's Supplementary Material, which is not public.
Defaults below (square cell of 0.4 mfp, uniform entry along the left
face, isotropic incidence, sigma_s = 5 cm^-1) were calibrated to match
the figure's observable features: a flat forward exit-angle plateau of
~0.85 with a backscatter shelf of ~0.17, path lengths peaked near
6e-2 cm and spread over several decades, and the
backscattered-particles-travel-farther correlation.  Adjust via CLI.

Usage: python scripts/fig2b_singlecell.py [--W 1] [--H 1] [--sigs 10]
              [--n 80000] [--entry-pos F] [--entry-mu MU] [--seed 1]
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import sample_single_cell  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=0.4, help="cell width, mfp")
    ap.add_argument("--H", type=float, default=0.4, help="cell height, mfp")
    ap.add_argument("--sigs", type=float, default=5.0,
                    help="physical sigma_s (cm^-1) used to convert mfp->cm")
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--entry-pos", type=float, default=None,
                    help="entry height fraction in [0,1]; default uniform")
    ap.add_argument("--entry-mu", type=float, default=None,
                    help="fixed entry x-cosine; default sampled by --entry-law")
    ap.add_argument("--entry-law", default="isotropic",
                    choices=["isotropic", "cosine"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    r = sample_single_cell(args.n, args.W, args.H, mode="boundary",
                           entry_pos=args.entry_pos, entry_mu=args.entry_mu,
                           entry_law=args.entry_law, seed=args.seed)
    u = r["dir"][:, 0]
    s_cm = r["s"] / args.sigs

    fig = plt.figure(figsize=(6.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axu = fig.add_subplot(gs[0, 0], sharex=ax)
    axs = fig.add_subplot(gs[1, 1], sharey=ax)

    hb = ax.hexbin(u, np.log10(s_cm), gridsize=60,
                   extent=(-1, 1, -5, 1), cmap="Reds", mincnt=1)
    ax.set_xlabel(r"$u_{\rm exit}$ (x-direction cosine)")
    ax.set_ylabel(r"$\log_{10}$ path length (cm)")
    ax.set_ylim(-5, 1)

    axu.hist(u, bins=80, range=(-1, 1), density=True,
             color="crimson", histtype="step", lw=1.4)
    axu.set_ylabel("PDF")
    plt.setp(axu.get_xticklabels(), visible=False)

    axs.hist(np.log10(s_cm), bins=80, range=(-5, 1), density=True,
             orientation="horizontal", color="crimson",
             histtype="step", lw=1.4)
    axs.set_xlabel("PDF")
    plt.setp(axs.get_yticklabels(), visible=False)

    axu.set_title(rf"MC single-cell exit statistics, $N$={args.n}, "
                  rf"$\tilde W$={args.W:g}, $\tilde H$={args.H:g} mfp, "
                  rf"$\sigma_s$={args.sigs:g} cm$^{{-1}}$", fontsize=9)
    out = ROOT / "figures" / "fig2b_singlecell_mc.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)
    print(f"backscatter fraction (u<0): {(u < 0).mean():.3f}; "
          f"median path {np.median(s_cm):.3g} cm; "
          f"mean scatters {r['k'].mean():.2f}")


if __name__ == "__main__":
    main()
