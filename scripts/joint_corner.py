#!/usr/bin/env python3
"""Corner plot of the JOINT exit-state distribution: model vs Monte Carlo.

The marginal figure (gmc_boundary_eval.png) shows each exit variable on its
own.  This shows the pairwise joint densities, which is where a generative
model can fail invisibly: every marginal can match while the correlations
are wrong.

Diagonal   : 1-D marginals, MC (filled) vs GMC (line).
Lower tri  : 2-D densities, MC as filled contours, GMC as overlaid lines.
Upper tri  : signed difference of the 2-D densities, GMC minus MC.

Both clouds are generated from the SAME held-out entry conditions, so any
difference is the model, not a difference in conditioning.

Usage:
  python scripts/joint_corner.py [--w 1.0] [--ckpt models/boundary_v1]
      [--ode-steps 5] [--data data/lattice_singlecell.npz]
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
sys.path.insert(0, str(ROOT / "scripts"))
from gmc_end_to_end import load_sampler                    # noqa: E402

LABELS = [r"$p/4\tilde W$", r"$\Omega_x$", r"$\Omega_y$",
          r"$\Omega_z$", r"$\log_{10} s$"]
RANGES = [(0, 1), (-1, 1), (-1, 1), (-1, 1), None]   # None = data-driven


def physical_vec(p, W, ox, oy, oz, s):
    return np.stack([p / (4 * W), ox, oy, oz, np.log10(s)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--ckpt", default="models/boundary_v1")
    ap.add_argument("--data", default="data/lattice_singlecell.npz")
    ap.add_argument("--ode-steps", type=int, default=5)
    ap.add_argument("--bins", type=int, default=44)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    sampler = load_sampler(ROOT / args.ckpt, ode_steps=args.ode_steps,
                           device=args.device)
    d = np.load(ROOT / args.data)
    w = args.w
    mw = np.isclose(d["W"], w) & (d["k"] > 0)
    print(f"W = {w} mfp : {mw.sum():,} held-out collided MC samples")

    # generate one model sample per reference sample, at the same conditions
    conds = np.unique(np.stack([d["y0"][mw], d["oxi"][mw], d["oyi"][mw]]),
                      axis=1)
    rng = np.random.default_rng(args.seed)
    R, G = [], []
    for i in range(conds.shape[1]):
        y0i, oxii, oyii = conds[:, i]
        mi = mw & np.isclose(d["y0"], y0i) & np.isclose(d["oxi"], oxii)
        k = int(mi.sum())
        if k < 4:
            continue
        R.append(physical_vec(d["p"][mi], w, d["oxo"][mi], d["oyo"][mi],
                              d["ozo"][mi], d["s"][mi]))
        g = sampler.sample(np.full(k, w), np.full(k, w), np.full(k, y0i),
                           np.full(k, oxii), np.full(k, oyii),
                           seed=int(rng.integers(2**31)), uncollided="off")
        G.append(physical_vec(g["p"], w, g["dir"][:, 0], g["dir"][:, 1],
                              g["dir"][:, 2], g["s"]))
    R, G = np.concatenate(R), np.concatenate(G)
    print(f"matched clouds: {len(R):,} MC vs {len(G):,} model samples "
          f"over {conds.shape[1]} conditions")

    # shared ranges so the two densities are directly comparable
    rng_ = []
    for j, r in enumerate(RANGES):
        if r is None:
            lo = np.percentile(np.concatenate([R[:, j], G[:, j]]), 0.2)
            hi = np.percentile(np.concatenate([R[:, j], G[:, j]]), 99.8)
            rng_.append((lo, hi))
        else:
            rng_.append(r)

    nv = len(LABELS)
    fig, axes = plt.subplots(nv, nv, figsize=(13.5, 12.5))
    b = args.bins

    for i in range(nv):
        for j in range(nv):
            ax = axes[i][j]
            if i == j:
                for arr, col, fill, lab in [
                        (R, "k", True, "MC"), (G, "crimson", False, "GMC")]:
                    h, e = np.histogram(arr[:, i], bins=b, range=rng_[i],
                                        density=True)
                    c = 0.5 * (e[:-1] + e[1:])
                    if fill:
                        ax.fill_between(c, h, color="0.75", step="mid")
                        ax.plot(c, h, color=col, lw=1.1, label=lab)
                    else:
                        ax.plot(c, h, color=col, lw=1.3, label=lab)
                ax.set_xlim(*rng_[i])
                ax.set_yticks([])
                if i == 0:
                    ax.legend(fontsize=7, loc="upper right")
            elif i > j:                                  # lower: overlay
                hR, xe, ye = np.histogram2d(R[:, j], R[:, i], bins=b,
                                            range=[rng_[j], rng_[i]],
                                            density=True)
                hG, _, _ = np.histogram2d(G[:, j], G[:, i], bins=b,
                                          range=[rng_[j], rng_[i]],
                                          density=True)
                X = 0.5 * (xe[:-1] + xe[1:])
                Y = 0.5 * (ye[:-1] + ye[1:])
                ax.contourf(X, Y, hR.T, levels=7, cmap="Greys")
                ax.contour(X, Y, hG.T, levels=7, colors="crimson",
                           linewidths=0.75)
            else:                                        # upper: difference
                hR, xe, ye = np.histogram2d(R[:, j], R[:, i], bins=b,
                                            range=[rng_[j], rng_[i]],
                                            density=True)
                hG, _, _ = np.histogram2d(G[:, j], G[:, i], bins=b,
                                          range=[rng_[j], rng_[i]],
                                          density=True)
                dif = (hG - hR).T
                v = np.percentile(np.abs(dif), 99.5) or 1.0
                ax.imshow(dif, origin="lower", cmap="RdBu_r",
                          vmin=-v, vmax=v, aspect="auto",
                          extent=[rng_[j][0], rng_[j][1],
                                  rng_[i][0], rng_[i][1]])
            if i == nv - 1:
                ax.set_xlabel(LABELS[j], fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[i], fontsize=10)
            elif j != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Joint exit-state distribution, $\\tilde W$ = {w:g} mfp, "
        f"{len(R):,} matched samples at held-out entry conditions\n"
        "lower: MC filled / GMC contours   ·   upper: GMC $-$ MC   ·   "
        "diagonal: marginals", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = ROOT / "figures" / f"joint_corner_W{w:g}.png"
    fig.savefig(out, dpi=135)
    print("wrote", out)

    # numeric companion: pairwise correlation error
    cR, cG = np.corrcoef(R.T), np.corrcoef(G.T)
    iu = np.triu_indices(nv, 1)
    print(f"\npairwise correlation, MC vs GMC ({len(iu[0])} pairs):")
    for a, bb in zip(*iu):
        print(f"  {LABELS[a]:>14} x {LABELS[bb]:<14} "
              f"{cR[a,bb]:+.3f}  {cG[a,bb]:+.3f}   "
              f"diff {abs(cR[a,bb]-cG[a,bb]):.3f}")
    print(f"max |correlation error| = {np.abs(cR-cG)[iu].max():.4f}")


if __name__ == "__main__":
    main()
