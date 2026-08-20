#!/usr/bin/env python3
"""Conditional PDFs of the exit state: does GMC fit MC *jointly*?

Marginal PDFs can all match while the joint is wrong.  The direct way to
see the joint as readable 1-D curves is to condition: slice on one exit
variable and plot the PDF of another.  If the joint is right, every
conditional overlays; if only the marginals were right, conditionals come
apart.

Rows:
  1. P(log10 s | exit face)        -- couples exit position to path length
  2. P(Omega_x | exit face)        -- couples exit position to exit angle
  3. P(log10 s | Omega_x band)     -- couples exit angle to path length
  4. P(p/4W | Omega_x band)        -- couples exit angle to exit position

Both clouds come from the same held-out entry conditions, so a gap is the
model and not a difference in conditioning.

Usage: python scripts/joint_conditionals.py [--w 1.0] [--ode-steps 5]
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
from gmc_end_to_end import load_sampler          # noqa: E402
from gmc.sampler import perimeter_decode         # noqa: E402

FACES = ["bottom", "right (transmitted)", "top", "left (reflected)"]


def curve(ax, ref, gen, bins, rng_, label, show_legend=False):
    """Overlay two PDFs; annotate with the fraction of samples in the slice."""
    if len(ref) < 40 or len(gen) < 40:
        ax.text(.5, .5, "too few samples", ha="center", va="center",
                fontsize=8, transform=ax.transAxes, color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
        return
    hR, e = np.histogram(ref, bins=bins, range=rng_, density=True)
    hG, _ = np.histogram(gen, bins=bins, range=rng_, density=True)
    c = 0.5 * (e[:-1] + e[1:])
    ax.fill_between(c, hR, color="0.78", step="mid")
    ax.plot(c, hR, color="k", lw=1.3, drawstyle="steps-mid", label="MC")
    ax.plot(c, hG, color="crimson", lw=1.3, drawstyle="steps-mid",
            label="GMC")
    # total-variation distance: 0 = identical, 1 = disjoint
    tv = 0.5 * np.abs(hR - hG).sum() * (e[1] - e[0])
    ax.set_title(f"{label}\nTV = {tv:.3f}", fontsize=8.5)
    ax.set_yticks([])
    if show_legend:
        ax.legend(fontsize=7.5, loc="upper left")
    return tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--ckpt", default="models/boundary_v1")
    ap.add_argument("--data", default="data/lattice_singlecell.npz")
    ap.add_argument("--ode-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    sampler = load_sampler(ROOT / args.ckpt, ode_steps=args.ode_steps,
                           device=args.device)
    d = np.load(ROOT / args.data)
    w = args.w
    mw = np.isclose(d["W"], w) & (d["k"] > 0)

    conds = np.unique(np.stack([d["y0"][mw], d["oxi"][mw], d["oyi"][mw]]),
                      axis=1)
    rng = np.random.default_rng(args.seed)
    Rp, Ru, Rs, Gp, Gu, Gs = [], [], [], [], [], []
    for i in range(conds.shape[1]):
        y0i, oxii, oyii = conds[:, i]
        mi = mw & np.isclose(d["y0"], y0i) & np.isclose(d["oxi"], oxii)
        k = int(mi.sum())
        if k < 4:
            continue
        Rp.append(d["p"][mi]); Ru.append(d["oxo"][mi]); Rs.append(d["s"][mi])
        g = sampler.sample(np.full(k, w), np.full(k, w), np.full(k, y0i),
                           np.full(k, oxii), np.full(k, oyii),
                           seed=int(rng.integers(2**31)), uncollided="off")
        Gp.append(g["p"]); Gu.append(g["dir"][:, 0]); Gs.append(g["s"])
    Rp, Ru, Rs = map(np.concatenate, (Rp, Ru, Rs))
    Gp, Gu, Gs = map(np.concatenate, (Gp, Gu, Gs))
    n = len(Rp)
    print(f"W = {w} mfp : {n:,} matched samples at held-out conditions")

    nn = np.full(n, w)
    _, _, fR = perimeter_decode(Rp.astype(float), nn, nn)
    _, _, fG = perimeter_decode(Gp.astype(float), nn, nn)
    lsR, lsG = np.log10(Rs), np.log10(Gs)
    s_rng = (np.percentile(lsR, 0.3), np.percentile(lsR, 99.7))

    fig, ax = plt.subplots(4, 4, figsize=(15, 12.5))
    tvs = []

    for j in range(4):
        mR, mG = fR == j, fG == j
        tvs.append(curve(ax[0][j], lsR[mR], lsG[mG], 46, s_rng,
                         f"$\\log_{{10}}s$ | exit {FACES[j]}\n"
                         f"({mR.mean()*100:.1f}% of MC)", show_legend=(j == 0)))
        tvs.append(curve(ax[1][j], Ru[mR], Gu[mG], 46, (-1, 1),
                         f"$\\Omega_x$ | exit {FACES[j]}"))

    bands = [(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)]
    for j, (lo, hi) in enumerate(bands):
        mR = (Ru >= lo) & (Ru < hi)
        mG = (Gu >= lo) & (Gu < hi)
        tvs.append(curve(ax[2][j], lsR[mR], lsG[mG], 46, s_rng,
                         f"$\\log_{{10}}s$ | $\\Omega_x\\in$[{lo:g},{hi:g})\n"
                         f"({mR.mean()*100:.1f}% of MC)"))
        tvs.append(curve(ax[3][j], Rp[mR] / (4 * w), Gp[mG] / (4 * w), 46,
                         (0, 1), f"$p/4\\tilde W$ | $\\Omega_x\\in$"
                                 f"[{lo:g},{hi:g})"))

    for a in ax[3]:
        a.set_xlabel("value", fontsize=9)
    tvs = [t for t in tvs if t is not None]
    fig.suptitle(
        f"Conditional PDFs, $\\tilde W$ = {w:g} mfp, {n:,} matched samples "
        f"at held-out entry conditions\n"
        f"every panel is a slice of the JOINT distribution  ·  "
        f"median total-variation distance {np.median(tvs):.3f}, "
        f"max {max(tvs):.3f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = ROOT / "figures" / f"joint_conditionals_W{w:g}.png"
    fig.savefig(out, dpi=135)
    print(f"wrote {out}")
    print(f"total-variation distance over {len(tvs)} conditionals: "
          f"median {np.median(tvs):.4f}, max {max(tvs):.4f}")
    print("(TV = 0 identical, 1 disjoint; ~0.02-0.05 is sampling noise "
          "at these counts)")


if __name__ == "__main__":
    main()
