#!/usr/bin/env python3
"""Inspect the generated lattice single-cell dataset.

Six panels:
  1. conditioning coverage on the entry half-disk
  2. exit-angle distribution vs cell optical size W
  3. path-length distribution vs W
  4. transmitted / reflected / side-exit fractions vs W
  5. mean scattering events vs W, against the <k> = W invariant
  6. exit perimeter density for a few W (which face particles leave by)

Usage: python scripts/inspect_lattice_dataset.py [--data data/lattice_singlecell.npz]
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


def face_of(p, W):
    """0 bottom, 1 right (transmitted), 2 top, 3 left (reflected)."""
    f = np.zeros(len(p), dtype=np.int8)
    f[(p >= W) & (p < 2 * W)] = 1
    f[(p >= 2 * W) & (p < 3 * W)] = 2
    f[p >= 3 * W] = 3
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lattice_singlecell.npz")
    args = ap.parse_args()

    d = np.load(ROOT / args.data)
    W, oxi, oyi = d["W"], d["oxi"], d["oyi"]
    p, s, k, oxo = d["p"], d["s"], d["k"], d["oxo"]
    ws = np.unique(W)
    print(f"{len(W)} records, {len(ws)} optical sizes "
          f"[{ws.min():g}, {ws.max():g}] mfp")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # 1. conditioning coverage
    sub = np.random.default_rng(0).choice(len(W), 40000, replace=False)
    ax[0][0].scatter(oxi[sub], oyi[sub], s=1, alpha=0.12, color="navy",
                     edgecolors="none")
    th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    ax[0][0].plot(np.cos(th), np.sin(th), "r-", lw=1)
    ax[0][0].set_aspect("equal")
    ax[0][0].set_xlabel(r"$\Omega_x$ in"); ax[0][0].set_ylabel(r"$\Omega_y$ in")
    ax[0][0].set_title("Entry conditioning coverage (half-disk)")

    show = [w for w in ws if w in (0.1, 0.5, 2.0, 5.0, 20.0)] or list(ws[::4])

    # 2. exit angle
    for w in show:
        ax[0][1].hist(oxo[W == w], bins=80, range=(-1, 1), density=True,
                      histtype="step", lw=1.3, label=f"W={w:g}")
    ax[0][1].set_xlabel(r"$\Omega_x$ exit"); ax[0][1].set_ylabel("PDF")
    ax[0][1].legend(fontsize=7); ax[0][1].set_title("Exit angle vs W")

    # 3. path length
    for w in show:
        v = s[W == w]
        ax[0][2].hist(np.log10(np.maximum(v, 1e-8)), bins=80, density=True,
                      histtype="step", lw=1.3, label=f"W={w:g}")
    ax[0][2].set_xlabel(r"$\log_{10}$ path length (mfp)")
    ax[0][2].set_ylabel("PDF")
    ax[0][2].legend(fontsize=7); ax[0][2].set_title("Path length vs W")

    # 4. exit face fractions
    fr = np.zeros((len(ws), 4))
    for i, w in enumerate(ws):
        m = W == w
        f = face_of(p[m], w)
        fr[i] = [(f == j).mean() for j in range(4)]
    for j, lab in enumerate(["bottom", "right (transmitted)", "top",
                             "left (reflected)"]):
        ax[1][0].semilogx(ws, fr[:, j], "o-", ms=3, label=lab)
    ax[1][0].set_xlabel("W (mfp)"); ax[1][0].set_ylabel("fraction")
    ax[1][0].legend(fontsize=7); ax[1][0].set_title("Exit face vs W")

    # 5. mean scatters vs the <k> = W invariant
    mk = np.array([k[W == w].mean() for w in ws])
    ax[1][1].loglog(ws, mk, "o-", ms=4, label=r"dataset $\langle k\rangle$")
    ax[1][1].loglog(ws, ws, "k--", lw=1, label=r"$\langle k\rangle = W$")
    ax[1][1].set_xlabel("W (mfp)")
    ax[1][1].set_ylabel("mean scattering events")
    ax[1][1].legend(fontsize=8)
    ax[1][1].set_title("Cost invariant (uniform-disk entry)")

    # 6. exit perimeter density, normalised to one lap
    for w in show:
        m = W == w
        ax[1][2].hist(p[m] / (4 * w), bins=120, range=(0, 1), density=True,
                      histtype="step", lw=1.3, label=f"W={w:g}")
    for b in (0.25, 0.5, 0.75):
        ax[1][2].axvline(b, color="0.7", lw=0.8, ls=":")
    ax[1][2].set_xlabel("perimeter coordinate / 4W  "
                        "(bottom | right | top | left)")
    ax[1][2].set_ylabel("PDF")
    ax[1][2].legend(fontsize=7); ax[1][2].set_title("Exit location vs W")

    fig.suptitle("Lattice single-cell transmission dataset")
    fig.tight_layout()
    out = ROOT / "figures" / "lattice_dataset.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)

    print("\n   W      <k>     transmitted  reflected   side")
    for i, w in enumerate(ws):
        print("  %7.3f %8.3f %10.3f %10.3f %7.3f"
              % (w, mk[i], fr[i][1], fr[i][3], fr[i][0] + fr[i][2]))


if __name__ == "__main__":
    main()
