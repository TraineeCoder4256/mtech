#!/usr/bin/env python3
"""Reproduce the standard-MC side of Fig. 3 of arXiv:2512.13965v1.

For each benchmark (lattice, hohlraum) this produces the three panels:
  (a/d) geometry, (b/e) log10 scalar flux heatmap, (c/f) lineouts,
with our MC overlaid on the MC/GMC curves digitized from the paper.

Usage:  python scripts/fig3_benchmarks.py [--problem lattice|hohlraum|both]
                                          [--n 1000000] [--seed 1]
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import run_transport, lattice_problem, hohlraum_problem  # noqa: E402
from mc2d.problems import LATTICE_ABSORBERS  # noqa: E402


def draw_lattice_geometry(ax):
    ax.set_xlim(0, 7); ax.set_ylim(0, 7); ax.set_aspect("equal")
    for (x0, y0) in LATTICE_ABSORBERS:
        ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor="navy"))
    ax.add_patch(Rectangle((3, 3), 1, 1, facecolor="none",
                           edgecolor="crimson", hatch="xxxx", lw=1.2))
    ax.set_title("Geometry (cm)")


def draw_hohlraum_geometry(ax):
    L, t = 1.3, 0.05
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    for box, color, hatch in [
        ((0, 0, L, t), "black", None), ((0, L - t, L, L), "black", None),
        ((L - t, 0, L, L), "black", None),
        ((0.0, 0.25, 0.05, 1.05), "crimson", "////"),
        ((0.45, 0.25, 0.85, 1.05), "seagreen", None),
        ((0.50, 0.30, 0.85, 1.00), "navy", None),
    ]:
        x0, y0, x1, y1 = box
        fc = "none" if hatch else color
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                               edgecolor=color, hatch=hatch, lw=1.0))
    ax.set_title("Geometry (cm)")


def lineout(phi, prob):
    """Extract the horizontal/vertical lineouts at the marker positions."""
    n = prob["nx"]
    d = prob["Lx"] / n
    centers = (np.arange(n) + 0.5) * d
    yq, xq = prob["lineout_y"], prob["lineout_x"]

    def cut(coord, axis):
        k = coord / d
        if abs(k - round(k)) < 1e-9:      # marker on a cell boundary:
            i = int(round(k))             # average the two adjacent rows
            sl = (phi[i - 1] + phi[i]) / 2 if axis == 0 else \
                 (phi[:, i - 1] + phi[:, i]) / 2
        else:
            i = int(k)
            sl = phi[i] if axis == 0 else phi[:, i]
        return sl

    return centers, cut(yq, 0), cut(xq, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="both",
                    choices=["lattice", "hohlraum", "both"])
    ap.add_argument("--n", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # Digitized paper curves are optional: without them we still produce our
    # own MC lineouts, just with no paper overlay to compare against.
    ref_path = ROOT / "reference" / "paper_fig3_lineouts.npz"
    ref = np.load(ref_path) if ref_path.exists() else None
    if ref is None:
        print(f"note: {ref_path.name} not found -- plotting our MC only, "
              f"no paper comparison")
    todo = ["lattice", "hohlraum"] if args.problem == "both" else [args.problem]

    for name in todo:
        prob = lattice_problem() if name == "lattice" else hohlraum_problem()
        t0 = time.time()
        phi = run_transport(prob, int(args.n), seed=args.seed)
        print(f"{name}: N={int(args.n):.2e} particles in {time.time()-t0:.1f} s"
              f"  (max log10 phi = {np.log10(phi.max()):.2f})")
        np.savez(ROOT / "data" / f"fig3_{name}_phi.npz", phi=phi,
                 n=args.n, seed=args.seed)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        # (a/d) geometry
        (draw_lattice_geometry if name == "lattice"
         else draw_hohlraum_geometry)(axes[0])

        # (b/e) flux map, same scale as the paper: log10 phi in [-6, 0]
        L = prob["Lx"]
        im = axes[1].imshow(np.log10(np.maximum(phi, 1e-30)),
                            origin="lower", extent=[0, L, 0, L],
                            cmap="plasma", vmin=-6, vmax=0)
        axes[1].axhline(prob["lineout_y"], color="w", lw=1)
        axes[1].axvline(prob["lineout_x"], color="w", lw=1)
        fig.colorbar(im, ax=axes[1], label=r"$\log_{10}(\phi)$")
        axes[1].set_title(rf"$\log_{{10}}$(scalar flux), MC, N={int(args.n):.0e}")

        # (c/f) lineouts vs the curves digitized from the paper
        c, hor, ver = lineout(phi, prob)
        pfx = "c" if name == "lattice" else "f"
        for key, style, lab in [
            (f"{pfx}_mc_h", dict(color="0.55", ls="--", lw=1.2), "paper MC (horiz)"),
            (f"{pfx}_mc_v", dict(color="0.75", ls="--", lw=1.2), "paper MC (vert)"),
        ]:
            if ref is None or key not in ref:
                continue
            d = ref[key]; d = d[d[:, 0].argsort()]
            axes[2].plot(d[:, 0], d[:, 1], **style, label=lab)
        axes[2].plot(c, np.log10(np.maximum(hor, 1e-30)), color="crimson",
                     lw=1.4, label=f"ours (horiz, y={prob['lineout_y']:g})")
        axes[2].plot(c, np.log10(np.maximum(ver, 1e-30)), color="royalblue",
                     lw=1.4, label=f"ours (vert, x={prob['lineout_x']:g})")
        axes[2].set_xlabel("position (cm)")
        axes[2].set_ylabel(r"$\log_{10}(\phi)$")
        axes[2].legend(fontsize=8)
        axes[2].set_title("Lineout comparison")

        fig.suptitle(f"{name} benchmark (arXiv:2512.13965 Fig. 3, MC part)")
        fig.tight_layout()
        out = ROOT / "figures" / f"fig3_{name}.png"
        fig.savefig(out, dpi=170)
        print("wrote", out)


if __name__ == "__main__":
    main()
