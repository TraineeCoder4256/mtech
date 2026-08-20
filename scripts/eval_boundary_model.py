#!/usr/bin/env python3
"""Evaluate a trained boundary model against held-out Monte Carlo.

Uses the OTHER dataset variant (192 entry conditions x 1024 samples,
different RNG seed) as the reference: its entry conditions were drawn
independently of the training set's, so every evaluated condition is one
the model has never seen -- this tests conditional generalisation, not
memorisation.

For each evaluated W:
  * aggregate over all its entry conditions: overlay model vs MC marginals
    of exit angle, log10 path length, and normalised perimeter coordinate;
  * per-condition Wasserstein-1 distances on those marginals (model 1024
    samples vs MC 1024 samples), reported as median over conditions, with
    a same-size MC-vs-MC split as the statistical floor;
  * exit-face fractions.

The model runs with uncollided="off" and is compared against k>0 MC rows:
the analytic branch is exact by construction, so the interesting question
is whether the *learned* part matches the collided distribution.

Usage:
  python scripts/eval_boundary_model.py [--ckpt models/boundary_v1]
      [--data data/lattice_singlecell.npz] [--w 0.5 2.5 10]
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmc import VelocityField, GMCBoundarySampler  # noqa: E402
from gmc.data import load_normalizers  # noqa: E402


def w1(a, b):
    """1-d Wasserstein distance between empirical samples."""
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    qa = np.quantile(a, np.linspace(0, 1, 256))
    qb = np.quantile(b, np.linspace(0, 1, 256))
    return float(np.mean(np.abs(qa - qb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/boundary_v1")
    ap.add_argument("--data", default="data/lattice_singlecell.npz")
    ap.add_argument("--w", type=float, nargs="+", default=[0.5, 2.5, 10.0])
    ap.add_argument("--ode-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ckpt_dir = ROOT / args.ckpt
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu",
                       weights_only=True)
    cfg = state["config"]
    model = VelocityField(x_dim=cfg["x_dim"], c_dim=cfg["c_dim"],
                          width=cfg["width"], depth=cfg["depth"])
    model.load_state_dict(state["ema"])  # sample from the EMA weights
    ynorm, cnorm, _ = load_normalizers(ckpt_dir / "normalizers.json")
    sampler = GMCBoundarySampler(model, ynorm, cnorm,
                                 ode_steps=args.ode_steps,
                                 s_param=cfg.get("s_param", "logW"))
    print(f"checkpoint {args.ckpt}: s_param = {sampler.s_param}")

    d = np.load(ROOT / args.data)
    W, y0 = d["W"], d["y0"]
    oxi, oyi = d["oxi"], d["oyi"]
    p, s, k = d["p"], d["s"], d["k"]
    oxo = d["oxo"]

    fig, axes = plt.subplots(len(args.w), 3,
                             figsize=(13, 3.6 * len(args.w)), squeeze=False)
    print(f"{'W':>6} {'metric':>10} {'model-vs-MC':>12} {'MC-vs-MC floor':>15}")

    for row, wsel in enumerate(args.w):
        mw = np.isclose(W, wsel) & (k > 0)          # collided reference
        conds = np.unique(np.stack([y0[mw], oxi[mw], oyi[mw]]), axis=1)
        n_cond = conds.shape[1]

        # --- generate: same number of model samples per condition --------
        per = max(1, int(mw.sum()) // n_cond)
        cw = np.repeat(conds, per, axis=1)
        out = sampler.sample(np.full(cw.shape[1], wsel),
                             np.full(cw.shape[1], wsel),
                             cw[0], cw[1], cw[2],
                             seed=args.seed, uncollided="off")

        # --- aggregate marginals ----------------------------------------
        panels = [
            ("exit angle $\\Omega_x$", oxo[mw], out["dir"][:, 0],
             np.linspace(-1, 1, 81)),
            ("$\\log_{10} s$ (mfp)", np.log10(s[mw]),
             np.log10(out["s"]), 60),
            ("perimeter $p/4W$", p[mw] / (4 * wsel),
             out["p"] / (4 * wsel), np.linspace(0, 1, 81)),
        ]
        for col, (lab, ref, gen, bins) in enumerate(panels):
            ax = axes[row][col]
            ax.hist(ref, bins=bins, density=True, histtype="step",
                    lw=1.6, color="k", label="MC (k>0)")
            ax.hist(gen, bins=bins, density=True, histtype="step",
                    lw=1.4, color="crimson", label="GMC model")
            ax.set_xlabel(lab)
            if col == 0:
                ax.set_ylabel(f"W = {wsel:g} mfp\nPDF")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

        # --- per-condition W1 vs the MC-split floor ---------------------
        rng = np.random.default_rng(args.seed)
        stats = {"exit angle": [], "log10 s": [], "p/4W": []}
        floor = {kk: [] for kk in stats}
        # Both comparisons MUST use the same two sample sizes, or the
        # "floor" is not a floor: W1 between finite samples shrinks like
        # 1/sqrt(n), so comparing model(1024) vs ref(m) against
        # ref(m/2) vs ref(m/2) flatters the model by ~sqrt(2).
        # Split the reference into halves A and B, generate |A| model
        # samples, and score model-vs-A against B-vs-A.
        for i in rng.choice(n_cond, min(48, n_cond), replace=False):
            y0i, oxii, oyii = conds[:, i]
            mi = mw & np.isclose(y0, y0i) & np.isclose(oxi, oxii)
            ref_u, ref_s, ref_p = oxo[mi], np.log10(s[mi]), p[mi] / (4 * wsel)
            m_ref = len(ref_u)
            half = m_ref // 2
            if half < 32:                     # too few for a stable estimate
                continue
            perm = rng.permutation(m_ref)
            A, B = perm[:half], perm[half:2 * half]
            gi = sampler.sample(np.full(half, wsel), np.full(half, wsel),
                                np.full(half, y0i), np.full(half, oxii),
                                np.full(half, oyii),
                                seed=int(rng.integers(2**31)),
                                uncollided="off")
            stats["exit angle"].append(w1(ref_u[A], gi["dir"][:, 0]))
            stats["log10 s"].append(w1(ref_s[A], np.log10(gi["s"])))
            stats["p/4W"].append(w1(ref_p[A], gi["p"] / (4 * wsel)))
            floor["exit angle"].append(w1(ref_u[A], ref_u[B]))
            floor["log10 s"].append(w1(ref_s[A], ref_s[B]))
            floor["p/4W"].append(w1(ref_p[A], ref_p[B]))
        for kk in stats:
            print(f"{wsel:6g} {kk:>10} {np.nanmedian(stats[kk]):12.4f} "
                  f"{np.nanmedian(floor[kk]):15.4f}")

        # --- exit-face fractions ----------------------------------------
        from gmc.sampler import perimeter_decode
        _, _, face_ref = perimeter_decode(p[mw], np.full(mw.sum(), wsel),
                                          np.full(mw.sum(), wsel))
        fr_ref = np.bincount(face_ref, minlength=4) / mw.sum()
        fr_gen = np.bincount(out["face"], minlength=4) / len(out["face"])
        names = ["bottom", "right", "top", "left"]
        print(f"{wsel:6g} {'faces':>10} " +
              "  ".join(f"{names[j]} {fr_gen[j]:.3f}/{fr_ref[j]:.3f}"
                        for j in range(4)) + "   (model/MC)")

    fig.suptitle("Boundary GMC model vs held-out MC "
                 "(unseen entry conditions, collided part)")
    fig.tight_layout()
    out_png = ROOT / "figures" / "gmc_boundary_eval.png"
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
