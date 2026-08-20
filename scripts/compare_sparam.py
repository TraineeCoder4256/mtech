#!/usr/bin/env python3
"""Did making the straight-line bound structural remove the clamp spike?

The whole-joint classifier test (scripts/joint_whole.py) found that the
first boundary model had to push 4.22% of its sampled path lengths up to
s_min(p), the straight-line distance from the entry point to the exit point
it had just generated.  That clamp puts a Dirac spike at s = s_min which
the Monte Carlo reference has no counterpart for, so a classifier can find
it even though every marginal looks right.

boundary_v2 re-parameterises the sixth target as u = log(s / s_min(p)), so
the decoder reconstructs s = s_min(p) exp(u) and the bound holds for any
u >= 0.  This script measures whether that worked, on held-out entry
conditions, for both checkpoints at once:

  * clamp fraction              -- how often the decoder still has to push
  * P(log10 s/s_min)            -- the spike, if there is one, is visible
                                   as a delta at 0; this is the figure
  * C2ST held-out accuracy      -- can a classifier separate the 5-D joint
                                   from MC?  50% = no
  * sliced Wasserstein ratio    -- median over random 5-D directions,
                                   against the MC-vs-MC floor

Usage: python scripts/compare_sparam.py [--w 1.0 4.0] [--ode-steps 5]
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
from gmc.geometry import s_min_of                     # noqa: E402
from gmc_end_to_end import load_sampler               # noqa: E402
from joint_whole import c2st, sliced_test             # noqa: E402

COLORS = {"logW": "#C1443C", "detour": "#2F6F8F"}


def held_out_conditions(d, w):
    """Entry conditions present at this optical size, with their row masks."""
    mw = np.isclose(d["W"], w) & (d["k"] > 0)
    conds = np.unique(np.stack([d["y0"][mw], d["oxi"][mw], d["oyi"][mw]]),
                      axis=1)
    out = []
    for i in range(conds.shape[1]):
        y0i, oxii, oyii = conds[:, i]
        mi = mw & np.isclose(d["y0"], y0i) & np.isclose(d["oxi"], oxii)
        if int(mi.sum()) >= 4:
            out.append((y0i, oxii, oyii, mi))
    return out


def clouds(sampler, d, w, conds, seed):
    """Matched MC / model clouds plus the model's mean clamp fraction."""
    rng = np.random.default_rng(seed)
    R, G, clamp, wt = [], [], [], []
    for y0i, oxii, oyii, mi in conds:
        k = int(mi.sum())
        R.append(np.stack([d["p"][mi] / (4 * w), d["oxo"][mi], d["oyo"][mi],
                           d["ozo"][mi], np.log10(d["s"][mi])], axis=1))
        g = sampler.sample(np.full(k, w), np.full(k, w), np.full(k, y0i),
                           np.full(k, oxii), np.full(k, oyii),
                           seed=int(rng.integers(2**31)), uncollided="off")
        G.append(np.stack([g["p"] / (4 * w), g["dir"][:, 0], g["dir"][:, 1],
                           g["dir"][:, 2], np.log10(g["s"])], axis=1))
        clamp.append(sampler.last_clamp_frac)
        wt.append(k)
    wt = np.asarray(wt, float)
    return (np.concatenate(R), np.concatenate(G),
            float(np.average(clamp, weights=wt)))


def detour_of(cloud, w, y0_all):
    """log10(s / s_min) for a cloud stored as [p/4W, Ox, Oy, Oz, log10 s]."""
    p = cloud[:, 0] * 4 * w
    s = 10.0 ** cloud[:, 4]
    n = np.full(len(p), w)
    return np.log10(s / np.maximum(s_min_of(p, n, n, y0_all), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--ckpts", nargs="+",
                    default=["models/boundary_v1", "models/boundary_v2"])
    ap.add_argument("--data", default="data/lattice_singlecell.npz")
    ap.add_argument("--ode-steps", type=int, default=5)
    ap.add_argument("--n-dir", type=int, default=300)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    samplers = {}
    for c in args.ckpts:
        s = load_sampler(ROOT / c, ode_steps=args.ode_steps,
                         device=args.device)
        samplers[s.s_param] = (c, s)
        print(f"{c}: s_param = {s.s_param}")
    d = np.load(ROOT / args.data)

    fig, axes = plt.subplots(1, len(args.w), figsize=(6.2 * len(args.w), 4.4),
                             squeeze=False)
    rows = []

    for col, w in enumerate(args.w):
        conds = held_out_conditions(d, w)
        # entry ordinate per reference row, needed to recompute s_min
        y0_all = np.concatenate([np.full(int(mi.sum()), y0i)
                                 for y0i, _, _, mi in conds])
        ax = axes[0][col]
        first = True
        for sp, (name, sam) in samplers.items():
            R, G, cf = clouds(sam, d, w, conds, args.seed)
            mu, sd = R.mean(0), R.std(0) + 1e-9
            Rs, Gs = (R - mu) / sd, (G - mu) / sd
            rng = np.random.default_rng(args.seed)
            model, floor = sliced_test(Rs, Gs, args.n_dir, rng)
            acc, se, nte = c2st(Rs, Gs, rng)
            dG = detour_of(G, w, y0_all)
            dR = detour_of(R, w, y0_all)
            # mass within 0.01 dex of the straight-line bound
            spike_G = float((dG < 0.01).mean())
            spike_R = float((dR < 0.01).mean())
            rows.append((w, sp, name, cf, spike_R, spike_G,
                         float(np.median(model / floor)), acc, se, nte))

            bins = np.linspace(-0.05, 1.6, 130)
            if first:
                hR, e = np.histogram(dR, bins=bins, density=True)
                c_ = .5 * (e[:-1] + e[1:])
                ax.fill_between(c_, hR, color="0.80", step="mid")
                ax.plot(c_, hR, "k", lw=1.4, drawstyle="steps-mid",
                        label=f"MC  ({spike_R*100:.2f}% at bound)")
                first = False
            hG, e = np.histogram(dG, bins=bins, density=True)
            ax.plot(.5 * (e[:-1] + e[1:]), hG, color=COLORS[sp], lw=1.4,
                    drawstyle="steps-mid",
                    label=f"GMC {sp}  ({spike_G*100:.2f}% at bound, "
                          f"clamp {cf*100:.2f}%)")

        ax.axvline(0.0, color="0.4", ls=":", lw=1.1)
        ax.set_xlabel(r"$\log_{10}(s / s_{\min}(p))$   "
                      r"— 0 is the straight-line bound")
        ax.set_ylabel("density")
        ax.set_title(rf"$\tilde W$ = {w:g} mfp", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Path length relative to the straight-line bound it cannot "
                 "undercut\nthe clamp spike at 0 is what the classifier "
                 "two-sample test was detecting", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = ROOT / "figures" / "sparam_comparison.png"
    fig.savefig(out, dpi=145)
    print(f"\nwrote {out}\n")

    hdr = (f"{'W~':>5} {'s_param':>8} {'clamped':>9} {'MC at bnd':>10} "
           f"{'GMC at bnd':>11} {'sliced W1/floor':>16} {'C2ST':>16}")
    print(hdr)
    print("-" * len(hdr))
    for w, sp, name, cf, sR, sG, ratio, acc, se, nte in rows:
        z = (acc - 0.5) / se
        print(f"{w:>5g} {sp:>8} {cf*100:>8.2f}% {sR*100:>9.2f}% "
              f"{sG*100:>10.2f}% {ratio:>16.2f} "
              f"{acc*100:>7.2f}% ({z:+.1f}s)")
    print("\n'at bnd' = fraction of samples within 0.01 dex of s_min(p).")
    print("C2ST: 50% means a classifier trained to separate the two 5-D "
          "clouds cannot; sigma is distance from chance.")


if __name__ == "__main__":
    main()
