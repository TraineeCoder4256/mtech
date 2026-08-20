#!/usr/bin/env python3
"""Test the exit-state joint distribution AS A WHOLE, not slice by slice.

The joint is 5-D -- (p, Omega_x, Omega_y, Omega_z, log s) -- so it cannot be
drawn directly; corner plots and conditionals are projections.  These two
tests operate on all five dimensions at once.

1. SLICED / RANDOM-PROJECTION TEST.  By the Cramer-Wold theorem two
   distributions are identical if and only if *every* 1-D projection of them
   is identical.  So we draw many random unit directions, project both
   clouds onto each, and measure the 1-D Wasserstein distance.  Comparing
   against the same measurement on two halves of the MC sample gives a per
   direction floor.  This is a genuine whole-joint test whose result can
   still be plotted, because the output is a distribution over directions.

2. CLASSIFIER TWO-SAMPLE TEST (C2ST).  Train a classifier to tell MC
   samples from model samples.  If the two distributions are identical the
   best achievable held-out accuracy is 50%: chance.  Any systematic
   difference anywhere in the 5-D density -- in a marginal, a correlation,
   a tail, a corner -- is something the classifier can exploit, so a score
   at chance is strong evidence of a match.  The number is directly
   interpretable: "a network trained specifically to separate them cannot."

Usage: python scripts/joint_whole.py [--w 1.0] [--ode-steps 5]
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gmc_end_to_end import load_sampler          # noqa: E402


def build_clouds(sampler, data, w, seed):
    """Matched MC and model clouds at the same held-out entry conditions."""
    d = np.load(data)
    mw = np.isclose(d["W"], w) & (d["k"] > 0)
    conds = np.unique(np.stack([d["y0"][mw], d["oxi"][mw], d["oyi"][mw]]),
                      axis=1)
    rng = np.random.default_rng(seed)
    R, G = [], []
    for i in range(conds.shape[1]):
        y0i, oxii, oyii = conds[:, i]
        mi = mw & np.isclose(d["y0"], y0i) & np.isclose(d["oxi"], oxii)
        k = int(mi.sum())
        if k < 4:
            continue
        R.append(np.stack([d["p"][mi] / (4 * w), d["oxo"][mi], d["oyo"][mi],
                           d["ozo"][mi], np.log10(d["s"][mi])], axis=1))
        g = sampler.sample(np.full(k, w), np.full(k, w), np.full(k, y0i),
                           np.full(k, oxii), np.full(k, oyii),
                           seed=int(rng.integers(2**31)), uncollided="off")
        G.append(np.stack([g["p"] / (4 * w), g["dir"][:, 0], g["dir"][:, 1],
                           g["dir"][:, 2], np.log10(g["s"])], axis=1))
    return np.concatenate(R), np.concatenate(G)


def w1_1d(a, b, q=400):
    qa = np.quantile(a, np.linspace(0, 1, q))
    qb = np.quantile(b, np.linspace(0, 1, q))
    return float(np.mean(np.abs(qa - qb)))


def sliced_test(Rs, Gs, n_dir, rng):
    """1-D Wasserstein along many random directions: model and MC-MC floor."""
    dim = Rs.shape[1]
    V = rng.normal(size=(n_dir, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    h = rng.permutation(len(Rs))
    half = len(h) // 2
    A, B = h[:half], h[half:2 * half]
    model, floor = [], []
    for v in V:
        pr, pg = Rs @ v, Gs @ v
        model.append(w1_1d(pr[A], pg[A]))
        floor.append(w1_1d(pr[A], pr[B]))
    return np.array(model), np.array(floor)


def c2st(Rs, Gs, rng, epochs=40, hidden=128):
    """Classifier two-sample test; returns held-out accuracy and its s.e."""
    n = min(len(Rs), len(Gs))
    idx_r = rng.choice(len(Rs), n, replace=False)
    idx_g = rng.choice(len(Gs), n, replace=False)
    X = np.concatenate([Rs[idx_r], Gs[idx_g]]).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(X))
    Xtr, ytr = torch.from_numpy(X[:cut]), torch.from_numpy(y[:cut])
    Xte, yte = torch.from_numpy(X[cut:]), torch.from_numpy(y[cut:])

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    bs = 4096
    for ep in range(epochs):
        p = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = p[i:i + bs]
            opt.zero_grad()
            lossf(net(Xtr[b]).squeeze(1), ytr[b]).backward()
            opt.step()
    with torch.no_grad():
        pred = (net(Xte).squeeze(1) > 0).float()
        acc = (pred == yte).float().mean().item()
    se = np.sqrt(acc * (1 - acc) / len(yte))
    return acc, se, len(yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--ckpt", default="models/boundary_v1")
    ap.add_argument("--data", default="data/lattice_singlecell.npz")
    ap.add_argument("--ode-steps", type=int, default=5)
    ap.add_argument("--n-dir", type=int, default=400)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    sampler = load_sampler(ROOT / args.ckpt, ode_steps=args.ode_steps)
    R, G = build_clouds(sampler, ROOT / args.data, args.w, args.seed)
    mu, sd = R.mean(0), R.std(0) + 1e-9
    Rs, Gs = (R - mu) / sd, (G - mu) / sd
    rng = np.random.default_rng(args.seed)
    print(f"W = {args.w} mfp : {len(R):,} matched samples, "
          f"{R.shape[1]}-D joint\n")

    print("1. SLICED TEST (Cramer-Wold: equal iff all 1-D projections equal)")
    model, floor = sliced_test(Rs, Gs, args.n_dir, rng)
    ratio = model / floor
    print(f"   {args.n_dir} random directions through the 5-D joint")
    print(f"   model W1 : median {np.median(model):.4f}")
    print(f"   MC floor : median {np.median(floor):.4f}")
    print(f"   ratio    : median {np.median(ratio):.2f}, "
          f"90th pct {np.percentile(ratio, 90):.2f}, "
          f"max {ratio.max():.2f}")
    worst = np.argmax(ratio)
    print(f"   worst direction ratio {ratio[worst]:.2f}\n")

    print("2. CLASSIFIER TWO-SAMPLE TEST (50% = indistinguishable)")
    acc, se, nte = c2st(Rs, Gs, rng)
    z = (acc - 0.5) / se
    print(f"   held-out accuracy {acc*100:.2f}% +/- {se*100:.2f}% "
          f"on {nte:,} samples")
    print(f"   {z:+.1f} sigma from chance")
    print(f"   verdict: {'indistinguishable' if abs(z) < 3 else 'separable'}"
          f" at the 3-sigma level")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
    ax[0].hist(ratio, bins=44, color="#4C3A8F", alpha=.85)
    ax[0].axvline(1.0, color="k", ls="--", lw=1.2, label="MC-vs-MC floor")
    ax[0].set_xlabel("model W1 / MC floor, per random direction")
    ax[0].set_ylabel("directions")
    ax[0].set_title(f"Sliced test over {args.n_dir} random directions\n"
                    f"median {np.median(ratio):.2f}, max {ratio.max():.2f}",
                    fontsize=9.5)
    ax[0].legend(fontsize=8)

    ax[1].scatter(floor, model, s=9, alpha=.5, color="#4C3A8F")
    lim = [0, max(model.max(), floor.max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1.2, label="model = floor")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("MC-vs-MC W1 (floor)")
    ax[1].set_ylabel("model-vs-MC W1")
    ax[1].set_title(f"per-direction agreement\nC2ST accuracy "
                    f"{acc*100:.2f}% (chance = 50%)", fontsize=9.5)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Whole-joint tests, $\\tilde W$ = {args.w:g} mfp, "
                 f"{len(R):,} matched samples (all 5 dimensions at once)",
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "figures" / f"joint_whole_W{args.w:g}.png"
    fig.savefig(out, dpi=145)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
