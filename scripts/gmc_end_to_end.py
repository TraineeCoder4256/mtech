#!/usr/bin/env python3
"""End-to-end test: GMC sampler chained across the full lattice geometry.

Three independent questions, three tests.

1. ACCURACY.  Solve the lattice with the learned sampler in place of the
   in-cell random walk and compare the resulting flux field against the
   Monte Carlo reference, coarsened to the same macro-cell grid.  A
   Monte-Carlo-versus-Monte-Carlo run at the same particle count gives the
   statistical floor, so "how close" has a reference point.

2. JOINT DISTRIBUTION.  Marginal agreement does not imply the joint is
   right, and the transport solve consumes (p, Omega, s) together.  We
   measure the multivariate energy distance

       E = 2/(nm) SUM|x_i - y_j| - 1/n^2 SUM|x_i - x_j| - 1/m^2 SUM|y_i - y_j|

   on the standardised 6-d exit vector, again against an MC-vs-MC floor,
   plus the full 6x6 correlation matrix (which is a pure joint property --
   every marginal can be perfect while these are wrong).

3. SPEED.  The lattice at its published scale has optically *thin* cells,
   where GMC cannot win: 25 network evaluations cost more than one
   scattering event.  We therefore sweep the optical scale of the same
   geometry and locate the crossover, which is the honest way to show what
   the method buys.

Usage:
  python scripts/gmc_end_to_end.py [--n 20000] [--ckpt models/boundary_v1]
      [--scales 1 4 10 20] [--skip-speed]
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import run_transport, sample_single_cell            # noqa: E402
from mc2d.lattice import build_lattice                        # noqa: E402
from gmc import VelocityField, GMCBoundarySampler             # noqa: E402
from gmc.data import load_normalizers                         # noqa: E402
from gmc.device import pick_device, describe                  # noqa: E402
from gmc.transport import (run_gmc_transport, macro_problem,  # noqa: E402
                           coarsen)


# ----------------------------------------------------------------- utils
def load_sampler(ckpt_dir, ode_steps=25, solver="heun", device=None):
    """Rebuild a trained sampler.  device=None/"auto" -> CUDA if present."""
    dev = pick_device(device)
    st = torch.load(ckpt_dir / "model.pt", map_location="cpu",
                    weights_only=True)
    cfg = st["config"]
    m = VelocityField(x_dim=cfg["x_dim"], c_dim=cfg["c_dim"],
                      width=cfg["width"], depth=cfg["depth"])
    m.load_state_dict(st["ema"])
    yn, cn, _ = load_normalizers(ckpt_dir / "normalizers.json")
    # checkpoints written before the detour encoding existed have no
    # s_param key and were all trained with log(s/W~)
    return GMCBoundarySampler(m, yn, cn, device=dev, ode_steps=ode_steps,
                              solver=solver,
                              s_param=cfg.get("s_param", "logW"))


def birth_analog(n, W, seed):
    """Analog MC for the birth cell (internal isotropic source)."""
    return sample_single_cell(n, W, W, mode="internal", seed=seed,
                              n_blocks=8)


def energy_distance(X, Y, max_n=1400, rng=None):
    """Multivariate energy distance between two point clouds."""
    rng = rng or np.random.default_rng(0)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    def pd(A, B):
        d = A[:, None, :] - B[None, :, :]
        return np.sqrt((d * d).sum(-1))
    return float(2 * pd(X, Y).mean() - pd(X, X).mean() - pd(Y, Y).mean())


def joint_vec(p, W, ox, oy, oz, s):
    """The 6-d exit vector the transport solve actually consumes."""
    th = 2 * np.pi * p / (4 * W)
    return np.stack([np.cos(th), np.sin(th), ox, oy, oz,
                     np.log(s / W)], axis=1)


# ------------------------------------------------------------- test 1+2
def accuracy_and_joint(args, sampler):
    print("=" * 68)
    print("TEST 1 - ACCURACY: full lattice, GMC vs Monte Carlo")
    print("=" * 68)

    prob = build_lattice(5, pitch=1.0, cells_per_pitch=16)
    ss_m, sa_m, uniform = macro_problem(prob, 1.0, 16)
    print(f"macro grid {ss_m.shape[0]}x{ss_m.shape[1]} cells of 1 cm; "
          f"materially uniform: {uniform}")
    print(f"cell optical sizes present: "
          f"{sorted(set(np.round(1.0 * ss_m.ravel(), 6)))} mfp")
    src = (3, 3)   # central lattice cell holds the source

    n = args.n
    t0 = time.time()
    phi_mc_fine = run_transport(prob, n, seed=11)
    t_mc = time.time() - t0
    phi_mc = coarsen(phi_mc_fine, 16)

    # independent MC run -> statistical floor for the field comparison
    phi_mc2 = coarsen(run_transport(prob, n, seed=22), 16)

    t0 = time.time()
    phi_gmc, stats = run_gmc_transport(ss_m, sa_m, 1.0, n, sampler, src,
                                       seed=33, birth_sampler=birth_analog,
                                       return_stats=True)
    t_gmc = time.time() - t0

    print(f"\nMC  {n:,} particles in {t_mc:6.2f} s")
    print(f"GMC {n:,} particles in {t_gmc:6.2f} s   "
          f"({stats['crossings_per_particle']:.2f} cell crossings/particle, "
          f"{stats['batched_calls']} batched sampler calls)")

    m = (phi_mc > 0) & (phi_gmc > 0)
    rel = np.abs(phi_gmc - phi_mc) / phi_mc
    rel_floor = np.abs(phi_mc2 - phi_mc) / phi_mc
    print(f"\ncell-wise relative difference over {m.sum()} cells:")
    print(f"  GMC vs MC : median {np.median(rel[m])*100:6.2f}%   "
          f"mean {rel[m].mean()*100:6.2f}%   max {rel[m].max()*100:6.2f}%")
    print(f"  MC  vs MC : median {np.median(rel_floor[m])*100:6.2f}%   "
          f"mean {rel_floor[m].mean()*100:6.2f}%   max "
          f"{rel_floor[m].max()*100:6.2f}%   (statistical floor)")
    tot_g, tot_m = phi_gmc.sum(), phi_mc.sum()
    print(f"  integral  : GMC {tot_g:.4f}  MC {tot_m:.4f}  "
          f"({100*(tot_g/tot_m-1):+.2f}%)")

    # ---------------- test 2: joint distribution ----------------------
    print()
    print("=" * 68)
    print("TEST 2 - JOINT DISTRIBUTION of the exit state (p, Omega, s)")
    print("=" * 68)
    d = np.load(ROOT / "data/lattice_singlecell.npz")
    rng = np.random.default_rng(7)
    print(f"{'W':>6} {'energy dist':>12} {'MC-MC floor':>12} {'ratio':>7}"
          f"   {'max |corr| err':>14}")
    joint_rows = []
    for wsel in (0.5, 1.0, 2.5, 10.0):
        mw = np.isclose(d["W"], wsel) & (d["k"] > 0)
        if mw.sum() < 400:
            continue
        conds = np.unique(np.stack([d["y0"][mw], d["oxi"][mw],
                                    d["oyi"][mw]]), axis=1)
        # pool a fixed set of conditions so both clouds share conditioning
        pick = rng.choice(conds.shape[1], min(64, conds.shape[1]),
                          replace=False)
        ref, gen = [], []
        for i in pick:
            y0i, oxii, oyii = conds[:, i]
            mi = mw & np.isclose(d["y0"], y0i) & np.isclose(d["oxi"], oxii)
            k = int(mi.sum())
            if k < 8:
                continue
            ref.append(joint_vec(d["p"][mi], wsel, d["oxo"][mi],
                                 d["oyo"][mi], d["ozo"][mi], d["s"][mi]))
            g = sampler.sample(np.full(k, wsel), np.full(k, wsel),
                               np.full(k, y0i), np.full(k, oxii),
                               np.full(k, oyii),
                               seed=int(rng.integers(2**31)),
                               uncollided="off")
            gen.append(joint_vec(g["p"], wsel, g["dir"][:, 0],
                                 g["dir"][:, 1], g["dir"][:, 2], g["s"]))
        R, G = np.concatenate(ref), np.concatenate(gen)
        mu, sd = R.mean(0), R.std(0) + 1e-9
        Rs, Gs = (R - mu) / sd, (G - mu) / sd
        h = rng.permutation(len(Rs))
        A, B = h[: len(h) // 2], h[len(h) // 2: 2 * (len(h) // 2)]
        # Gs is built row-for-row from the same conditions as Rs, so index
        # it with the SAME rows as the reference half -- slicing Gs[:len(A)]
        # instead would compare clouds drawn from different conditions.
        e_model = energy_distance(Rs[A], Gs[A], rng=rng)
        e_floor = energy_distance(Rs[A], Rs[B], rng=rng)
        cR, cG = np.corrcoef(Rs.T), np.corrcoef(Gs.T)
        cerr = np.abs(cR - cG)[np.triu_indices(6, 1)].max()
        print(f"{wsel:6g} {e_model:12.4f} {e_floor:12.4f} "
              f"{e_model/max(e_floor,1e-12):7.2f}   {cerr:14.3f}")
        joint_rows.append((wsel, e_model, e_floor, cerr, cR, cG))

    return prob, phi_mc, phi_mc2, phi_gmc, joint_rows, t_mc, t_gmc, stats


# --------------------------------------------------------------- test 3
def speed_sweep(args, sampler):
    print()
    print("=" * 68)
    print("TEST 3 - SPEED: same geometry, swept optical scale")
    print("=" * 68)
    print("cells scale as W = pitch * sigma_s; the lattice at its published")
    print("scale is optically THIN, where GMC cannot win.\n")
    print(f"{'W_bg':>6} {'W_abs':>6} {'MC (s)':>9} {'GMC (s)':>9} "
          f"{'speedup':>8} {'MC scat/part':>13}")
    rows = []
    n = args.speed_n
    for scale in args.scales:
        prob = build_lattice(5, pitch=1.0, cells_per_pitch=16,
                             background={"sig_s": 1.0 * scale, "sig_a": 0.0},
                             absorber={"sig_s": 0.5 * scale,
                                       "sig_a": 9.5 * scale})
        ss_m, sa_m, _ = macro_problem(prob, 1.0, 16)
        run_transport(prob, 200, seed=1)            # warm the JIT
        t0 = time.time(); run_transport(prob, n, seed=11); t_mc = time.time() - t0
        t0 = time.time()
        _, st = run_gmc_transport(ss_m, sa_m, 1.0, n, sampler, (3, 3),
                                  seed=33, birth_sampler=birth_analog,
                                  return_stats=True)
        t_gmc = time.time() - t0
        _, kmean = __import__("mc2d").time_single_cell(4000, 1.0 * scale,
                                                       n_blocks=4)
        print(f"{1.0*scale:6.1f} {0.5*scale:6.1f} {t_mc:9.2f} {t_gmc:9.2f} "
              f"{t_mc/t_gmc:8.2f}x {kmean:13.1f}")
        rows.append((scale, t_mc, t_gmc, kmean, st["crossings_per_particle"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--speed-n", type=int, default=4000)
    ap.add_argument("--ckpt", default="models/boundary_v1")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 4.0, 10.0, 20.0])
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--ode-steps", type=int, default=25)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    sampler = load_sampler(ROOT / args.ckpt, ode_steps=args.ode_steps,
                           device=args.device)
    print(describe(sampler.device))
    print(f"checkpoint {args.ckpt}: s_param = {sampler.s_param}, "
          f"{args.ode_steps} ODE steps ({sampler.solver})")
    prob, phi_mc, phi_mc2, phi_gmc, joint, t_mc, t_gmc, stats = \
        accuracy_and_joint(args, sampler)
    speed = None if args.skip_speed else speed_sweep(args, sampler)

    # ------------------------------------------------------------ figure
    nrow = 2 if speed else 1
    fig = plt.figure(figsize=(14, 4.3 * nrow))
    gs = fig.add_gridspec(nrow, 4, hspace=0.42, wspace=0.34)

    vmin = np.log10(max(phi_mc[phi_mc > 0].min(), 1e-12))
    vmax = np.log10(phi_mc.max())
    for j, (f, lab) in enumerate([(phi_mc, "Monte Carlo"),
                                  (phi_gmc, "GMC (learned sampler)")]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(np.log10(np.maximum(f, 1e-30)), origin="lower",
                       extent=[0, 7, 0, 7], cmap="plasma",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"{lab}\n$\\log_{{10}}\\phi$, {args.n:,} particles",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    rel = 100 * (phi_gmc - phi_mc) / phi_mc
    im = ax.imshow(rel, origin="lower", extent=[0, 7, 0, 7], cmap="RdBu_r",
                   vmin=-25, vmax=25)
    ax.set_title("relative difference (%)\nGMC $-$ MC", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 3])
    m = (phi_mc > 0) & (phi_gmc > 0)
    ax.loglog(phi_mc[m], phi_gmc[m], "o", ms=4, alpha=.65,
              color="#4C3A8F", label="GMC")
    ax.loglog(phi_mc[m], phi_mc2[m], "s", ms=3.4, alpha=.5,
              color="0.55", label="MC (indep. run)")
    lo, hi = phi_mc[m].min(), phi_mc[m].max()
    ax.loglog([lo, hi], [lo, hi], "k--", lw=1, label="exact")
    ax.set_xlabel("MC cell flux"); ax.set_ylabel("predicted cell flux")
    ax.set_title("cell-by-cell agreement", fontsize=9)
    ax.legend(fontsize=7.5)

    if speed:
        s = np.array([[r[0], r[1], r[2], r[3]] for r in speed], float)
        ax = fig.add_subplot(gs[1, 0])
        ax.loglog(s[:, 0], s[:, 1], "o-", label="MC", color="k")
        ax.loglog(s[:, 0], s[:, 2], "s-", label="GMC", color="#4C3A8F")
        ax.set_xlabel("optical scale factor"); ax.set_ylabel("wall time (s)")
        ax.set_title("cost vs optical scale", fontsize=9); ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, 1])
        ax.loglog(s[:, 3], s[:, 2] / s[:, 1], "o-", color="#A8630F")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.text(s[0, 3], 1.35, "break-even", fontsize=7.5)
        ax.set_xlabel("mean scatters per crossing (= W)")
        ax.set_ylabel("GMC time / MC time")
        ax.set_title("GMC is slower by this factor\n(<1 would favour GMC)",
                     fontsize=9)

        ax = fig.add_subplot(gs[1, 2])
        ax.loglog(s[:, 3], s[:, 1], "o-", color="k", label="MC")
        ax.loglog(s[:, 3], s[:, 2], "s-", color="#4C3A8F", label="GMC")
        ax.set_xlabel("mean scatters per cell crossing")
        ax.set_ylabel("wall time (s)")
        ax.set_title("cost vs work per crossing", fontsize=9)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, 3])
        if joint:
            w = [r[0] for r in joint]
            em = [r[1] for r in joint]
            fl = [r[2] for r in joint]
            ax.semilogx(w, np.array(em) / np.array(fl), "o-",
                        color="#4C3A8F")
            ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set_xlabel("cell optical size (mfp)")
            ax.set_ylabel("energy dist / MC floor")
            ax.set_title("joint distribution\n(1.0 = indistinguishable)",
                         fontsize=9)

    fig.suptitle("GMC end-to-end: full lattice solve, joint distribution, "
                 "and cost", fontsize=11)
    out = ROOT / "figures" / "gmc_end_to_end.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"\nwrote {out}")

    if joint:
        w, _, _, _, cR, cG = joint[1] if len(joint) > 1 else joint[0]
        f2, ax2 = plt.subplots(1, 3, figsize=(11.5, 3.5))
        lbl = ["cos", "sin", "$\\Omega_x$", "$\\Omega_y$", "$\\Omega_z$",
               "log s"]
        dif = cG - cR
        dmax = max(np.abs(dif).max(), 1e-3)
        for a, M, t, v in [
                (ax2[0], cR, f"MC correlation, W={w:g}", 1.0),
                (ax2[1], cG, "GMC correlation", 1.0),
                (ax2[2], dif, f"difference (max {dmax:.3f})", dmax)]:
            im = a.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
            a.set_xticks(range(6)); a.set_xticklabels(lbl, fontsize=7.5,
                                                      rotation=45)
            a.set_yticks(range(6)); a.set_yticklabels(lbl, fontsize=7.5)
            a.set_title(t, fontsize=9)
            f2.colorbar(im, ax=a, fraction=0.046)
        f2.tight_layout()
        out2 = ROOT / "figures" / "gmc_joint_correlation.png"
        f2.savefig(out2, dpi=145)
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()
