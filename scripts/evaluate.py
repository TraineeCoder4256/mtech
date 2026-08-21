#!/usr/bin/env python3
"""End-to-end evaluation.  Run: python scripts/evaluate.py

One geometry throughout: the 7x7 cm lattice from mc2d.problems -- absorbing
blocks in a checkerboard, isotropic source in the middle cell.

ACCURACY.  Solve it twice with Monte Carlo (different seeds) and once with
the sampler.  The two MC runs differ only by seed, so the gap between them
is pure statistical noise: that is the floor, and the only honest yardstick.
Everything is compared on the macro-cell grid, because the sampler returns
total path length in a cell but not where inside it went.

SPEED.  The same geometry with every cross section scaled up, which makes
cells optically thicker without changing the layout.  MC cost per cell grows
with thickness; the sampler's does not.  Somewhere there is a crossover.

Writes figures/geometry.pdf, figures/accuracy.pdf, figures/speed.pdf and
results/evaluation.txt.
"""
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
from mc2d import run_transport                                  # noqa: E402
from mc2d.problems import lattice_problem                       # noqa: E402
from gmc import (VelocityField, GMCBoundarySampler,             # noqa: E402
                 load_normalizers, run_gmc_transport,
                 macro_problem, coarsen)

# ---- settings ----------------------------------------------------------
CKPT = ROOT / "models" / "boundary"
N = 20000                       # particles for the accuracy solve
SPEED_N = 4000                  # particles per point in the speed sweep
SCALES = [1.0, 4.0, 10.0, 20.0]  # cross-section multipliers
ODE_STEPS, SOLVER = 5, "heun"
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PITCH = 1.0                     # macro cell = one 1 cm lattice cell
NPC = 16                        # fine cells per macro cell (112 / 7)
SOURCE_CELL = (3, 3)            # the source box is [3,4] x [3,4] cm
MIN_TIME = 0.25                 # repeat MC until it has run this long


def load_sampler():
    st = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True)
    c = st["config"]
    m = VelocityField(c["x_dim"], c["c_dim"], c["width"], c["depth"])
    m.load_state_dict(st["ema"])                # sample from the EMA weights
    yn, cn, _ = load_normalizers(CKPT / "normalizers.json")
    return GMCBoundarySampler(m, yn, cn, DEVICE, ODE_STEPS, SOLVER)


def lattice(scale=1.0):
    """The one geometry, with every cross section multiplied by `scale`."""
    p = lattice_problem()
    p["sig_s"] = p["sig_s"] * scale
    p["sig_a"] = p["sig_a"] * scale
    return p


def timed_mc(prob, n, seed):
    """Run MC, repeating until timing it is meaningful, and return the mean.

    One 20k-particle solve takes a few milliseconds at the physical scale,
    which is timer noise.
    """
    run_transport(prob, 200, seed=0)                    # warm the numba JIT
    reps, t0, elapsed = 0, time.perf_counter(), 0.0
    while elapsed < MIN_TIME:
        phi, stats = run_transport(prob, n, seed=seed + reps, return_stats=True)
        reps += 1
        elapsed = time.perf_counter() - t0
    return phi, stats, elapsed / reps


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------- figures
def figure_geometry(phi_fine, phi_macro, path):
    """The problem itself: materials, optical thickness, and the flux."""
    prob = lattice()
    L = prob["Lx"]
    ss_m, _, _ = macro_problem(prob, PITCH, NPC)
    ext = [0, L, 0, L]
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.3))

    a = ax[0]
    a.imshow((prob["sig_a"] > 0).astype(float), origin="lower", extent=ext,
             cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    x0, y0, x1, y1 = prob["source"]["box"]
    a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#D64545",
                              edgecolor="k", lw=1.2))
    for g in np.arange(0, L + .01, PITCH):
        a.axhline(g, color="#3C6E9F", lw=.6, alpha=.6)
        a.axvline(g, color="#3C6E9F", lw=.6, alpha=.6)
    a.set_title("geometry: 7x7 cm lattice", fontsize=10)
    a.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="k",
                      label=r"background  $\sigma_s$=1, $\sigma_a$=0"),
        plt.Rectangle((0, 0), 1, 1, fc="0.35", ec="k",
                      label=r"absorber  $\sigma_s$=0.5, $\sigma_a$=9.5"),
        plt.Rectangle((0, 0), 1, 1, fc="#D64545", ec="k",
                      label="isotropic source"),
        plt.Line2D([0], [0], color="#3C6E9F", lw=1, label="macro cells")],
        fontsize=7, loc="upper right", framealpha=.92)

    a = ax[1]
    W_cell = PITCH * ss_m
    im = a.imshow(W_cell, origin="lower", extent=ext, cmap="magma",
                  interpolation="nearest")
    for j in range(W_cell.shape[0]):
        for i in range(W_cell.shape[1]):
            a.text(i + .5, j + .5, f"{W_cell[j, i]:.1f}", ha="center",
                   va="center", fontsize=7.5,
                   color="w" if W_cell[j, i] < W_cell.max() * .6 else "k")
    a.set_title("optical width of each cell\n"
                r"$\tilde W$ = pitch $\times\ \sigma_s$ (mfp)", fontsize=10)
    fig.colorbar(im, ax=a, fraction=.046, label="mfp")

    lo = np.log10(max(phi_fine[phi_fine > 0].min(), 1e-10))
    hi = np.log10(phi_fine.max())
    for a, M, ttl in ((ax[2], phi_fine, f"Monte Carlo flux\nfine mesh, "
                                        f"{phi_fine.shape[0]}x{phi_fine.shape[1]}"),
                      (ax[3], phi_macro, f"same flux on macro cells\n"
                                         f"{phi_macro.shape[0]}x{phi_macro.shape[1]}"
                                         f" -- what GMC is compared against")):
        im = a.imshow(np.log10(np.maximum(M, 1e-10)), origin="lower",
                      extent=ext, cmap="viridis", vmin=lo, vmax=hi,
                      interpolation="nearest")
        a.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=a, fraction=.046, label=r"$\log_{10}\phi$")

    for a in ax:
        a.set_xlabel("x (cm)")
        a.set_ylabel("y (cm)")
    fig.suptitle("The problem being solved", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path)
    plt.close(fig)


def figure_accuracy(r, path):
    A, B, G = r["mc"], r["mc_repeat"], r["gmc"]
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.0))
    lo = np.log10(max(A[A > 0].min(), 1e-10))
    hi = np.log10(A.max())

    for a, M, ttl in ((ax[0], A, "Monte Carlo (reference)"),
                      (ax[1], G, "GMC (learned sampler)")):
        im = a.imshow(np.log10(np.maximum(M, 1e-10)), origin="lower",
                      vmin=lo, vmax=hi, cmap="viridis")
        a.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=a, fraction=.046, label=r"$\log_{10}\phi$")

    d = (G - A) / np.maximum(A, 1e-30) * 100
    v = np.percentile(np.abs(d), 98)
    im = ax[2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax[2].set_title(f"(GMC $-$ MC) / MC\nrelative $L_2$ {r['err']*100:.2f}%, "
                    f"floor {r['floor']*100:.2f}%", fontsize=10)
    fig.colorbar(im, ax=ax[2], fraction=.046, label="%")

    row = A.shape[0] // 2
    x = np.arange(A.shape[1])
    ax[3].semilogy(x, A[row], "k-o", ms=4, label="MC")
    ax[3].semilogy(x, B[row], color="0.6", ls="--", marker="s", ms=3,
                   label="MC, other seed (noise floor)")
    ax[3].semilogy(x, G[row], "r-^", ms=4, label="GMC")
    ax[3].set_title(f"lineout through the source row (y = {row})", fontsize=10)
    ax[3].set_ylabel(r"$\phi$ per source particle")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=.3)
    for a in ax[:3]:
        a.set_xlabel("cell x"); a.set_ylabel("cell y")
    ax[3].set_xlabel("cell x")

    fig.suptitle(f"Accuracy on the lattice, {N:,} particles per solve",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path)
    plt.close(fig)


def figure_speed(rows, path):
    W = np.array([r["W"] for r in rows])
    tm = np.array([r["t_mc"] for r in rows])
    tg = np.array([r["gmc"]["wall"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].loglog(W, tm, "k-o", label="Monte Carlo")
    ax[0].loglog(W, tg, "r-^", label="GMC")
    ax[0].set_ylabel("wall time (s)")
    ax[0].set_title("cost vs optical thickness\nMC grows, GMC is flat",
                    fontsize=10)
    ax[0].legend(fontsize=9)

    ax[1].loglog(W, tm / tg, "b-o")
    ax[1].axhline(1.0, color="k", ls="--", lw=1.2)
    ax[1].set_ylabel("MC time / GMC time")
    ax[1].set_title("speedup\nabove 1 the sampler wins", fontsize=10)

    net = np.array([r["gmc"]["wall_sampler"] for r in rows]) / tg * 100
    birth = np.array([r["gmc"]["wall_birth"] for r in rows]) / tg * 100
    host = np.array([r["gmc"]["wall_overhead"] for r in rows]) / tg * 100
    b = np.arange(len(W))
    ax[2].bar(b, net, label="network (ODE solve)", color="#C1443C")
    ax[2].bar(b, birth, bottom=net, label="analog MC birth cell",
              color="#E0A458")
    ax[2].bar(b, host, bottom=net + birth, label="geometry / tallies",
              color="#2F6F8F")
    ax[2].set_xticks(b)
    ax[2].set_xticklabels([f"{w:g}" for w in W])
    ax[2].set_ylabel("% of GMC wall time")
    ax[2].set_title("where GMC's time goes", fontsize=10)
    ax[2].legend(fontsize=8)

    for a in ax:
        a.set_xlabel("cell optical width (mfp)")
        a.grid(alpha=.3, which="both")
    fig.suptitle("Speed on the same lattice, cross sections scaled",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------- report
def report(acc, rows, sampler, path):
    a, g = acc["mc_stats"], acc["gmc_stats"]
    tm, tg = acc["t_mc"], acc["gmc_stats"]["wall"]
    err = np.abs(acc["gmc"] - acc["mc"]) / np.maximum(acc["mc"], 1e-30)

    # per-unit costs: MC pays per scattering event, GMC per cell crossing
    cross = max(g["crossings"], 1)
    t_scatter = tm / max(a["scatters"], 1)
    t_nfe = g["wall_sampler"] / max(g["nfe"], 1)
    per_cross_mc = a["scatters"] / cross * t_scatter
    per_cross_net = g["wall_sampler"] / cross
    per_cross_host = (g["wall_overhead"] + g["wall_birth"]) / cross
    per_cross_gmc = tg / cross
    scat_per_cross = a["scatters"] / cross
    breakeven = per_cross_gmc / t_scatter

    sweep = "\n".join(
        f"{r['scale']:>6g} {r['W']:>7.1f} {r['t_mc']:>9.3f} "
        f"{r['gmc']['wall']:>9.2f} {r['t_mc']/r['gmc']['wall']:>9.4g}x "
        f"{r['mc']['scatters_per_particle']:>10.1f} "
        f"{r['mc']['scatters']/max(r['gmc']['crossings'],1):>11.1f} "
        f"{r['gmc']['wall_sampler']/r['gmc']['wall']*100:>6.1f}% "
        f"{(r['gmc']['wall_overhead']+r['gmc']['wall_birth'])/r['gmc']['wall']*100:>6.1f}%"
        for r in rows)

    sp = np.array([r["t_mc"] / r["gmc"]["wall"] for r in rows])
    Wb = np.array([r["W"] for r in rows])
    if (sp > 1).any() and (sp < 1).any():
        i = int(np.argmax(sp > 1))
        xc = Wb[i-1] * (Wb[i]/Wb[i-1]) ** ((1 - sp[i-1]) / (sp[i] - sp[i-1]))
        verdict = f"CROSSOVER: GMC overtakes MC at about {xc:.1f} mfp per cell"
    elif (sp > 1).all():
        verdict = f"GMC is faster at every scale tested (from {Wb.min():g} mfp)"
    else:
        verdict = (f"GMC is slower at every scale tested (up to {Wb.max():g} "
                   f"mfp).\nBest {sp.max():.3g}x at {Wb[int(np.argmax(sp))]:g}"
                   f" mfp; extrapolate with the break-even figure above.")

    text = f"""\
==========================================================================
END-TO-END EVALUATION -- RAW NUMBERS
==========================================================================
device        {DEVICE}
solver        {SOLVER}, {ODE_STEPS} steps = {sampler.nfe_per_sample} network evaluations per sample
geometry      7x7 cm lattice, 112x112 fine mesh, 7x7 macro cells of {PITCH:g} cm

--------------------------------------------------------------------------
1. ACCURACY   ({N:,} particles per solve)
--------------------------------------------------------------------------
GMC relative L2 error vs MC          {acc['err']*100:8.3f} %
MC-vs-MC floor (same n, new seed)    {acc['floor']*100:8.3f} %
ratio to the floor                   {acc['err']/acc['floor']:8.2f} x

The floor is what two identical MC runs differ by, so it is the best any
method could score.  A ratio of 1 means the model's error is
indistinguishable from statistical noise.

per-cell relative error   median {np.median(err)*100:6.2f} %   \
90th pct {np.percentile(err, 90)*100:6.2f} %   max {err.max()*100:6.2f} %
path lengths clamped to the straight-line minimum: \
{g['clamped']/max(g['to_network'],1)*100:.2f} %

--------------------------------------------------------------------------
2. COST AT THE PHYSICAL SCALE (scale = 1)
--------------------------------------------------------------------------
MONTE CARLO
  wall time                       {tm:10.4f} s   (mean of repeated runs)
  particles                       {a['particles']:10,}
  scattering events               {a['scatters']:10,.0f}
  scatters per particle           {a['scatters_per_particle']:10.2f}
  time per particle               {tm/a['particles']*1e6:10.2f} us
  time per scattering event       {t_scatter*1e9:10.1f} ns

GMC
  wall time                       {tg:10.4f} s
  particles                       {g['particles']:10,}
  macro-cell crossings            {g['crossings']:10,}
  crossings per particle          {g['crossings_per_particle']:10.2f}
  batched sampler calls           {g['calls']:10,}
  mean / median live batch        {g['mean_batch']:10.0f} / {g['median_batch']:.0f}
  crossings reaching the network  {g['to_network']:10,}   \
({g['to_network']/cross*100:.1f} %; the rest were uncollided)
  network evaluations (NFE)       {g['nfe']:10,}
  time per particle               {tg/g['particles']*1e6:10.2f} us
  time per macro-cell crossing    {per_cross_gmc*1e6:10.2f} us
  time per NFE                    {t_nfe*1e6:10.2f} us
  NFE per second                  {g['nfe']/max(g['wall_sampler'],1e-9):10,.0f}

  where GMC's wall time goes
    network (ODE solve)           {g['wall_sampler']:10.4f} s  \
({g['wall_sampler']/tg*100:5.1f} %)
    analog MC in the birth cell   {g['wall_birth']:10.4f} s  \
({g['wall_birth']/tg*100:5.1f} %)
    geometry / tallies (NumPy)    {g['wall_overhead']:10.4f} s  \
({g['wall_overhead']/tg*100:5.1f} %)

SPEEDUP AT SCALE 1                {tm/tg:10.4g} x   \
(GMC is {tg/tm:,.0f}x slower)

--------------------------------------------------------------------------
3. WHY THE SPEEDUP IS WHAT IT IS
--------------------------------------------------------------------------
MC does one unit of work per scattering event.  GMC does a fixed amount of
work per cell crossing, however many scatters that replaces.  So: what does
one crossing cost each way?

  MC   per scattering event       {t_scatter*1e9:10.1f} ns
       scatters per crossing      {scat_per_cross:10.2f}
       => per crossing            {per_cross_mc*1e6:10.2f} us

  GMC  per NFE                    {t_nfe*1e6:10.2f} us
       NFE per sample             {sampler.nfe_per_sample:10d}
       => network per crossing    {per_cross_net*1e6:10.2f} us
       + host (geometry, tallies) {per_cross_host*1e6:10.2f} us
       => per crossing            {per_cross_gmc*1e6:10.2f} us

BREAK-EVEN: one GMC crossing costs as much as {breakeven:,.0f} scattering events.
            This geometry has {scat_per_cross:.1f} per crossing, so cells must be
            about {breakeven/max(scat_per_cross,1e-9):,.0f}x optically thicker before GMC breaks even.

Which problem you have depends on the wall-time split above:
  * network dominates  -> the model is too expensive (fewer ODE steps, a
    cheaper solver, distillation).
  * host loop dominates -> the model is not the bottleneck at all, and no
    amount of model work will fix it.

Note what MC is being timed against: mc2d is numba-compiled and
multi-threaded, so the per-scatter time above is already a hard target.  A
slower baseline would flatter GMC without changing anything real.

--------------------------------------------------------------------------
4. SPEED SWEEP   ({SPEED_N:,} particles per point)
--------------------------------------------------------------------------
 scale    W_bg    MC (s)   GMC (s)   speedup  scat/part  scat/cross   net %  host %
-----------------------------------------------------------------------------------
{sweep}

{verdict}
==========================================================================
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ------------------------------------------------------------------- main
def main():
    print(f"device {DEVICE}")
    sampler = load_sampler()
    print(f"solver {SOLVER}, {ODE_STEPS} steps "
          f"= {sampler.nfe_per_sample} NFE per sample\n")
    (ROOT / "figures").mkdir(exist_ok=True)

    print(f"1. accuracy ({N:,} particles per solve)")
    prob = lattice()
    ss, sa, uniform = macro_problem(prob, PITCH, NPC)
    assert uniform, "macro cells must be materially uniform"
    phi_a, mc_stats, t_mc = timed_mc(prob, N, SEED)
    phi_b = run_transport(prob, N, seed=SEED + 1000)      # independent repeat
    phi_g, g_stats = run_gmc_transport(ss, sa, PITCH, N, sampler, SOURCE_CELL,
                                       seed=SEED + 7)
    acc = {"mc": coarsen(phi_a, NPC), "mc_repeat": coarsen(phi_b, NPC),
           "gmc": phi_g, "mc_fine": phi_a, "t_mc": t_mc,
           "mc_stats": mc_stats, "gmc_stats": g_stats}
    acc["err"] = rel_l2(acc["gmc"], acc["mc"])
    acc["floor"] = rel_l2(acc["mc_repeat"], acc["mc"])
    print(f"   GMC {acc['err']*100:.3f} %   floor {acc['floor']*100:.3f} %   "
          f"ratio {acc['err']/acc['floor']:.2f}x")
    figure_accuracy(acc, ROOT / "figures" / "accuracy.pdf")
    figure_geometry(acc["mc_fine"], acc["mc"], ROOT / "figures" / "geometry.pdf")
    print("   wrote figures/accuracy.pdf, figures/geometry.pdf")

    print(f"\n2. speed sweep ({SPEED_N:,} particles per point)")
    rows = []
    for sc in SCALES:
        p = lattice(sc)
        ss, sa, _ = macro_problem(p, PITCH, NPC)
        _, mcs, t = timed_mc(p, SPEED_N, SEED + 10)
        _, gs = run_gmc_transport(ss, sa, PITCH, SPEED_N, sampler,
                                  SOURCE_CELL, seed=SEED + 13)
        rows.append({"scale": sc, "W": PITCH * sc, "t_mc": t, "mc": mcs,
                     "gmc": gs})
        print(f"   scale {sc:5g}  MC {t:8.4f}s  GMC {gs['wall']:7.2f}s  "
              f"speedup {t/gs['wall']:8.4g}x  "
              f"({mcs['scatters_per_particle']:.0f} scatters/particle)",
              flush=True)
    figure_speed(rows, ROOT / "figures" / "speed.pdf")
    print("   wrote figures/speed.pdf")

    print()
    print(report(acc, rows, sampler, ROOT / "results" / "evaluation.txt"))
    print("wrote results/evaluation.txt")


if __name__ == "__main__":
    main()
