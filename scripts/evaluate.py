#!/usr/bin/env python3
"""End-to-end evaluation: does the learned sampler solve a real problem, and
is it faster?

ONE geometry throughout: the 7x7 cm lattice of mc2d.problems.lattice_problem
-- a checkerboard of absorbing blocks in a scattering background, with an
isotropic source in the middle cell.  Two questions, asked separately.

ACCURACY.  Solve it twice with Monte Carlo (different seeds) and once with
the generative sampler, then compare all three on the 7x7 macro-cell grid.
The two MC runs differ only by seed, so the difference between them is pure
statistical noise -- that is the floor, and the only honest yardstick for
the model's error.  Beating the floor is impossible; approaching it is the
goal.

Why the macro grid and not the fine 112x112 mesh: the sampler returns the
total path length inside a cell but not where inside it went, so the flux
it produces is inherently cell-averaged.  Comparing it against the fine
mesh would be comparing two different quantities.

SPEED.  The same geometry with every cross section multiplied by a scale
factor.  Scaling makes the cells optically thicker without changing the
layout, and optical thickness is the whole game: Monte Carlo cost per cell
grows with it (more scattering events to simulate) while the sampler's cost
is fixed (the same number of network evaluations regardless).  Somewhere
there is a crossover.  This finds it, and the numbers file explains why it
sits where it does.

Outputs:
    figures/geometry.pdf     what the problem actually is: materials, source,
                             optical thickness, and the resulting flux
    figures/accuracy.pdf     flux fields, error map, lineout
    figures/speed.pdf        wall time and speedup vs optical thickness
    results/evaluation.txt   every raw number behind both figures

Usage:
  python scripts/evaluate.py --ckpt models/boundary --device auto
  python scripts/evaluate.py --n 50000 --speed-n 4000 --scales 1 4 10 20
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
from mc2d import run_transport                              # noqa: E402
from mc2d.problems import lattice_problem                   # noqa: E402
from gmc import VelocityField, GMCBoundarySampler           # noqa: E402
from gmc.data import load_normalizers                       # noqa: E402
from gmc.device import pick_device, describe                # noqa: E402
from gmc.transport import (run_gmc_transport, macro_problem,  # noqa: E402
                           coarsen)

PITCH = 1.0            # macro cell = 1 cm lattice cell
CELLS_PER_PITCH = 16   # 112 fine cells / 7 cm
SOURCE_CELL = (3, 3)   # the source box is [3,4] x [3,4] cm


def load_sampler(ckpt_dir, ode_steps, solver, device):
    st = torch.load(ckpt_dir / "model.pt", map_location="cpu",
                    weights_only=True)
    cfg = st["config"]
    m = VelocityField(x_dim=cfg["x_dim"], c_dim=cfg["c_dim"],
                      width=cfg["width"], depth=cfg["depth"])
    m.load_state_dict(st["ema"])          # sample from the EMA weights
    yn, cn, _ = load_normalizers(ckpt_dir / "normalizers.json")
    return GMCBoundarySampler(m, yn, cn, device=device, ode_steps=ode_steps,
                              solver=solver)


def scaled_lattice(scale):
    """The one geometry, with every cross section multiplied by `scale`."""
    p = lattice_problem()
    p["sig_s"] = p["sig_s"] * scale
    p["sig_a"] = p["sig_a"] * scale
    return p


def timed_mc(prob, n, seed, min_seconds=0.25):
    """Run the MC solve, repeating until the total is long enough to time.

    At the physical scale MC finishes 20,000 particles in a few
    milliseconds, which is close to timer noise and to the cost of the
    surrounding Python.  Repeating until a quarter of a second has elapsed
    and dividing is the difference between a real number and an artefact.
    """
    run_transport(prob, 200, seed=0)                 # warm the numba JIT
    reps, elapsed = 0, 0.0
    t0 = time.perf_counter()
    while elapsed < min_seconds:
        phi, stats = run_transport(prob, n, seed=seed + reps,
                                   return_stats=True)
        reps += 1
        elapsed = time.perf_counter() - t0
    return phi, stats, elapsed / reps, reps


def rel_l2(a, b):
    """Relative L2 difference between two flux fields."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------------------ geometry
def figure_geometry(phi_fine, phi_macro, path):
    """Draw the problem itself, so the reader knows what is being solved.

    Four panels, left to right: what the materials are, how optically thick
    that makes each cell (which is the quantity the whole speed argument
    turns on), the Monte Carlo flux on the fine mesh, and the same flux
    averaged onto the macro cells -- which is the resolution the generative
    sampler works at and therefore the only fair basis for comparison.
    """
    prob = scaled_lattice(1.0)
    ss, sa = prob["sig_s"], prob["sig_a"]
    L = prob["Lx"]
    ss_m, sa_m, _ = macro_problem(prob, PITCH, CELLS_PER_PITCH)
    ext = [0, L, 0, L]

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.3))

    # ---- 1. materials ------------------------------------------------
    a = ax[0]
    a.imshow((sa > 0).astype(float), origin="lower", extent=ext,
             cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    src = prob["source"]["box"]
    a.add_patch(plt.Rectangle((src[0], src[1]), src[2] - src[0],
                              src[3] - src[1], facecolor="#D64545",
                              edgecolor="k", lw=1.2, alpha=.9))
    for g in np.arange(0, L + .01, PITCH):     # macro-cell grid
        a.axhline(g, color="#3C6E9F", lw=.6, alpha=.65)
        a.axvline(g, color="#3C6E9F", lw=.6, alpha=.65)
    a.set_title("geometry: 7x7 cm lattice", fontsize=10)
    a.set_xlabel("x (cm)"); a.set_ylabel("y (cm)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="k",
                      label=r"background  $\sigma_s$=1, $\sigma_a$=0"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.35", edgecolor="k",
                      label=r"absorber  $\sigma_s$=0.5, $\sigma_a$=9.5"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#D64545", edgecolor="k",
                      label="isotropic source"),
        plt.Line2D([0], [0], color="#3C6E9F", lw=1,
                   label=f"macro cells ({PITCH:g} cm)"),
    ]
    a.legend(handles=handles, fontsize=7, loc="upper right",
             framealpha=.92)

    # ---- 2. optical thickness ----------------------------------------
    a = ax[1]
    W_cell = PITCH * ss_m
    im = a.imshow(W_cell, origin="lower", extent=ext, cmap="magma",
                  interpolation="nearest")
    for j in range(W_cell.shape[0]):
        for i in range(W_cell.shape[1]):
            a.text(i + .5, j + .5, f"{W_cell[j, i]:.1f}", ha="center",
                   va="center", fontsize=7.5,
                   color="w" if W_cell[j, i] < W_cell.max() * .6 else "k")
    a.set_title("optical width of each macro cell\n"
                r"$\tilde W = \mathrm{pitch}\times\sigma_s$ (mfp)",
                fontsize=10)
    a.set_xlabel("x (cm)"); a.set_ylabel("y (cm)")
    fig.colorbar(im, ax=a, fraction=.046, label="mfp")

    # ---- 3. flux, fine mesh ------------------------------------------
    a = ax[2]
    lo = np.log10(max(phi_fine[phi_fine > 0].min(), 1e-10))
    hi = np.log10(phi_fine.max())
    im = a.imshow(np.log10(np.maximum(phi_fine, 1e-10)), origin="lower",
                  extent=ext, cmap="viridis", vmin=lo, vmax=hi)
    a.set_title(f"Monte Carlo flux\nfine mesh, "
                f"{phi_fine.shape[0]}x{phi_fine.shape[1]}", fontsize=10)
    a.set_xlabel("x (cm)"); a.set_ylabel("y (cm)")
    fig.colorbar(im, ax=a, fraction=.046, label=r"$\log_{10}\phi$")

    # ---- 4. flux, macro cells ----------------------------------------
    a = ax[3]
    im = a.imshow(np.log10(np.maximum(phi_macro, 1e-10)), origin="lower",
                  extent=ext, cmap="viridis", vmin=lo, vmax=hi,
                  interpolation="nearest")
    a.set_title(f"the same flux on macro cells\n"
                f"{phi_macro.shape[0]}x{phi_macro.shape[1]} -- what GMC "
                f"is compared against", fontsize=10)
    a.set_xlabel("x (cm)"); a.set_ylabel("y (cm)")
    fig.colorbar(im, ax=a, fraction=.046, label=r"$\log_{10}\phi$")

    fig.suptitle("The problem being solved: a checkerboard of absorbing "
                 "blocks in a scattering background, source in the centre",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ accuracy
def accuracy(sampler, n, seed=1):
    prob = scaled_lattice(1.0)
    ss, sa, uniform = macro_problem(prob, PITCH, CELLS_PER_PITCH)
    assert uniform, "macro cells must be materially uniform"

    phi_a, mc_stats, t_mc, mc_reps = timed_mc(prob, n, seed)
    phi_b = run_transport(prob, n, seed=seed + 1000)     # independent repeat

    phi_g, g_stats = run_gmc_transport(ss, sa, PITCH, n, sampler,
                                       SOURCE_CELL, seed=seed + 7,
                                       return_stats=True)

    A, B, G = (coarsen(phi_a, CELLS_PER_PITCH),
               coarsen(phi_b, CELLS_PER_PITCH), phi_g)
    return {"mc": A, "mc_repeat": B, "gmc": G, "mc_fine": phi_a,
            "t_mc": t_mc,
            "mc_reps": mc_reps, "mc_stats": mc_stats, "gmc_stats": g_stats,
            "err_model": rel_l2(G, A), "err_floor": rel_l2(B, A)}


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
        a.set_xlabel("cell x"); a.set_ylabel("cell y")
        fig.colorbar(im, ax=a, fraction=.046, label=r"$\log_{10}\phi$")

    d = (G - A) / np.maximum(A, 1e-30) * 100
    v = np.percentile(np.abs(d), 98)
    im = ax[2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax[2].set_title(f"(GMC $-$ MC) / MC\nrelative $L_2$ "
                    f"{r['err_model']*100:.2f}%, floor "
                    f"{r['err_floor']*100:.2f}%", fontsize=10)
    ax[2].set_xlabel("cell x"); ax[2].set_ylabel("cell y")
    fig.colorbar(im, ax=ax[2], fraction=.046, label="%")

    row = A.shape[0] // 2
    x = np.arange(A.shape[1])
    ax[3].semilogy(x, A[row], "k-o", ms=4, label="MC")
    ax[3].semilogy(x, B[row], color="0.6", ls="--", marker="s", ms=3,
                   label="MC, different seed (noise floor)")
    ax[3].semilogy(x, G[row], "r-^", ms=4, label="GMC")
    ax[3].set_xlabel("cell x"); ax[3].set_ylabel(r"$\phi$ (per source particle)")
    ax[3].set_title(f"lineout through the source row (y = {row})", fontsize=10)
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=.3)

    fig.suptitle(
        f"Accuracy on the 7x7 lattice, {r['mc_stats']['particles']:,} "
        f"particles per solve, {A.shape[0]}x{A.shape[1]} macro cells",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------- speed
def speed(sampler, n, scales, seed=11):
    rows = []
    for sc in scales:
        prob = scaled_lattice(sc)
        ss, sa, _ = macro_problem(prob, PITCH, CELLS_PER_PITCH)
        _, mcs, t_mc, reps = timed_mc(prob, n, seed)
        _, gs = run_gmc_transport(ss, sa, PITCH, n, sampler, SOURCE_CELL,
                                  seed=seed + 3, return_stats=True)
        rows.append({"scale": sc, "W_bg": PITCH * 1.0 * sc, "t_mc": t_mc,
                     "mc_reps": reps, "mc": mcs, "gmc": gs,
                     "speedup": t_mc / gs["wall"]})
        print(f"  scale {sc:5g}  W_bg {PITCH*sc:6.1f} mfp   "
              f"MC {t_mc:8.4f}s   GMC {gs['wall']:7.2f}s   "
              f"speedup {t_mc/gs['wall']:8.4g}x   "
              f"({mcs['scatters_per_particle']:.0f} scatters/particle)",
              flush=True)
    return rows


def figure_speed(rows, path):
    W = np.array([r["W_bg"] for r in rows])
    tm = np.array([r["t_mc"] for r in rows])
    tg = np.array([r["gmc"]["wall"] for r in rows])
    sp = tm / tg
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].loglog(W, tm, "k-o", label="Monte Carlo")
    ax[0].loglog(W, tg, "r-^", label="GMC")
    ax[0].set_xlabel("background cell optical width (mfp)")
    ax[0].set_ylabel("wall time (s)")
    ax[0].set_title("cost vs optical thickness\nMC grows, GMC is flat",
                    fontsize=10)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, which="both")

    ax[1].loglog(W, sp, "b-o")
    ax[1].axhline(1.0, color="k", ls="--", lw=1.2)
    ax[1].text(W[0], 1.05, "break-even", fontsize=8, va="bottom")
    ax[1].set_xlabel("background cell optical width (mfp)")
    ax[1].set_ylabel("MC time / GMC time")
    ax[1].set_title("speedup\nabove 1 the sampler wins", fontsize=10)
    ax[1].grid(alpha=.3, which="both")

    # where the GMC time actually goes
    net = np.array([r["gmc"]["wall_sampler"] for r in rows])
    birth = np.array([r["gmc"]["wall_birth"] for r in rows])
    over = np.array([r["gmc"]["wall_overhead"] for r in rows])
    b = np.arange(len(W))
    ax[2].bar(b, net / tg * 100, label="network (ODE solve)", color="#C1443C")
    ax[2].bar(b, birth / tg * 100, bottom=net / tg * 100,
              label="analog MC birth cell", color="#E0A458")
    ax[2].bar(b, over / tg * 100, bottom=(net + birth) / tg * 100,
              label="geometry / tallies (NumPy host)", color="#2F6F8F")
    ax[2].set_xticks(b)
    ax[2].set_xticklabels([f"{w:g}" for w in W])
    ax[2].set_xlabel("background cell optical width (mfp)")
    ax[2].set_ylabel("% of GMC wall time")
    ax[2].set_title("where GMC's time goes\nif the network is a small slice,\n"
                    "the sampler is not the bottleneck", fontsize=10)
    ax[2].legend(fontsize=8)

    fig.suptitle("Speed on the same lattice, cross sections scaled",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------- numbers on paper
def write_report(path, acc, rows, sampler, dev, n, speed_n):
    L = []
    def w(s=""):
        L.append(s)

    w("=" * 74)
    w("END-TO-END EVALUATION -- RAW NUMBERS")
    w("=" * 74)
    w(f"device            {describe(dev)}")
    w(f"ODE solver        {sampler.solver}, {sampler.ode_steps} steps "
      f"= {sampler.nfe_per_sample} network evaluations per sample")
    w(f"geometry          7x7 cm lattice, 112x112 fine mesh, "
      f"7x7 macro cells of {PITCH:g} cm")
    w("")

    a, g = acc["mc_stats"], acc["gmc_stats"]
    w("-" * 74)
    w("1. ACCURACY")
    w("-" * 74)
    w(f"particles per solve                {n:,}")
    w(f"GMC relative L2 error vs MC        {acc['err_model']*100:8.3f} %")
    w(f"MC-vs-MC floor (same n, new seed)  {acc['err_floor']*100:8.3f} %")
    w(f"ratio to the floor                 {acc['err_model']/acc['err_floor']:8.2f} x")
    w("   The floor is what two identical MC runs differ by, so it is the")
    w("   best any method could score here.  A ratio of 1 means the model's")
    w("   error is indistinguishable from statistical noise.")
    w("")
    d = np.abs(acc["gmc"] - acc["mc"]) / np.maximum(acc["mc"], 1e-30)
    w(f"per-cell relative error   median {np.median(d)*100:6.2f} %   "
      f"90th pct {np.percentile(d, 90)*100:6.2f} %   max {d.max()*100:6.2f} %")
    w("")

    w("-" * 74)
    w("2. COST BREAKDOWN AT THE PHYSICAL SCALE (scale = 1)")
    w("-" * 74)
    tm, tg = acc["t_mc"], g["wall"]
    w("MONTE CARLO")
    w(f"  wall time                        {tm:10.4f} s   "
      f"(mean of {acc['mc_reps']} repeats; one solve is too fast to time)")
    w(f"  particles                        {a['particles']:10,}")
    w(f"  scattering events                {a['scatters']:10,.0f}")
    w(f"  scatters per particle            {a['scatters_per_particle']:10.2f}")
    w(f"  fine-mesh crossings              {a['mesh_crossings']:10,.0f}")
    w(f"  time per particle                {tm/a['particles']*1e6:10.2f} us")
    w(f"  time per scattering event        {tm/max(a['scatters'],1)*1e9:10.1f} ns")
    w("")
    w("GMC")
    w(f"  wall time                        {tg:10.3f} s")
    w(f"  particles                        {g['particles']:10,}")
    w(f"  macro-cell crossings             {g['crossings']:10,}")
    w(f"  crossings per particle           {g['crossings_per_particle']:10.2f}")
    w(f"  batched sampler calls            {g['batched_calls']:10,}")
    w(f"  mean / median live batch         {g['mean_batch']:10.0f} / "
      f"{g['median_batch']:.0f}")
    w(f"  crossings reaching the network   {g['to_network']:10,}  "
      f"({g['to_network']/max(g['crossings'],1)*100:.1f}% -- the rest took")
    w( "                                              the analytic uncollided branch)")
    w(f"  network evaluations (NFE)        {g['nfe']:10,}")
    w(f"  time per particle                {tg/g['particles']*1e6:10.2f} us")
    w(f"  time per macro-cell crossing     {tg/max(g['crossings'],1)*1e6:10.2f} us")
    w(f"  time per NFE (one sample, one    "
      f"{g['wall_sampler']/max(g['nfe'],1)*1e6:10.2f} us")
    w( "    network evaluation)")
    w(f"  NFE per second                   "
      f"{g['nfe']/max(g['wall_sampler'],1e-9):10,.0f}")
    w( "    An NFE here is one sample through the network once.  The rate")
    w( "    depends strongly on batch size -- see the live-batch numbers")
    w( "    above; small batches are launch-bound, not arithmetic-bound.")
    w("")
    w("  where GMC's wall time goes")
    w(f"    network + decode (sampler)     {g['wall_sampler']:10.3f} s  "
      f"({g['wall_sampler']/tg*100:5.1f} %)")
    w(f"    analog MC in the birth cell    {g['wall_birth']:10.3f} s  "
      f"({g['wall_birth']/tg*100:5.1f} %)")
    w(f"    geometry / tallies (NumPy)     {g['wall_overhead']:10.3f} s  "
      f"({g['wall_overhead']/tg*100:5.1f} %)")
    w("")
    w(f"OVERALL SPEEDUP AT SCALE 1         {tm/tg:10.4g} x   "
      f"(GMC is {tg/tm:,.0f}x SLOWER)" if tg > tm else
      f"OVERALL SPEEDUP AT SCALE 1         {tm/tg:10.4g} x")
    w("")

    # ---- the break-even arithmetic ------------------------------------
    w("-" * 74)
    w("3. WHY THE SPEEDUP IS WHAT IT IS")
    w("-" * 74)
    t_scat = tm / max(a["scatters"], 1)
    t_nfe = g["wall_sampler"] / max(g["nfe"], 1)
    cross = max(g["crossings"], 1)
    gmc_net_per_cross = g["wall_sampler"] / cross
    gmc_host_per_cross = (g["wall_overhead"] + g["wall_birth"]) / cross
    gmc_per_cross = tg / cross
    scat_per_cross = a["scatters"] / cross
    w("MC does one unit of work per scattering event.  GMC does a fixed")
    w("amount of work per cell crossing, no matter how many scatters it")
    w("replaces.  So the comparison is: what does one crossing cost each way?")
    w("")
    w(f"  MC   per scattering event        {t_scat*1e9:10.1f} ns")
    w(f"       scatters per macro crossing {scat_per_cross:10.2f}")
    w(f"       => per crossing             {scat_per_cross*t_scat*1e6:10.2f} us")
    w("")
    w(f"  GMC  per NFE                     {t_nfe*1e6:10.2f} us")
    w(f"       NFE per sample              {sampler.nfe_per_sample:10d}")
    w(f"       => network per crossing     {gmc_net_per_cross*1e6:10.2f} us")
    w(f"       + host (geometry, tallies)  {gmc_host_per_cross*1e6:10.2f} us")
    w(f"       => per crossing             {gmc_per_cross*1e6:10.2f} us")
    w("")
    be = gmc_per_cross / t_scat
    w(f"BREAK-EVEN: GMC's cost per crossing equals {be:,.0f} scattering events.")
    w(f"            This geometry currently has {scat_per_cross:.1f} per crossing.")
    if scat_per_cross > 0:
        w(f"            So cells must be about {be/scat_per_cross:,.0f}x optically")
        w( "            thicker before the sampler breaks even.")
    w("")
    w("Read that with the wall-time split above.  Two different problems can")
    w("hide behind a poor speedup:")
    w("  * if 'network' dominates, the model is too expensive -- fewer ODE")
    w("    steps, a cheaper solver, or distillation would help;")
    w("  * if 'geometry / tallies' dominates, the network is not the problem")
    w("    at all and no amount of model work will fix it -- the host loop is")
    w("    single-threaded NumPy against a numba MC that uses every core.")
    w("")
    w("Also note what MC is being timed against here: mc2d is compiled with")
    w("numba and runs multi-threaded, so 'time per scattering event' above is")
    w("already a hard target.  A slower MC baseline would flatter GMC without")
    w("changing anything real.")
    w("")

    w("-" * 74)
    w("4. SPEED SWEEP (same geometry, cross sections scaled)")
    w("-" * 74)
    w(f"particles per solve   {speed_n:,}")
    w("")
    hdr = (f"{'scale':>6} {'W_bg':>7} {'MC (s)':>9} {'GMC (s)':>9} "
           f"{'speedup':>8} {'scat/part':>10} {'scat/cross':>11} "
           f"{'net %':>7} {'host %':>7}")
    w(hdr)
    w("-" * len(hdr))
    for r in rows:
        gg, mm = r["gmc"], r["mc"]
        spc = mm["scatters"] / max(gg["crossings"], 1)
        w(f"{r['scale']:>6g} {r['W_bg']:>7.1f} {r['t_mc']:>9.2f} "
          f"{gg['wall']:>9.2f} {r['speedup']:>7.4g}x "
          f"{mm['scatters_per_particle']:>10.1f} {spc:>11.1f} "
          f"{gg['wall_sampler']/gg['wall']*100:>6.1f}% "
          f"{(gg['wall_overhead']+gg['wall_birth'])/gg['wall']*100:>6.1f}%")
    w("")
    sp = np.array([r["speedup"] for r in rows])
    Wb = np.array([r["W_bg"] for r in rows])
    if (sp > 1).any() and (sp < 1).any():
        i = int(np.argmax(sp > 1))
        x0, x1, y0, y1 = Wb[i-1], Wb[i], sp[i-1], sp[i]
        xc = x0 * (x1/x0) ** ((1 - y0) / (y1 - y0)) if y1 != y0 else x1
        w(f"CROSSOVER: GMC overtakes MC at about {xc:.1f} mfp per cell")
        w(f"           (interpolated between {x0:g} and {x1:g} mfp)")
    elif (sp > 1).all():
        w(f"GMC is faster at every scale tested (from {Wb.min():g} mfp).")
    else:
        w(f"GMC is slower at every scale tested (up to {Wb.max():g} mfp).")
        w(f"Best speedup {sp.max():.2f}x at {Wb[int(np.argmax(sp))]:g} mfp; "
          f"extrapolate with the break-even figure in section 3.")
    w("")
    w("=" * 74)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/boundary")
    ap.add_argument("--n", type=int, default=20000,
                    help="particles for the accuracy solve")
    ap.add_argument("--speed-n", type=int, default=4000,
                    help="particles per point in the speed sweep")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 4.0, 10.0, 20.0])
    ap.add_argument("--ode-steps", type=int, default=5)
    ap.add_argument("--solver", default="heun",
                    choices=["euler", "heun", "rk4"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(describe(dev))
    sampler = load_sampler(ROOT / args.ckpt, args.ode_steps, args.solver, dev)
    print(f"checkpoint {args.ckpt}: {args.solver}, {args.ode_steps} steps "
          f"= {sampler.nfe_per_sample} NFE per sample\n")

    (ROOT / "figures").mkdir(exist_ok=True)
    print(f"1. accuracy  ({args.n:,} particles per solve)")
    acc = accuracy(sampler, args.n, seed=args.seed)
    print(f"   GMC error {acc['err_model']*100:.3f} %   "
          f"MC-vs-MC floor {acc['err_floor']*100:.3f} %   "
          f"ratio {acc['err_model']/acc['err_floor']:.2f}x")
    figure_accuracy(acc, ROOT / "figures" / "accuracy.pdf")
    print("   wrote figures/accuracy.pdf")
    figure_geometry(acc["mc_fine"], acc["mc"], ROOT / "figures" / "geometry.pdf")
    print("   wrote figures/geometry.pdf")

    rows = []
    if not args.skip_speed:
        print(f"\n2. speed sweep  ({args.speed_n:,} particles per point)")
        rows = speed(sampler, args.speed_n, args.scales, seed=args.seed + 10)
        figure_speed(rows, ROOT / "figures" / "speed.pdf")
        print("   wrote figures/speed.pdf")

    txt = write_report(ROOT / "results" / "evaluation.txt", acc, rows,
                       sampler, dev, args.n, args.speed_n)
    print("\n   wrote results/evaluation.txt\n")
    print(txt)


if __name__ == "__main__":
    main()
