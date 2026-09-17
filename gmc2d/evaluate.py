#!/usr/bin/env python3
"""Step 3: evaluation.  Run: python evaluate.py

Three questions, three answers.

  DOES IT LEARN THE CELL PHYSICS?  Compare sampled exit distributions
  against Monte Carlo for several cell shapes, INCLUDING RECTANGLES, at
  entry conditions never seen in training.  Reported as a table, because
  the interesting content is a handful of numbers.

  DOES IT SOLVE A REAL PROBLEM?  Chain the sampler across the 7x7 lattice
  and compare the flux against MC.  Two MC runs with different seeds give
  the statistical floor -- the best score anything could get.
  -> figures/accuracy.pdf

  IS IT FASTER?  The same lattice with cross sections scaled up, which
  thickens the cells without changing the layout.  MC cost per cell grows
  with thickness; the sampler's does not.
  -> figures/speed.pdf

Everything numeric lands in results/evaluation.txt.
"""
import pathlib
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mc
from data import load_norm
from model import VelocityField
from sampler import Sampler
from solve import gmc_solve, macro, coarsen

# ---- settings ----------------------------------------------------------
CKPT = pathlib.Path("models")
N = 20000                            # particles for the accuracy solve
SPEED_N = 4000                       # particles per speed point
SCALES = [1, 4, 10, 20]              # cross-section multipliers
STEPS, SOLVER = 5, "heun"            # ODE steps and solver
SEED = 1
CELL_SHAPES = [(1.0, 1.0), (4.0, 4.0), (4.0, 1.0), (1.0, 4.0), (8.0, 2.0)]
CELL_N = 20000                       # samples per shape in the cell check
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_TIME = 0.25                      # repeat MC until it has run this long


def load_sampler():
    st = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True)
    c = st["config"]
    net = VelocityField(c["x_dim"], c["c_dim"], c["width"], c["depth"])
    net.load_state_dict(st["ema"])
    yn, cn, _ = load_norm(CKPT / "norm.json")
    return Sampler(net, yn, cn, DEVICE, STEPS, SOLVER)


def timed_mc(problem, n, seed):
    """Run MC, repeating until timing it is meaningful.  One 20k solve takes
    a few ms at the physical scale, which is timer noise."""
    mc.solve(problem, 200, seed=0)                  # warm the numba JIT
    reps, t0, elapsed = 0, time.perf_counter(), 0.0
    while elapsed < MIN_TIME:
        phi, stats = mc.solve(problem, n, seed=seed + reps)
        reps += 1
        elapsed = time.perf_counter() - t0
    return phi, stats, elapsed / reps


def w1(a, b):
    """Wasserstein-1 between two samples, via quantiles."""
    q = np.linspace(0, 1, 400)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


# ------------------------------------------------- 1. does it learn cells?
def cell_check(sampler, rng):
    """Exit distributions vs MC for several shapes, at unseen conditions."""
    rows = []
    for W, H in CELL_SHAPES:
        xi = rng.uniform(0.15, 0.85)
        r, th = np.sqrt(rng.uniform(0.1, 0.9)), rng.uniform(-1.0, 1.0)
        ox, oy = max(r * np.cos(th), 1e-3), r * np.sin(th)

        a = mc.sample_cell(CELL_N, W, H, xi=xi, direction=(ox, oy), seed=101)
        b = mc.sample_cell(CELL_N, W, H, xi=xi, direction=(ox, oy), seed=202)
        g = sampler.sample(np.full(CELL_N, W), np.full(CELL_N, H),
                           np.full(CELL_N, xi), np.full(CELL_N, ox),
                           np.full(CELL_N, oy), seed=303)
        out = {"W": W, "H": H, "xi": xi, "ox": ox, "oy": oy,
               "uncollided_mc": float((a["k"] == 0).mean()),
               "uncollided_gmc": float(g["uncollided"].mean())}
        for name, va, vb, vg in (
                ("p", a["p"] / (2 * (W + H)), b["p"] / (2 * (W + H)),
                 g["p"] / (2 * (W + H))),
                ("Ox", a["dir"][:, 0], b["dir"][:, 0], g["dir"][:, 0]),
                ("log s", np.log10(a["s"]), np.log10(b["s"]),
                 np.log10(g["s"]))):
            sd = va.std() + 1e-12
            out[name] = (w1(va, vg) / sd, w1(va, vb) / sd)   # model, floor
        rows.append(out)
    return rows


# ------------------------------------------------- 2. does it solve a problem?
def figure_accuracy(prob, mc_macro, gmc_macro, err, floor, path):
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.0))
    L = prob["L"]
    ext = [0, L, 0, L]

    ax[0].imshow((prob["sig_a"] > 0).astype(float), origin="lower", extent=ext,
                 cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    x0, y0, x1, y1 = prob["source"]
    ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  facecolor="#D64545", edgecolor="k"))
    ax[0].set_title("the geometry\ngrey = absorber, red = source", fontsize=10)

    lo = np.log10(max(mc_macro[mc_macro > 0].min(), 1e-10))
    hi = np.log10(mc_macro.max())
    for a, M, ttl in ((ax[1], mc_macro, "Monte Carlo flux"),
                      (ax[2], gmc_macro, "GMC flux")):
        im = a.imshow(np.log10(np.maximum(M, 1e-10)), origin="lower",
                      extent=ext, cmap="viridis", vmin=lo, vmax=hi,
                      interpolation="nearest")
        a.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=a, fraction=.046, label=r"$\log_{10}\phi$")

    d = (gmc_macro - mc_macro) / np.maximum(mc_macro, 1e-30) * 100
    v = np.percentile(np.abs(d), 98)
    im = ax[3].imshow(d, origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-v, vmax=v, interpolation="nearest")
    ax[3].set_title(f"(GMC $-$ MC) / MC\nrelative $L_2$ {err*100:.2f}%, "
                    f"noise floor {floor*100:.2f}%", fontsize=10)
    fig.colorbar(im, ax=ax[3], fraction=.046, label="%")

    for a in ax:
        a.set_xlabel("x (cm)")
        a.set_ylabel("y (cm)")
    fig.suptitle(f"Accuracy on the lattice, {N:,} particles per solve  "
                 f"(the model never saw this geometry in training)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------- 3. is it faster?
def figure_speed(rows, path):
    W = np.array([r["W"] for r in rows])
    tm = np.array([r["t_mc"] for r in rows])
    tg = np.array([r["gmc"]["wall"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax[0].loglog(W, tm, "k-o", label="Monte Carlo")
    ax[0].loglog(W, tg, "r-^", label="GMC")
    ax[0].set_ylabel("wall time (s)")
    ax[0].set_title("cost vs optical thickness\nMC grows, GMC is flat",
                    fontsize=10)
    ax[0].legend(fontsize=9)

    ax[1].loglog(W, tm / tg, "b-o")
    ax[1].axhline(1.0, color="k", ls="--", lw=1.2)
    ax[1].text(W[0], 1.1, "break-even", fontsize=8)
    ax[1].set_ylabel("MC time / GMC time")
    ax[1].set_title("speedup\nabove 1 the sampler wins", fontsize=10)

    for a in ax:
        a.set_xlabel("cell optical width (mfp)")
        a.grid(alpha=.3, which="both")
    fig.suptitle("Speed on the same lattice, cross sections scaled",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ report
def report(cells, acc, rows, sampler, path):
    m, g = acc["mc_stats"], acc["gmc_stats"]
    tm, tg = acc["t_mc"], g["wall"]
    cross = max(g["crossings"], 1)
    t_scatter = tm / max(m["scatters"], 1)
    t_nfe = g["wall_net"] / max(g["nfe"], 1)
    per_cross_gmc = tg / cross
    scat_per_cross = m["scatters"] / cross
    breakeven = per_cross_gmc / t_scatter
    relerr = np.abs(acc["gmc"] - acc["mc"]) / np.maximum(acc["mc"], 1e-30)

    cell_tbl = "\n".join(
        f"{c['W']:>5.1f} {c['H']:>5.1f} {c['W']/c['H']:>7.2f} "
        + "  ".join(f"{c[k][0]:>6.2f} /{c[k][1]:>5.2f}"
                    for k in ("p", "Ox", "log s"))
        + f"   {c['uncollided_mc']*100:>5.1f} {c['uncollided_gmc']*100:>6.1f}"
        for c in cells)

    sweep = "\n".join(
        f"{r['scale']:>6g} {r['W']:>7.1f} {r['t_mc']:>9.3f} "
        f"{r['gmc']['wall']:>9.2f} {r['t_mc']/r['gmc']['wall']:>9.4g}x "
        f"{r['mc']['scatters_per_particle']:>10.1f} "
        f"{r['mc']['scatters']/max(r['gmc']['crossings'],1):>11.1f} "
        f"{r['gmc']['wall_net']/r['gmc']['wall']*100:>6.1f}%"
        for r in rows)

    sp = np.array([r["t_mc"] / r["gmc"]["wall"] for r in rows])
    Wb = np.array([r["W"] for r in rows])
    if (sp > 1).any() and (sp < 1).any():
        i = int(np.argmax(sp > 1))
        xc = Wb[i-1] * (Wb[i] / Wb[i-1]) ** ((1 - sp[i-1]) / (sp[i] - sp[i-1]))
        verdict = f"CROSSOVER: GMC overtakes MC at about {xc:.1f} mfp per cell"
    elif (sp > 1).all():
        verdict = f"GMC is faster at every scale tested (from {Wb.min():g} mfp)"
    else:
        verdict = (f"GMC is SLOWER at every scale tested (up to {Wb.max():g} mfp).\n"
                   f"Best {sp.max():.3g}x at {Wb[int(np.argmax(sp))]:g} mfp; "
                   f"extrapolate with the break-even figure below.")

    text = f"""\
======================================================================
EVALUATION -- RAW NUMBERS
======================================================================
device      {DEVICE}
solver      {SOLVER}, {STEPS} steps = {sampler.nfe_per_sample} network evaluations per sample

----------------------------------------------------------------------
1. CELL PHYSICS: sampled exit distributions vs Monte Carlo
----------------------------------------------------------------------
Wasserstein-1 distance in units of the quantity's own spread, shown as
model / MC-vs-MC floor.  The floor is what two MC runs differ by at the
same sample size, so 1.00 means indistinguishable from noise.  Entry
conditions are drawn fresh and were not in training.

    W     H  aspect     p (model/floor)  Ox (model/floor)  log s (model/floor)  unc% MC/GMC
{cell_tbl}

Rectangles matter: the model is conditioned on W and H separately, as in
the paper, so a 4x1 cell is not a special case bolted on -- it is drawn
from the same distribution the model was trained over.

----------------------------------------------------------------------
2. FULL PROBLEM: the 7x7 lattice ({N:,} particles per solve)
----------------------------------------------------------------------
GMC relative L2 error vs MC        {acc['err']*100:8.3f} %   (mean of 2 runs)
MC-vs-MC noise floor               {acc['floor']*100:8.3f} %   \
(3 pairs, range {acc['floor_lo']*100:.2f}-{acc['floor_hi']*100:.2f})
ratio to the floor                 {acc['err']/acc['floor']:8.2f} x
GMC-vs-GMC noise                   {acc['gmc_noise']*100:8.3f} %   \
(compare with the MC floor: equal means the sampler is neither
                                              over- nor under-dispersed)
systematic bias, noise averaged out {acc['bias']*100:7.3f} %   \
(mean of 2 GMC fields vs mean of 3 MC fields)
total flux, GMC / MC               {acc['flux_ratio']:8.4f}     (1.0 = mass conserved)

The floor is quoted as a range because it is itself a random quantity: two
MC runs at this particle count can differ by anything in that band purely
by seed.  A single-pair floor is not a yardstick, which is why three are
used.  Read the BIAS line for how wrong the model actually is; the error
line still contains the statistical scatter of both fields.

per-cell relative error   median {np.median(relerr)*100:6.2f} %   \
90th pct {np.percentile(relerr, 90)*100:6.2f} %   max {relerr.max()*100:6.2f} %
path lengths clamped to the straight-line minimum: \
{g['clamped']/max(g['to_network'],1)*100:.2f} %

----------------------------------------------------------------------
3. COST AT THE PHYSICAL SCALE
----------------------------------------------------------------------
MONTE CARLO
  wall time                     {tm:10.4f} s   (mean of repeated runs)
  scattering events             {m['scatters']:10,.0f}
  scatters per particle         {m['scatters_per_particle']:10.2f}
  time per particle             {tm/m['particles']*1e6:10.2f} us
  time per scattering event     {t_scatter*1e9:10.1f} ns

GMC
  wall time                     {tg:10.4f} s
  macro-cell crossings          {g['crossings']:10,}
  crossings per particle        {g['crossings_per_particle']:10.2f}
  batched sampler calls         {g['calls']:10,}
  mean / median live batch      {g['mean_batch']:10.0f} / {g['median_batch']:.0f}
  crossings reaching the network{g['to_network']:10,}   \
({g['to_network']/cross*100:.1f} %; the rest were uncollided)
  network evaluations           {g['nfe']:10,}
  time per particle             {tg/g['particles']*1e6:10.2f} us
  time per macro-cell crossing  {per_cross_gmc*1e6:10.2f} us
  time per network evaluation   {t_nfe*1e6:10.2f} us

  where the wall time goes
    network (ODE solve)         {g['wall_net']:10.4f} s  ({g['wall_net']/tg*100:5.1f} %)
    analog MC in the birth cell {g['wall_birth']:10.4f} s  ({g['wall_birth']/tg*100:5.1f} %)
    geometry / tallies (NumPy)  {g['wall_host']:10.4f} s  ({g['wall_host']/tg*100:5.1f} %)

SPEEDUP                         {tm/tg:10.4g} x

----------------------------------------------------------------------
4. WHY THE SPEEDUP IS WHAT IT IS
----------------------------------------------------------------------
MC does one unit of work per scattering event.  GMC does a fixed amount
per cell crossing, however many scatters that replaces.  So: what does
one crossing cost each way?

  MC   {scat_per_cross:.2f} scatters x {t_scatter*1e9:.1f} ns \
= {scat_per_cross*t_scatter*1e6:.2f} us per crossing
  GMC  {sampler.nfe_per_sample} evaluations + host \
= {per_cross_gmc*1e6:.2f} us per crossing

BREAK-EVEN: one GMC crossing costs as much as {breakeven:,.0f} scattering events,
            and this geometry has {scat_per_cross:.1f} per crossing -- so cells
            must be ~{breakeven/max(scat_per_cross,1e-9):,.0f}x optically thicker before GMC wins.

Which problem you have depends on the split above: if the network
dominates, the model is too expensive (fewer steps, cheaper solver,
distillation).  If the host loop dominates, the model is not the
bottleneck and no model work will fix it.

Note the baseline: mc.py is numba-compiled and multi-threaded, so the
per-scatter time is already a hard target.  A slower baseline would
flatter GMC without changing anything real.

----------------------------------------------------------------------
5. SPEED SWEEP ({SPEED_N:,} particles per point)
----------------------------------------------------------------------
 scale    W_bg    MC (s)   GMC (s)   speedup  scat/part  scat/cross   net%
--------------------------------------------------------------------------
{sweep}

{verdict}
======================================================================
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def main():
    print(f"device {DEVICE}")
    sampler = load_sampler()
    print(f"solver {SOLVER}, {STEPS} steps "
          f"= {sampler.nfe_per_sample} NFE per sample\n")
    pathlib.Path("figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("1. cell physics (including rectangles)")
    cells = cell_check(sampler, rng)
    for c in cells:
        print(f"   {c['W']:>4.1f} x {c['H']:<4.1f}  p {c['p'][0]:.2f}/"
              f"{c['p'][1]:.2f}   Ox {c['Ox'][0]:.2f}/{c['Ox'][1]:.2f}   "
              f"log s {c['log s'][0]:.2f}/{c['log s'][1]:.2f}  (model/floor)")

    print(f"\n2. full problem ({N:,} particles)")
    prob = mc.lattice()
    phi_a, mc_stats, t_mc = timed_mc(prob, N, SEED)

    # Several realisations of each, because the floor itself is noisy: two
    # MC runs can differ by 0.8% or 1.9% purely by seed, so a floor taken
    # from ONE pair is not a yardstick.  Averaging the fields also separates
    # the model's systematic bias from its statistical scatter.
    MCs = [coarsen(phi_a, prob["per_cm"])] + [
        coarsen(mc.solve(prob, N, seed=SEED + 1000 * i)[0], prob["per_cm"])
        for i in (1, 2)]
    GMCs, g_stats = [], None
    for i in (7, 77):
        g, g_stats = gmc_solve(prob, N, sampler, seed=SEED + i)
        GMCs.append(g)

    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    floors = [rel(MCs[i], MCs[j]) for i, j in ((1, 0), (2, 0), (2, 1))]
    errs = [rel(g, MCs[0]) for g in GMCs]
    acc = {"mc": MCs[0], "gmc": GMCs[0], "t_mc": t_mc,
           "mc_stats": mc_stats, "gmc_stats": g_stats,
           "err": float(np.mean(errs)),
           "floor": float(np.mean(floors)),
           "floor_lo": min(floors), "floor_hi": max(floors),
           "gmc_noise": rel(GMCs[1], GMCs[0]),
           "bias": rel(np.mean(GMCs, 0), np.mean(MCs, 0)),
           "flux_ratio": float(np.mean(GMCs, 0).sum() / np.mean(MCs, 0).sum())}
    print(f"   GMC {acc['err']*100:.3f} %   floor {acc['floor']*100:.3f} % "
          f"({acc['floor_lo']*100:.2f}-{acc['floor_hi']*100:.2f})   "
          f"ratio {acc['err']/acc['floor']:.2f}x   bias {acc['bias']*100:.3f} %")
    figure_accuracy(prob, MCs[0], GMCs[0], acc["err"], acc["floor"],
                    pathlib.Path("figures/accuracy.pdf"))
    print("   wrote figures/accuracy.pdf")

    print(f"\n3. speed sweep ({SPEED_N:,} particles per point)")
    rows = []
    for sc in SCALES:
        p = mc.lattice(sc)
        _, mcs, t = timed_mc(p, SPEED_N, SEED + 10)
        _, gs = gmc_solve(p, SPEED_N, sampler, seed=SEED + 13)
        rows.append({"scale": sc, "W": p["pitch"] * sc, "t_mc": t,
                     "mc": mcs, "gmc": gs})
        print(f"   scale {sc:>3}  MC {t:8.4f}s  GMC {gs['wall']:7.2f}s  "
              f"speedup {t/gs['wall']:8.4g}x", flush=True)
    figure_speed(rows, pathlib.Path("figures/speed.pdf"))
    print("   wrote figures/speed.pdf\n")

    print(report(cells, acc, rows, sampler,
                 pathlib.Path("results/evaluation.txt")))
    print("wrote results/evaluation.txt")


if __name__ == "__main__":
    main()
