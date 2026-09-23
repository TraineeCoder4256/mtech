#!/usr/bin/env python3
"""Train (if needed) and evaluate ANY registered model.  Run: python run.py <model>

    python run.py cfm            the flow-matching model
    python run.py oracle         exact Monte Carlo walks  (accuracy goalpost)
    python run.py free           a draw that costs nothing (cost goalpost)
    python run.py cfm train      retrain even if a checkpoint exists

Every model goes through exactly the same steps, which is the point:

  1. DATA      data/cells.npz, one split, one normaliser -- the same for
               every model.  Its SHA-256 is stored with each checkpoint and
               checked on load, so a model trained on an older dataset
               cannot be silently scored as if it were current.
  2. MODEL     load models/<name>/, or train it there if it does not exist.
  3. CELLS     exit distributions vs MC, marginals and joint (metrics.py).
  4. LATTICE   benchmark.N particles at every scale in benchmark.SCALES,
               scored against the frozen reference in reference/.
  5. SPEED     the same sweep evaluate.py ran, MC and GMC timed on this
               machine in this process.

Writes results/<name>/<timestamp>/:
    metrics.json     every number; compare.py reads only this
    report.txt       the same numbers for a human
    provenance.json  commit, model settings, dataset hash, reference hash,
                     hardware and library versions
    fields.npz       the GMC macro-cell fields, so any metric can be
                     recomputed later without rerunning the model
    accuracy.pdf, speed.pdf, scales.pdf
"""
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import numba
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import benchmark as B
import metrics as M
import mc
from data import load_dataset, save_norm, load_norm
from generators import REGISTRY
from sampler import CellSampler
from solve import gmc_solve

# ---- settings ----------------------------------------------------------
DATA = pathlib.Path("data/cells.npz")
MODELS = pathlib.Path("models")
RESULTS = pathlib.Path("results")
VAL_FRAC, SPLIT_SEED = 0.05, 0          # the split every model trains on
TIME_CAP = 3 * 3600                     # training budget, seconds, per model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_TIME = 0.25                         # repeat MC until it has run this long
# sampling-time settings per model (training settings live in the model file)
SETTINGS = {"cfm": {"steps": 5, "solver": "heun"}}


# ------------------------------------------------------------ 1-2. model
def get_model(name, ds, data_hash, retrain):
    cls = REGISTRY[name]
    info = {"checkpoint": None, "train": None, "warnings": []}
    if not cls.trainable:
        return cls(ds["ynorm"]), info

    ckpt = MODELS / name
    if (name == "cfm" and not (ckpt / "model.pt").exists()
            and (MODELS / "model.pt").exists() and not retrain):
        ckpt = MODELS                     # written by train.py before the pipeline
        info["warnings"].append("legacy checkpoint in models/ (from train.py)")

    if retrain or not (ckpt / "model.pt").exists():
        ckpt = MODELS / name
        print(f"training {name} into {ckpt}/ (cap {TIME_CAP / 3600:.1f} h)")
        gen = cls(device=DEVICE, **SETTINGS.get(name, {}))
        facts = gen.fit(ds, ckpt, TIME_CAP)
        gen.save(ckpt)
        save_norm(ckpt / "norm.json", ds["ynorm"], ds["cnorm"], gen.config)
        facts["data_sha256"] = data_hash
        facts["config"] = gen.config
        (ckpt / "train_info.json").write_text(json.dumps(facts, indent=1))

    gen = cls.load(ckpt, device=DEVICE, **SETTINGS.get(name, {}))
    yn, cn, _ = load_norm(ckpt / "norm.json")
    if not (np.array_equal(yn.mean, ds["ynorm"].mean) and np.array_equal(yn.std, ds["ynorm"].std)
            and np.array_equal(cn.mean, ds["cnorm"].mean) and np.array_equal(cn.std, ds["cnorm"].std)):
        sys.exit(f"{ckpt}/norm.json does not match the current dataset's normaliser: "
                 f"the model was trained on different data or a different split. "
                 f"Retrain with: python run.py {name} train")
    ti = ckpt / "train_info.json"
    if ti.exists():
        info["train"] = json.loads(ti.read_text())
        if info["train"].get("data_sha256") != data_hash:
            sys.exit(f"{ckpt} was trained on a different data/cells.npz "
                     f"(hash mismatch). Retrain with: python run.py {name} train")
    else:
        info["warnings"].append("no train_info.json: dataset hash not recorded at "
                                "training time; normaliser match checked instead")
    info["checkpoint"] = str(ckpt)
    return gen, info


# ------------------------------------------------------------- timing MC
def timed_mc(problem, n, seed):
    """evaluate.timed_mc: repeat until the timing is meaningful.  Only its
    TIME is used here; accuracy is scored against the frozen reference."""
    mc.solve(problem, 200, seed=0)
    reps, t0, elapsed = 0, time.perf_counter(), 0.0
    while elapsed < MIN_TIME:
        phi, stats = mc.solve(problem, n, seed=seed + reps)
        reps += 1
        elapsed = time.perf_counter() - t0
    return phi, stats, elapsed / reps


# --------------------------------------------------------------- figures
def figure_accuracy(prob, ref, gmc, acc, name, path):
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.0))
    L = prob["L"]
    ext = [0, L, 0, L]
    ax[0].imshow((prob["sig_a"] > 0).astype(float), origin="lower", extent=ext,
                 cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    x0, y0, x1, y1 = prob["source"]
    ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  facecolor="#D64545", edgecolor="k"))
    ax[0].set_title("the geometry\ngrey = absorber, red = source", fontsize=10)
    lo, hi = np.log10(max(ref[ref > 0].min(), 1e-10)), np.log10(ref.max())
    for a, F, ttl in ((ax[1], ref, "reference (mc.py, frozen)"),
                      (ax[2], gmc, f"GMC flux, {name}")):
        im = a.imshow(np.log10(np.maximum(F, 1e-10)), origin="lower", extent=ext,
                      cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        a.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=a, fraction=.046, label=r"$\log_{10}\phi$")
    d = (gmc - ref) / np.maximum(ref, 1e-30) * 100
    v = np.percentile(np.abs(d), 98)
    im = ax[3].imshow(d, origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v,
                      interpolation="nearest")
    ax[3].set_title(f"(GMC $-$ ref) / ref\nrelative $L_2$ {acc['error']*100:.2f}%, "
                    f"noise floor {acc['floor']*100:.2f}%", fontsize=10)
    fig.colorbar(im, ax=ax[3], fraction=.046, label="%")
    for a in ax:
        a.set_xlabel("x (cm)")
        a.set_ylabel("y (cm)")
    fig.suptitle(f"{name}: accuracy on the lattice, {B.N:,} particles per solve "
                 f"(the model never saw this geometry in training)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path)
    plt.close(fig)


def figure_speed(rows, name, path):
    W = np.array([r["W"] for r in rows])
    tm = np.array([r["t_mc"] for r in rows])
    tg = np.array([r["gmc"]["wall"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].loglog(W, tm, "k-o", label="Monte Carlo")
    ax[0].loglog(W, tg, "r-^", label=f"GMC ({name})")
    ax[0].set_ylabel("wall time (s)")
    ax[0].set_title("cost vs optical thickness", fontsize=10)
    ax[0].legend(fontsize=9)
    ax[1].loglog(W, tm / tg, "b-o")
    ax[1].axhline(1.0, color="k", ls="--", lw=1.2)
    ax[1].text(W[0], 1.1, "break-even", fontsize=8)
    ax[1].set_ylabel("MC time / GMC time")
    ax[1].set_title("speedup\nabove 1 the sampler wins", fontsize=10)
    for a in ax:
        a.set_xlabel("cell optical width (mfp)")
        a.grid(alpha=.3, which="both")
    fig.suptitle(f"{name}: speed on the lattice, cross sections scaled", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path)
    plt.close(fig)


def figure_scales(acc, name, path):
    s = sorted(acc)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.loglog(s, [acc[k]["error"] * 100 for k in s], "r-^", label=f"{name} error")
    ax.loglog(s, [acc[k]["floor"] * 100 for k in s], "k--o", label="noise floor (exact method)")
    ax.set_xlabel("cross-section scale")
    ax.set_ylabel(r"relative $L_2$ error vs reference (%)")
    ax.set_title(f"accuracy across thickness, {B.N:,} particles", fontsize=10)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------- provenance
def provenance(name, gen, model_info, data_hash):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                    if l.startswith("model name")), cpu)
    man = B.REFERENCE / "manifest.json"
    return {"model": gen.describe(), "command": " ".join(sys.argv),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "data_sha256": data_hash, **model_info,
            "reference_manifest_sha256": B.sha256(man) if man.exists() else None,
            "protocol": {"scales": B.SCALES, "N": B.N, "speed_n": B.SPEED_N,
                         "seed": B.SEED, "gmc_seeds": list(B.GMC_SEEDS),
                         "cell_n": M.CELL_N, "cell_shapes": M.CELL_SHAPES},
            "hardware": {"cpu": cpu, "cores": os.cpu_count(), "device": DEVICE,
                         "torch_threads": torch.get_num_threads(),
                         "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "numba": numba.__version__, "torch": torch.__version__}}


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ---------------------------------------------------------------- report
def speed_verdict(rows):
    """evaluate.py's crossover rule, unchanged."""
    sp = np.array([r["t_mc"] / r["gmc"]["wall"] for r in rows])
    Wb = np.array([r["W"] for r in rows])
    if (sp > 1).any() and (sp < 1).any():
        i = int(np.argmax(sp > 1))
        xc = Wb[i-1] * (Wb[i] / Wb[i-1]) ** ((1 - sp[i-1]) / (sp[i] - sp[i-1]))
        return f"CROSSOVER: GMC overtakes MC at about {xc:.1f} mfp per cell", float(xc)
    if (sp > 1).all():
        return f"GMC is faster at every scale tested (from {Wb.min():g} mfp)", float(Wb.min())
    return (f"GMC is SLOWER at every scale tested (up to {Wb.max():g} mfp).\n"
            f"Best {sp.max():.3g}x at {Wb[int(np.argmax(sp))]:g} mfp; "
            f"extrapolate with the break-even figure in section 4."), None


def cost_breakdown(t_mc, m, g):
    """Section 4 of evaluate.py as numbers: what one crossing costs each way."""
    cross = max(g["crossings"], 1)
    t_scatter = t_mc / max(m["scatters"], 1)
    per_cross = g["wall"] / cross
    scat_per_cross = m["scatters"] / cross
    return {"t_mc": t_mc, "t_gmc": g["wall"], "speedup": t_mc / g["wall"],
            "t_per_scatter": t_scatter, "t_per_crossing_gmc": per_cross,
            "t_per_nfe": g["wall_net"] / max(g["nfe"], 1),
            "scatters_per_crossing": scat_per_cross,
            "breakeven_scatters": per_cross / t_scatter,
            "breakeven_thicker": per_cross / t_scatter / max(scat_per_cross, 1e-9),
            "clamped_frac": g["clamped"] / max(g["to_network"], 1),
            "net_frac": g["wall_net"] / g["wall"]}


def report(name, gen, info, cells, lattice, cost, m, g, rows, verdict, extra):
    cell_tbl = "\n".join(
        f"{c['W']:>5.1f} {c['H']:>5.1f} "
        + "  ".join(f"{c[k][0]:>5.2f}/{c[k][1]:>4.2f}" for k in ("p", "Ox", "log s"))
        + f"   {c['sliced_w1'][0]:>5.2f}"
        + f"   {c['c2st_model'][0]*100:>5.1f} ({c['c2st_model'][1]:>+5.1f})"
        + f"  {c['c2st_floor'][0]*100:>5.1f} ({c['c2st_floor'][1]:>+5.1f})"
        + f"   {c['uncollided_mc']*100:>5.1f} {c['uncollided_gmc']*100:>5.1f}"
        for c in cells)
    lat_tbl = "\n".join(
        f"{s:>6g} {a['error']*100:>8.3f} {a['floor']*100:>8.3f} {a['ratio']:>7.2f}x "
        f"{a['bias']*100:>8.3f} {a['bias_noise_only']*100:>8.3f} {a['cell_chi2']:>9.2f} "
        f"{a['flux_ratio']:>8.4f} {a['cell_rel_median']*100:>7.2f} {a['cell_rel_max']*100:>8.1f}"
        for s, a in lattice.items())
    sweep = "\n".join(
        f"{r['scale']:>6g} {r['W']:>7.1f} {r['t_mc']:>9.3f} "
        f"{r['gmc']['wall']:>9.2f} {r['t_mc']/r['gmc']['wall']:>9.4g}x "
        f"{r['mc']['scatters_per_particle']:>10.1f} "
        f"{r['mc']['scatters']/max(r['gmc']['crossings'],1):>11.1f} "
        f"{r['gmc']['wall_net']/r['gmc']['wall']*100:>6.1f}%"
        for r in rows)
    a1 = lattice[B.SCALES[0]]
    notes = "\n".join(f"  ! {w}" for w in info["warnings"] + extra.get("notes", []))
    tr = info.get("train") or {}
    train_line = (f"trained     {tr.get('steps_done', '?')} steps in "
                  f"{tr.get('train_seconds', float('nan')) / 60:.1f} min"
                  + (" (stopped by the time cap)" if tr.get("steps_done", 0) < tr.get("config", {}).get("steps", 0)
                     else "")
                  if tr else "trained     n/a (not trainable, or trained before the pipeline)")
    d = gen.describe()
    return f"""\
======================================================================
{name.upper()} -- RAW NUMBERS
======================================================================
model       {json.dumps(d)}
{train_line}
device      {DEVICE}
cost class  {gen.nfe} network evaluations per collided sample
{notes}

----------------------------------------------------------------------
1. CELL PHYSICS: sampled exit distributions vs Monte Carlo
----------------------------------------------------------------------
Marginals: Wasserstein-1 in units of the quantity's own spread, as
model / MC-vs-MC floor (the same numbers evaluate.py prints).
Joint: sliced W1 over {M.N_DIRS} random directions of the whole exit state
(position, direction, log path length), as a ratio to the MC floor, so
1.00 = as close to MC as another MC run.  C2ST: a small classifier's
held-out accuracy at telling model from MC (50% = cannot tell), with its
z-score against 50%; the MC-vs-MC column is the same test between two MC
runs, i.e. what "indistinguishable" looks like at this sample size.

    W     H   p model/floor  Ox model/floor  log s m/f   sliced   C2ST model (z)  MC-vs-MC (z)   unc% MC GMC
{cell_tbl}

----------------------------------------------------------------------
2. FULL PROBLEM vs the frozen reference ({B.N:,} particles per solve)
----------------------------------------------------------------------
Reference: reference/lattice_x*.npz (mc.py, {extra['ref_histories']}),
checked against OpenMC.  Error = mean over {len(B.GMC_SEEDS)} GMC solves of the relative
L2 difference of the 7x7 cell fluxes.  Floor = the same number for
{extra['floor_k']} independent mc.py solves at the same particle count, i.e.
what an exact method scores.  Ratio 1.00 = as good as Monte Carlo.
Bias = the {len(B.GMC_SEEDS)} GMC fields averaged; "noise only" is what that
average would score if the model were exact.  cell chi2 ~ 1 means every
cell is right to within statistics; >> 1 means systematic error.

 scale  error %  floor %   ratio   bias %  noise %  cell chi2  flux GMC/ref  med %  worst %
{lat_tbl}

At the published scale: floor range {a1['floor_lo']*100:.2f}-{a1['floor_hi']*100:.2f} %, \
GMC-vs-GMC {a1.get('gmc_pairwise', float('nan'))*100:.3f} %
(vs MC-vs-MC {a1['floor_pairwise']*100:.3f} %: equal = neither over- nor under-dispersed).
Path lengths clamped to the straight-line minimum: {cost['clamped_frac']*100:.2f} %

----------------------------------------------------------------------
3. COST AT THE PUBLISHED SCALE (same particle count, this machine)
----------------------------------------------------------------------
MONTE CARLO
  wall time                     {cost['t_mc']:10.4f} s   (mean of repeated runs)
  scattering events             {m['scatters']:10,.0f}
  scatters per particle         {m['scatters_per_particle']:10.2f}
  time per scattering event     {cost['t_per_scatter']*1e9:10.1f} ns

GMC
  wall time                     {g['wall']:10.4f} s
  macro-cell crossings          {g['crossings']:10,}
  crossings per particle        {g['crossings_per_particle']:10.2f}
  batched sampler calls         {g['calls']:10,}
  crossings reaching the model  {g['to_network']:10,}   \
({g['to_network']/max(g['crossings'],1)*100:.1f} %; the rest were uncollided)
  network evaluations           {g['nfe']:10,}
  time per macro-cell crossing  {cost['t_per_crossing_gmc']*1e6:10.2f} us
  where the wall time goes
    model                       {g['wall_net']:10.4f} s  ({g['wall_net']/g['wall']*100:5.1f} %)
    analog MC in the birth cell {g['wall_birth']:10.4f} s  ({g['wall_birth']/g['wall']*100:5.1f} %)
    geometry / tallies (NumPy)  {g['wall_host']:10.4f} s  ({g['wall_host']/g['wall']*100:5.1f} %)

SPEEDUP                         {cost['speedup']:10.4g} x

----------------------------------------------------------------------
4. BREAK-EVEN
----------------------------------------------------------------------
  MC   {cost['scatters_per_crossing']:.2f} scatters x {cost['t_per_scatter']*1e9:.1f} ns \
= {cost['scatters_per_crossing']*cost['t_per_scatter']*1e6:.2f} us per crossing
  GMC  {cost['t_per_crossing_gmc']*1e6:.2f} us per crossing

One GMC crossing costs as much as {cost['breakeven_scatters']:,.0f} scattering events, and this
geometry has {cost['scatters_per_crossing']:.1f} per crossing, so cells must be
~{cost['breakeven_thicker']:,.0f}x optically thicker before GMC wins.

----------------------------------------------------------------------
5. SPEED SWEEP ({B.SPEED_N:,} particles per point)
----------------------------------------------------------------------
 scale    W_bg    MC (s)   GMC (s)   speedup  scat/part  scat/cross  model%
--------------------------------------------------------------------------
{sweep}

{verdict}
======================================================================
"""


# ------------------------------------------------------------------ main
def main():
    args = sys.argv[1:]
    if not args or args[0] not in REGISTRY or set(args[1:]) - {"train"}:
        sys.exit(f"usage: python run.py <{'|'.join(REGISTRY)}> [train]")
    name, retrain = args[0], "train" in args[1:]

    print(f"data  {DATA}")
    data_hash = B.sha256(DATA)
    torch.manual_seed(SPLIT_SEED)
    ds = load_dataset(DATA, VAL_FRAC, SPLIT_SEED)
    gen, info = get_model(name, ds, data_hash, retrain)
    sampler = CellSampler(gen, ds["ynorm"], ds["cnorm"])
    del ds
    print(f"model {json.dumps(gen.describe())}")
    for w in info["warnings"]:
        print(f"  ! {w}")

    out = RESULTS / name / time.strftime("%Y-%m-%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(B.SEED)
    extra = {"notes": []}
    if name == "free":
        extra["notes"].append("free draws random noise: its ACCURACY numbers are "
                              "meaningless by design; only its cost is a reference")

    print("\n1. cell physics (marginals and joint)")
    cells = M.cell_check(sampler, rng)
    for c in cells:
        print(f"   {c['W']:>4.1f} x {c['H']:<4.1f}  p {c['p'][0]:.2f}/{c['p'][1]:.2f}   "
              f"Ox {c['Ox'][0]:.2f}/{c['Ox'][1]:.2f}   log s {c['log s'][0]:.2f}/"
              f"{c['log s'][1]:.2f}   sliced {c['sliced_w1'][0]:.2f}   "
              f"C2ST {c['c2st_model'][0]*100:.1f}% (floor {c['c2st_floor'][0]*100:.1f}%)",
              flush=True)

    print(f"\n2. full problem vs the frozen reference ({B.N:,} particles)")
    lattice, fields, refs, g_stats = {}, {}, {}, {}
    for sc in B.SCALES:
        ref = B.load_reference(sc)
        if int(ref["floor_n"]) != B.N:
            sys.exit(f"reference noise floor was measured at {int(ref['floor_n']):,} "
                     f"particles but benchmark.N is {B.N:,}: rebuild with make_reference.py")
        refs[sc] = ref
        prob = B.problem(sc)
        GMCs = []
        for s in B.GMC_SEEDS:
            f, g_stats[sc] = gmc_solve(prob, B.N, sampler, seed=s)
            GMCs.append(f)
        fields[f"gmc_x{sc:g}"] = np.array(GMCs)
        lattice[sc] = M.lattice_accuracy(GMCs, ref)
        a = lattice[sc]
        print(f"   scale {sc:>3g}  error {a['error']*100:7.3f} %  floor {a['floor']*100:6.3f} %  "
              f"ratio {a['ratio']:6.2f}x  bias {a['bias']*100:6.3f} %  "
              f"cell chi2 {a['cell_chi2']:7.2f}", flush=True)
    s1 = B.SCALES[0]
    extra["ref_histories"] = f"{int(refs[s1]['mc_histories']):,} histories at scale {s1:g}"
    extra["floor_k"] = len(refs[s1]["floor_cells"])
    figure_accuracy(B.problem(s1), refs[s1]["mc_cell"], fields[f"gmc_x{s1:g}"][0],
                    lattice[s1], name, out / "accuracy.pdf")
    figure_scales(lattice, name, out / "scales.pdf")

    print(f"\n3. cost at the published scale ({B.N:,} particles)")
    _, m_stats, t_mc = timed_mc(B.problem(s1), B.N, B.SEED)
    cost = cost_breakdown(t_mc, m_stats, g_stats[s1])
    print(f"   MC {t_mc:.4f} s   GMC {g_stats[s1]['wall']:.3f} s   "
          f"speedup {cost['speedup']:.4g}x   break-even {cost['breakeven_scatters']:,.0f} "
          f"scatters per crossing")

    print(f"\n4. speed sweep ({B.SPEED_N:,} particles per point)")
    rows = []
    for sc in B.SCALES:
        p = B.problem(sc)
        _, mcs, t = timed_mc(p, B.SPEED_N, B.SEED + 10)
        _, gs = gmc_solve(p, B.SPEED_N, sampler, seed=B.SEED + 13)
        rows.append({"scale": sc, "W": p["pitch"] * sc, "t_mc": t, "mc": mcs, "gmc": gs})
        print(f"   scale {sc:>3}  MC {t:8.4f}s  GMC {gs['wall']:7.2f}s  "
              f"speedup {t/gs['wall']:8.4g}x", flush=True)
    verdict, crossover = speed_verdict(rows)
    figure_speed(rows, name, out / "speed.pdf")

    if hasattr(gen, "walks"):
        extra["oracle_walks"] = gen.walks
        extra["notes"].append(f"oracle ran {gen.walks:,} Monte Carlo walks in total "
                              f"(rejected uncollided walks included)")

    text = report(name, gen, info, cells, lattice, cost, m_stats, g_stats[s1],
                  rows, verdict, extra)
    metrics = {"model": gen.describe(), "name": name, "cells": cells,
               "lattice": lattice, "cost": cost,
               "mc_stats": m_stats, "gmc_stats": g_stats,
               "speed": rows, "verdict": verdict, "crossover_mfp": crossover,
               "train": info["train"], "warnings": info["warnings"],
               "extra": {k: v for k, v in extra.items() if k != "notes"},
               "notes": extra["notes"]}
    (out / "metrics.json").write_text(json.dumps(to_json(metrics), indent=1))
    (out / "provenance.json").write_text(
        json.dumps(to_json(provenance(name, gen, info, data_hash)), indent=1))
    np.savez(out / "fields.npz", **fields)
    (out / "report.txt").write_text(text)
    print("\n" + text)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
