#!/usr/bin/env python3
"""Step 4: where the time actually goes.  Run: python speed_profile.py

`evaluate.py` answers *is it faster*.  This file answers *what would have to
change*, by taking the two sides of that comparison apart.

Seven measurements, each aimed at one candidate explanation:

  A  arithmetic      how many floating-point operations one cell crossing
                     costs each way, before any code is run.  If this number
                     is hopeless, nothing measured later can rescue it.
  B  network scaling time per network evaluation against batch size, at one
                     thread and at all of them.  Separates fixed per-call
                     overhead from real arithmetic.
  C  host vs network the same lattice solved with the real model and with a
                     FREE one (the ODE replaced by the identity).  The second
                     time is the floor: what GMC would cost if the model were
                     instantaneous.  If that floor is already above MC, model
                     work cannot win.
  D  attribution     cProfile over one GMC solve, aggregated by function, so
                     the host loop is itemised rather than lumped.
  E  batch decay     live batch size per crossing iteration.  A sampler that
                     is efficient at 20,000 particles and called with 12 is
                     paying overhead, not arithmetic.
  F  fidelity knob   wall time AND accuracy across ODE solvers and step
                     counts.  The only speed knob that needs no retraining,
                     so its cost in accuracy has to be measured, not assumed.
  G  projection      the optical thickness at which GMC would overtake MC,
                     under the measured cost and under each intervention.

Writes results/speed_profile.txt and results/speed_profile.json.
Needs models/ from train.py.  Nothing here modifies the model or the physics.
"""
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import torch

import mc
from data import load_norm
from model import VelocityField
from sampler import Sampler, NFE_PER_STEP
from solve import gmc_solve, coarsen

CKPT = pathlib.Path("models")
OUT = pathlib.Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_LATTICE = 20000          # particles for the like-for-like solves
N_PROFILE = 4000           # particles for the cProfile run (it is ~2x slower)
BATCHES = [1, 8, 64, 512, 4096, 20000, 100000]
LADDER = [("euler", 1), ("euler", 2), ("euler", 4), ("heun", 2),
          ("heun", 5), ("rk4", 5)]
BASE_STEPS, BASE_SOLVER = 5, "heun"        # what evaluate.py uses
SEED = 1


# --------------------------------------------------------------- helpers
def load_parts():
    st = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True)
    c = st["config"]
    net = VelocityField(c["x_dim"], c["c_dim"], c["width"], c["depth"])
    net.load_state_dict(st["ema"])
    yn, cn, _ = load_norm(CKPT / "norm.json")
    return net, yn, cn, c


class FreeField(torch.nn.Module):
    """A velocity field that costs nothing.  Same shapes, no arithmetic --
    so a solve using it measures everything EXCEPT the model."""

    def __init__(self, x_dim=6):
        super().__init__()
        self.x_dim = x_dim

    def forward(self, x, t, c):
        return torch.zeros_like(x)


def timed(fn, min_time=0.3, warm=1):
    for _ in range(warm):
        fn()
    reps, t0, el = 0, time.perf_counter(), 0.0
    while el < min_time:
        fn()
        reps += 1
        el = time.perf_counter() - t0
    return el / reps


def timed_mc(problem, n, seed, min_time=0.3):
    mc.solve(problem, 200, seed=0)
    reps, t0, el = 0, time.perf_counter(), 0.0
    while el < min_time:
        phi, stats = mc.solve(problem, n, seed=seed + reps)
        reps += 1
        el = time.perf_counter() - t0
    return phi, stats, el / reps


def w1(a, b):
    q = np.linspace(0, 1, 400)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------ A. the arithmetic first
def flop_accounting(net, cfg, nfe_per_sample):
    """Multiply-accumulates in one forward pass, counted from the layers.

    A scattering event is a handful of flops: one log, one isotropic
    direction, two wall distances, a compare, an update.  Call it 30 -- the
    exact number does not matter when the other side is in the millions.
    """
    w, d, emb, t_dim = cfg["width"], cfg["depth"], 128, 64
    c_dim, x_dim = cfg["c_dim"], cfg["x_dim"]
    mac = 0
    mac += (t_dim + c_dim) * emb + emb * emb          # condition embedding
    mac += x_dim * w                                  # proj_in
    mac += d * (emb * 2 * w + w * 2 * w + 2 * w * w)  # FiLM + fc1 + fc2
    mac += w * x_dim                                  # proj_out
    return {"params": int(net.n_params()),
            "mac_per_nfe": int(mac),
            "flop_per_nfe": int(2 * mac),
            "nfe_per_sample": int(nfe_per_sample),
            "flop_per_crossing": int(2 * mac * nfe_per_sample),
            "flop_per_scatter_est": 30,
            "arithmetic_ratio": 2 * mac * nfe_per_sample / 30.0}


# --------------------------------------------- B. network vs batch size
def network_scaling(net, threads_list=(1, torch.get_num_threads())):
    net = net.eval()
    rows = []
    for nt in sorted(set(threads_list)):
        torch.set_num_threads(nt)
        for b in BATCHES:
            x = torch.randn(b, net.x_dim)
            t = torch.rand(b, 1)
            c = torch.randn(b, net.c_dim)
            with torch.no_grad():
                dt = timed(lambda: net(x, t, c), min_time=0.25)
            rows.append({"threads": nt, "batch": b, "t_call_us": dt * 1e6,
                         "t_per_sample_us": dt / b * 1e6})
    torch.set_num_threads(max(threads_list))
    return rows


def implementation_variants(net, batch=4096):
    """Same arithmetic, different execution: what is left on the table by
    the implementation alone, as opposed to by the model's size."""
    x, t, c = (torch.randn(batch, net.x_dim), torch.rand(batch, 1),
               torch.randn(batch, net.c_dim))
    rows = []

    def add(tag, fn, note=""):
        try:
            with torch.no_grad():
                dt = timed(fn, min_time=0.3, warm=3)
            rows.append({"variant": tag, "t_per_sample_us": dt / batch * 1e6,
                         "note": note})
        except Exception as e:                       # a variant may not build
            rows.append({"variant": tag, "t_per_sample_us": float("nan"),
                         "note": f"unavailable: {type(e).__name__}"})
        print(f"   {tag:<26} {rows[-1]['t_per_sample_us']:8.3f} us/sample "
              f"{rows[-1]['note']}", flush=True)

    net = net.eval()
    add("float32 (as shipped)", lambda: net(x, t, c))
    with torch.inference_mode():
        add("inference_mode", lambda: net(x, t, c))
    nb = net.to(torch.bfloat16)
    xb, tb, cb = x.bfloat16(), t.bfloat16(), c.bfloat16()
    add("bfloat16", lambda: nb(xb, tb, cb), "accuracy unchecked")
    net = net.float()
    try:
        cnet = torch.compile(net, dynamic=False)
        add("torch.compile (fixed batch)", lambda: cnet(x, t, c),
            "recompiles per batch shape")
    except Exception as e:
        rows.append({"variant": "torch.compile", "t_per_sample_us": float("nan"),
                     "note": f"unavailable: {type(e).__name__}"})
    return rows


def size_scaling(cfg, batch=4096):
    """Cost of smaller velocity fields.  UNTRAINED -- this measures what a
    given shape would cost, not what it would be worth.  Accuracy at these
    sizes is a training question and belongs to whoever owns the model."""
    rows = []
    for w, d in ((cfg["width"], cfg["depth"]), (256, 3), (128, 5), (128, 3),
                 (64, 3), (64, 2)):
        n = VelocityField(cfg["x_dim"], cfg["c_dim"], w, d).eval()
        x, t, c = (torch.randn(batch, n.x_dim), torch.rand(batch, 1),
                   torch.randn(batch, n.c_dim))
        with torch.no_grad():
            dt = timed(lambda: n(x, t, c), min_time=0.25)
        rows.append({"width": w, "depth": d, "params": int(n.n_params()),
                     "t_per_sample_us": dt / batch * 1e6})
        print(f"   width {w:>4} depth {d}  {n.n_params():>10,} params  "
              f"{rows[-1]['t_per_sample_us']:7.3f} us/sample", flush=True)
    base = rows[0]["t_per_sample_us"]
    for r in rows:
        r["cheaper_than_shipped"] = base / r["t_per_sample_us"]
    return rows


# ------------------------------------------ C. host loop vs network cost
def host_vs_network(net, yn, cn, prob, n):
    real = Sampler(net, yn, cn, DEVICE, BASE_STEPS, BASE_SOLVER)
    _, s_real = gmc_solve(prob, n, real, seed=SEED + 7)

    free = Sampler(FreeField(net.x_dim), yn, cn, DEVICE, BASE_STEPS,
                   BASE_SOLVER)
    _, s_free = gmc_solve(prob, n, free, seed=SEED + 7)

    def pack(s, tag):
        cr = max(s["crossings"], 1)
        return {"variant": tag, "wall": s["wall"], "wall_net": s["wall_net"],
                "wall_birth": s["wall_birth"], "wall_host": s["wall_host"],
                "crossings": s["crossings"], "calls": s["calls"],
                "us_per_crossing": s["wall"] / cr * 1e6,
                "us_per_crossing_host": s["wall_host"] / cr * 1e6,
                "us_per_crossing_net": s["wall_net"] / cr * 1e6}
    return [pack(s_real, "real model"), pack(s_free, "free model")]


# ------------------------------------------------------- D. attribution
def attribute(net, yn, cn, prob, n):
    import cProfile
    import pstats
    smp = Sampler(net, yn, cn, DEVICE, BASE_STEPS, BASE_SOLVER)
    gmc_solve(prob, 200, smp, seed=3)                      # warm
    pr = cProfile.Profile()
    pr.enable()
    gmc_solve(prob, n, smp, seed=SEED + 7)
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, _) in st.stats.items():
        rows.append({"func": f"{pathlib.Path(fn).name}:{line}({name})",
                     "calls": nc, "tottime": tt, "cumtime": ct,
                     "pct": tt / total * 100 if total else 0.0})
    rows.sort(key=lambda r: -r["tottime"])
    return {"total_profiled_s": total, "top": rows[:25]}


# -------------------------------------------------------- E. batch decay
def batch_decay(net, yn, cn, prob, n):
    """Re-run the crossing loop recording the live batch at each iteration.

    gmc_solve keeps only the mean and median, but the shape of the decay is
    the point: a long tail of tiny batches is overhead-bound, not compute-
    bound, and the fix for that is different.
    """
    smp = Sampler(net, yn, cn, DEVICE, BASE_STEPS, BASE_SOLVER)
    sizes = []
    orig = smp.sample

    def spy(W, *a, **k):
        sizes.append(int(np.asarray(W).size))
        return orig(W, *a, **k)

    smp.sample = spy
    _, s = gmc_solve(prob, n, smp, seed=SEED + 7)
    sizes = np.array(sizes, float)
    cum = np.cumsum(sizes) / sizes.sum()
    small = sizes < 512
    return {"iterations": int(sizes.size), "sizes": sizes.astype(int).tolist(),
            "crossings": float(sizes.sum()),
            "iters_under_512": int(small.sum()),
            "crossings_under_512_pct": float(sizes[small].sum()
                                             / sizes.sum() * 100),
            "iters_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
            "mean": float(sizes.mean()), "median": float(np.median(sizes))}


# ------------------------------------------------- F. the fidelity knob
def fidelity_ladder(net, yn, cn, prob, n, mc_macro, cell_ref):
    """Cost AND accuracy for every solver/step count, no retraining needed."""
    rows = []
    for solver, steps in LADDER:
        smp = Sampler(net, yn, cn, DEVICE, steps, solver)
        g, s = gmc_solve(prob, n, smp, seed=SEED + 7)
        err = rel(g, mc_macro)

        # single-cell check at one fixed unseen condition, same as evaluate.py
        W, H, xi, ox, oy, ref = cell_ref
        smp.reset()
        cs = smp.sample(np.full(ref["p"].size, W), np.full(ref["p"].size, H),
                        np.full(ref["p"].size, xi), np.full(ref["p"].size, ox),
                        np.full(ref["p"].size, oy), seed=303)
        per = 2 * (W + H)
        w1_p = w1(ref["p"] / per, cs["p"] / per) / (ref["p"] / per).std()
        w1_s = w1(np.log10(ref["s"]), np.log10(cs["s"])) / np.log10(ref["s"]).std()

        rows.append({"solver": solver, "steps": steps,
                     "nfe": steps * NFE_PER_STEP[solver],
                     "wall": s["wall"], "wall_net": s["wall_net"],
                     "wall_host": s["wall_host"],
                     "crossings": s["crossings"],
                     "lattice_l2_pct": err * 100,
                     "w1_p_over_spread": w1_p, "w1_logs_over_spread": w1_s})
    return rows


# --------------------------------------------------------- G. projection
def projection(sweep, per_crossing_us, tag):
    """At what optical thickness does a GMC crossing cost less than the
    scatters it replaces?  Fit scatters-per-crossing against scale from the
    measured sweep and solve, rather than guessing a power law."""
    W = np.array([r["W"] for r in sweep], float)
    spc = np.array([r["scatters_per_crossing"] for r in sweep], float)
    tsc = np.array([r["t_per_scatter_ns"] for r in sweep], float)
    t_scatter = float(np.median(tsc)) * 1e-3          # us
    a, b = np.polyfit(np.log(W), np.log(spc), 1)      # spc ~ exp(b) W^a
    need = per_crossing_us / t_scatter                # scatters to break even
    W_star = float(np.exp((np.log(need) - b) / a))
    return {"intervention": tag, "us_per_crossing": per_crossing_us,
            "t_per_scatter_us": t_scatter,
            "scatters_needed_per_crossing": float(need),
            "spc_exponent": float(a),
            "breakeven_W_mfp": W_star}


def mc_sweep(scales=(1, 4, 10, 20, 40)):
    rows = []
    for sc in scales:
        p = mc.lattice(sc)
        _, st, t = timed_mc(p, 4000, SEED + 10)
        # macro crossings: the lattice is 7x7 one-cm cells
        rows.append({"scale": sc, "W": 1.0 * sc, "t_mc": t,
                     "scatters": st["scatters"],
                     "scatters_per_particle": st["scatters_per_particle"],
                     "t_per_scatter_ns": t / max(st["scatters"], 1) * 1e9})
    return rows


# ------------------------------------------------------------- reporting
def report(d, path):
    a = d["arithmetic"]
    ns = d["network_scaling"]
    hv = {r["variant"]: r for r in d["host_vs_network"]}
    real, free = hv["real model"], hv["free model"]
    bd = d["batch_decay"]
    lad = d["fidelity_ladder"]
    base = next(r for r in lad
                if r["solver"] == BASE_SOLVER and r["steps"] == BASE_STEPS)
    mcs = d["mc_sweep"]
    mc1 = mcs[0]
    proj = d["projections"]

    net_tbl = "\n".join(
        f"  {r['threads']:>7} {r['batch']:>8,} {r['t_call_us']:>12.1f} "
        f"{r['t_per_sample_us']:>14.3f}" for r in ns)

    impl0 = d["implementation"][0]["t_per_sample_us"]
    impl_tbl = "\n".join(
        f"  {r['variant']:<26} {r['t_per_sample_us']:>10.3f} "
        f"{impl0 / r['t_per_sample_us']:>11.2f}x  {r['note']}"
        for r in d["implementation"])
    size_tbl = "\n".join(
        f"  {r['width']:>5} {r['depth']:>5} {r['params']:>12,} "
        f"{r['t_per_sample_us']:>11.3f} {r['cheaper_than_shipped']:>11.1f}x"
        for r in d["size_scaling"])

    prof_tbl = "\n".join(
        f"  {r['pct']:>5.1f} {r['tottime']:>8.3f} {r['calls']:>9,}  {r['func']}"
        for r in d["attribution"]["top"][:18])

    lad_tbl = "\n".join(
        f"  {r['solver']:>6} {r['steps']:>5} {r['nfe']:>5} {r['wall']:>9.2f}"
        f" {r['wall_net']/r['wall']*100:>7.1f}% {r['lattice_l2_pct']:>11.2f}"
        f" {r['w1_p_over_spread']:>10.2f} {r['w1_logs_over_spread']:>10.2f}"
        f" {base['wall']/r['wall']:>9.2f}x" for r in lad)

    mc_tbl = "\n".join(
        f"  {r['scale']:>6g} {r['W']:>7.1f} {r['t_mc']:>10.4f} "
        f"{r['scatters_per_particle']:>13.1f} {r['t_per_scatter_ns']:>14.1f}"
        for r in mcs)

    proj_tbl = "\n".join(
        f"  {p['intervention']:<34} {p['us_per_crossing']:>9.2f} "
        f"{p['scatters_needed_per_crossing']:>12,.0f} {p['breakeven_W_mfp']:>12,.1f}"
        for p in proj)

    sizes = bd["sizes"]
    decay = "  " + " ".join(f"{s:,}" for s in sizes[:14])
    if len(sizes) > 14:
        decay += f"  ... ({len(sizes) - 14} more, down to {sizes[-1]:,})"

    txt = f"""\
======================================================================
SPEED PROFILE -- WHERE THE TIME GOES
======================================================================
host        {d['env']['cpu']}
            {d['env']['cores']} cores, no GPU detected: device = {DEVICE}
            torch {d['env']['torch']}, {d['env']['torch_threads']} threads;
            numba {d['env']['numba']}, {d['env']['numba_threads']} threads
baseline    mc.py is numba-compiled and runs on all {d['env']['numba_threads']} threads.
            The GMC host loop is single-threaded NumPy; only the network's
            matrix multiplies are threaded.  Both sides are CPU, so the
            comparison is honest, but it is not per-core.

----------------------------------------------------------------------
A. THE ARITHMETIC, BEFORE ANYTHING IS RUN
----------------------------------------------------------------------
network parameters                     {a['params']:>14,}
flops per network evaluation           {a['flop_per_nfe']:>14,}
network evaluations per sample         {a['nfe_per_sample']:>14,}
flops per CELL CROSSING (GMC)          {a['flop_per_crossing']:>14,}
flops per SCATTERING EVENT (MC, est.)  {a['flop_per_scatter_est']:>14,}
ratio                                  {a['arithmetic_ratio']:>14,.0f} x

Read that last line first.  One generated crossing costs about
{a['arithmetic_ratio']:,.0f} scattering events in raw arithmetic.  For the sampler to
win, a cell must contain more scatters than that -- which is a statement
about the geometry, not about the code.  Everything below is either a
constant factor on top of this, or a way of reducing it.

----------------------------------------------------------------------
B. NETWORK COST vs BATCH SIZE
----------------------------------------------------------------------
  threads    batch   per call (us)  per sample (us)
{net_tbl}

Per-sample cost falls until the matrix multiplies are large enough to
amortise the per-call overhead, then flattens at the arithmetic limit.
Where the solve actually calls it (section E) decides which regime it
is paying.

----------------------------------------------------------------------
B2. THE SAME MODEL, EXECUTED DIFFERENTLY (batch 4,096)
----------------------------------------------------------------------
  variant                      us/sample   vs shipped  note
{impl_tbl}

B3. WHAT A SMALLER VELOCITY FIELD WOULD COST (untrained: cost only)
  width depth       params   us/sample   cheaper by
{size_tbl}

Cost scales with the layers, so this table is reliable.  Whether a
smaller field is ACCURATE enough is a training question and is not
answered here.

----------------------------------------------------------------------
C. HOST LOOP vs NETWORK: the same solve with a FREE model
----------------------------------------------------------------------
The second row replaces the ODE right-hand side with a function that
returns zeros.  Shapes, batching, tallies, rotations and dtype
conversions are identical; the arithmetic is gone.  Its wall time is the
floor GMC could reach with a perfect, instantaneous model.

  variant        wall (s)   network   birth MC      host    us/crossing
  real model   {real['wall']:>9.3f} {real['wall_net']:>9.3f} {real['wall_birth']:>10.3f} {real['wall_host']:>9.3f} {real['us_per_crossing']:>14.2f}
  free model   {free['wall']:>9.3f} {free['wall_net']:>9.3f} {free['wall_birth']:>10.3f} {free['wall_host']:>9.3f} {free['us_per_crossing']:>14.2f}

  MC on the same problem, same particle count: {mc1['t_mc']:.4f} s
  GMC with a free model is {free['wall']/mc1['t_mc']:,.1f}x MC's time.

----------------------------------------------------------------------
D. WHERE THE HOST TIME GOES (cProfile, {N_PROFILE:,} particles)
----------------------------------------------------------------------
  %     self(s)     calls  function
{prof_tbl}

----------------------------------------------------------------------
E. BATCH DECAY ALONG THE CROSSING LOOP
----------------------------------------------------------------------
live particles per sampler call:
{decay}

  sampler calls                      {bd['iterations']:>10,}
  calls carrying 90% of all crossings{bd['iters_for_90pct']:>10,}
  calls with fewer than 512 particles{bd['iters_under_512']:>10,}
  crossings in those small calls     {bd['crossings_under_512_pct']:>9.1f} %

----------------------------------------------------------------------
F. THE FIDELITY KNOB (no retraining required)
----------------------------------------------------------------------
Accuracy columns: lattice L2 against the same MC field, and single-cell
Wasserstein-1 in units of the quantity's own spread (lower is better).

  solver steps   nfe  wall (s)     net%   lattice L2    W1 p/sd  W1 logs/sd    speedup
{lad_tbl}

----------------------------------------------------------------------
G. THE BASELINE'S OWN SCALING
----------------------------------------------------------------------
  scale    W_bg    MC (s)   scatters/part  ns per scatter
{mc_tbl}

MC cost per particle grows because the number of scattering events does.
That growth is the only thing working in the sampler's favour.

----------------------------------------------------------------------
H. BREAK-EVEN PROJECTION
----------------------------------------------------------------------
Scatters per crossing grows with optical thickness as W^{proj[0]['spc_exponent']:.2f} (fitted
over the measured sweep).  For each cost below, the thickness at which
one GMC crossing finally costs less than the scatters it replaces:

  intervention                       us/crossing  scatters req   W* (mfp)
{proj_tbl}

======================================================================
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)
    return txt


def main():
    print(f"device {DEVICE}, torch threads {torch.get_num_threads()}")
    net, yn, cn, cfg = load_parts()
    nfe = BASE_STEPS * NFE_PER_STEP[BASE_SOLVER]
    prob = mc.lattice()

    print("A. arithmetic")
    arith = flop_accounting(net, cfg, nfe)
    print(f"   {arith['flop_per_crossing']:,} flops per crossing "
          f"= {arith['arithmetic_ratio']:,.0f} scattering events")

    print("B. network scaling")
    ns = network_scaling(net)

    print("B2. implementation variants at batch 4,096")
    impl = implementation_variants(net)

    print("B3. cost of smaller velocity fields (untrained, cost only)")
    sizes = size_scaling(cfg)

    print("C. host vs network (real model, then a free one)")
    hv = host_vs_network(net, yn, cn, prob, N_LATTICE)
    for r in hv:
        print(f"   {r['variant']:<12} {r['wall']:8.3f} s  "
              f"host {r['wall_host']:.3f} s  net {r['wall_net']:.3f} s")

    print("D. attribution")
    attr = attribute(net, yn, cn, prob, N_PROFILE)

    print("E. batch decay")
    bd = batch_decay(net, yn, cn, prob, N_LATTICE)
    print(f"   {bd['iterations']} calls, median batch {bd['median']:.0f}")

    print("F. fidelity ladder")
    mc_macro = coarsen(mc.solve(prob, N_LATTICE, seed=SEED)[0], prob["per_cm"])
    W, H, xi, ox, oy = 4.0, 4.0, 0.5, 0.7, 0.2
    ref = mc.sample_cell(20000, W, H, xi=xi, direction=(ox, oy), seed=101)
    lad = fidelity_ladder(net, yn, cn, prob, N_LATTICE, mc_macro,
                          (W, H, xi, ox, oy, ref))
    for r in lad:
        print(f"   {r['solver']:>5} {r['steps']}  nfe {r['nfe']:>2}  "
              f"{r['wall']:7.2f} s  L2 {r['lattice_l2_pct']:.2f}%")

    print("G. baseline scaling")
    sweep = mc_sweep()

    # scatters per MACRO crossing: measure crossings from a GMC run at the
    # same scale, since MC does not count macro-cell crossings itself
    smp = Sampler(net, yn, cn, DEVICE, BASE_STEPS, BASE_SOLVER)
    for r in sweep:
        p = mc.lattice(r["scale"])
        _, gs = gmc_solve(p, 2000, smp, seed=SEED + 13)
        r["gmc_crossings_per_particle"] = gs["crossings_per_particle"]
        r["scatters_per_crossing"] = (r["scatters_per_particle"]
                                      / max(gs["crossings_per_particle"], 1e-9))
        print(f"   scale {r['scale']:>3}  {r['scatters_per_crossing']:8.1f} "
              f"scatters per macro crossing")

    print("H. projection")
    real = next(x for x in hv if x["variant"] == "real model")
    free = next(x for x in hv if x["variant"] == "free model")
    cr = max(real["crossings"], 1)
    net_only = real["wall_net"] / cr * 1e6
    host_only = free["wall"] / max(free["crossings"], 1) * 1e6
    projs = [
        projection(sweep, real["us_per_crossing"], "as measured today"),
        projection(sweep, net_only, "host loop made free (network only)"),
        projection(sweep, host_only, "network made free (host loop only)"),
        projection(sweep, net_only / 10 + host_only,
                   "network 10x cheaper (1 NFE, smaller net)"),
        projection(sweep, (net_only / 10 + host_only) / 20,
                   "  + host loop 20x cheaper (compiled)"),
    ]
    for p in projs:
        print(f"   {p['intervention']:<38} W* = {p['breakeven_W_mfp']:,.1f} mfp")

    env = {"cpu": platform.processor() or platform.machine(),
           "cores": len(__import__("os").sched_getaffinity(0)),
           "torch": torch.__version__,
           "torch_threads": torch.get_num_threads(),
           "numba": __import__("numba").__version__,
           "numba_threads": __import__("numba").config.NUMBA_NUM_THREADS,
           "python": sys.version.split()[0], "device": DEVICE}
    try:
        env["cpu"] = subprocess.run(
            ["bash", "-c", "grep -m1 'model name' /proc/cpuinfo"],
            capture_output=True, text=True).stdout.split(":", 1)[1].strip()
    except Exception:
        pass

    d = {"env": env, "arithmetic": arith, "network_scaling": ns,
         "implementation": impl, "size_scaling": sizes,
         "host_vs_network": hv, "attribution": attr, "batch_decay": bd,
         "fidelity_ladder": lad, "mc_sweep": sweep, "projections": projs,
         "config": cfg, "n_lattice": N_LATTICE}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speed_profile.json").write_text(json.dumps(d, indent=1))
    print("\n" + report(d, OUT / "speed_profile.txt"))
    print("wrote results/speed_profile.txt and results/speed_profile.json")


if __name__ == "__main__":
    main()
