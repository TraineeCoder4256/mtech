#!/usr/bin/env python3
"""Step 5: the fixes, measured.  Run: python speed_prototype.py

speed_profile.py says where the time goes.  This file tests whether the
obvious remedies actually recover it, on the same lattice, scored against
the same Monte Carlo field.  Nothing here edits mc.py, solve.py, sampler.py
or model.py -- every variant is built on top of their public interfaces, so
a result that looks good can be folded in afterwards deliberately.

Four levers, in increasing order of how much they ask of the model:

  1 BATCH SIZE      the sampler's per-particle cost falls with the size of
                    the batch it is called with, and the batch is the number
                    of live particles.  MC's per-particle cost is flat.  So
                    the comparison depends on the particle count, and the
                    honest version quotes it.

  2 THRESHOLD       the sampler replaces a random walk whatever that walk
                    would have cost.  In a 0.5 mfp cell it replaces about
                    one scattering event.  Sending thin cells back to analog
                    MC and keeping the sampler for thick ones costs nothing
                    in accuracy -- MC is the reference -- and removes the
                    crossings where the sampler could never have won.

  3 NFE             fewer ODE steps, or a cheaper solver: a pure speed knob
                    that needs no retraining.  Its accuracy cost is measured
                    in speed_profile.py section F; here it is combined with
                    the others.

  4 THREADS         the MC baseline is numba-parallel over every core; the
                    GMC host loop is single-threaded.  Reported both ways so
                    the algorithmic claim is separable from the hardware one.

Writes results/speed_prototype.txt and results/speed_prototype.json.
"""
import json
import pathlib
import time

import numpy as np
import torch
from numba import njit, prange

import mc
from data import load_norm
from model import VelocityField
from sampler import Sampler, NFE_PER_STEP
from solve import gmc_solve, coarsen

CKPT = pathlib.Path("models")
OUT = pathlib.Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1
STEPS, SOLVER = 5, "heun"
N_ACC = 20000                 # particles for the accuracy-scored solves
COUNTS = [2000, 20000, 200000]
THRESHOLDS = [0.0, 1.0, 2.0, 5.0, 1e9]     # optical width below which analog
SCALES = [1, 4, 10, 20]


# ------------------------------------------------- analog MC, but batched
@njit(cache=True, parallel=True)
def _walk_batch(W, H, xi, ox_in, oy_in, seed, n_blocks):
    """The same single-cell walk as mc.sample_cell, but each particle gets
    its OWN cell and entry state.

    mc.sample_cell pushes n particles through one cell; inside the solver
    every live particle sits in a different cell, so the per-condition
    version cannot be called there without a Python loop over particles.
    This is the shape the solver actually needs, and it is what makes an
    apples-to-apples per-crossing cost for analog MC possible at all.
    """
    n = W.shape[0]
    p = np.empty(n)
    dirs = np.empty((n, 3))
    s = np.empty(n)
    k = np.empty(n, np.int64)
    per = (n + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        np.random.seed(seed + b)
        for i in range(b * per, min(n, b * per + per)):
            w, h = W[i], H[i]
            x, y = 0.0, xi[i] * h
            ox, oy = ox_in[i], oy_in[i]
            rz = 1.0 - ox * ox - oy * oy
            oz = np.sqrt(rz) if rz > 0.0 else 0.0
            if np.random.random() < 0.5:
                oz = -oz
            xe, ye, oxe, oye, oze, path, ns = mc._walk(x, y, ox, oy, oz, w, h)
            p[i] = mc._perimeter(xe, ye, w, h)
            dirs[i, 0], dirs[i, 1], dirs[i, 2] = oxe, oye, oze
            s[i] = path
            k[i] = ns
    return p, dirs, s, k


def walk_batch(W, H, xi, ox, oy, seed=1, n_blocks=8):
    return _walk_batch(np.ascontiguousarray(W, np.float64),
                       np.ascontiguousarray(H, np.float64),
                       np.ascontiguousarray(xi, np.float64),
                       np.ascontiguousarray(ox, np.float64),
                       np.ascontiguousarray(oy, np.float64),
                       int(seed), int(n_blocks))


class HybridSampler:
    """Sampler interface, but thin cells are walked instead of generated.

    Drop-in for Sampler in gmc_solve: same .sample(), .reset() and .stats.
    A cell whose optical width is below `thresh` costs the network more than
    the walk it replaces, so it goes to analog MC -- which is not an
    approximation, it is the reference.  thresh = 0 is pure GMC;
    thresh = inf is analog MC driven through the same host loop, which is
    the cost the sampler has to beat crossing for crossing.
    """

    def __init__(self, inner, thresh):
        self.inner, self.thresh = inner, thresh
        self.nfe_per_sample = inner.nfe_per_sample
        self.reset()

    def reset(self):
        self.inner.reset()
        self.stats = {"calls": 0, "particles": 0, "to_network": 0, "nfe": 0,
                      "clamped": 0, "to_walk": 0, "walk_scatters": 0}

    def sample(self, W, H, xi, ox_in, oy_in, seed=0):
        W, H, xi = np.broadcast_arrays(np.atleast_1d(np.asarray(W, float)),
                                       np.asarray(H, float),
                                       np.asarray(xi, float))
        ox_in = np.broadcast_to(np.asarray(ox_in, float), W.shape)
        oy_in = np.broadcast_to(np.asarray(oy_in, float), W.shape)
        n = W.size
        thin = W < self.thresh
        p, s = np.empty(n), np.empty(n)
        dirs = np.empty((n, 3))
        unc = np.zeros(n, bool)

        if thin.any():
            m = thin
            pw, dw, sw, kw = walk_batch(W[m], H[m], xi[m], ox_in[m], oy_in[m],
                                        seed=seed)
            p[m], s[m], dirs[m] = pw, sw, dw
            unc[m] = kw == 0
            self.stats["to_walk"] += int(m.sum())
            self.stats["walk_scatters"] += int(kw.sum())

        if (~thin).any():
            m = ~thin
            g = self.inner.sample(W[m], H[m], xi[m], ox_in[m], oy_in[m],
                                  seed=seed + 1)
            p[m], s[m], dirs[m] = g["p"], g["s"], g["dir"]
            unc[m] = g["uncollided"]

        self.stats["calls"] += 1
        self.stats["particles"] += n
        for k in ("to_network", "nfe", "clamped"):
            self.stats[k] = self.inner.stats[k]
        from sampler import decode_p
        _, _, face = decode_p(p, W, H)
        return {"p": p, "dir": dirs, "s": s, "face": face, "uncollided": unc}


# --------------------------------------------------------------- helpers
def load_sampler(steps=STEPS, solver=SOLVER):
    st = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True)
    c = st["config"]
    net = VelocityField(c["x_dim"], c["c_dim"], c["width"], c["depth"])
    net.load_state_dict(st["ema"])
    yn, cn, _ = load_norm(CKPT / "norm.json")
    return Sampler(net, yn, cn, DEVICE, steps, solver)


def timed_mc(problem, n, seed, min_time=0.3):
    mc.solve(problem, 200, seed=0)
    reps, t0, el = 0, time.perf_counter(), 0.0
    while el < min_time:
        phi, stats = mc.solve(problem, n, seed=seed + reps)
        reps += 1
        el = time.perf_counter() - t0
    return phi, stats, el / reps


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def mc_reference(prob, n, seeds=(SEED, SEED + 1000, SEED + 2000)):
    """Reference field and its own noise floor, so a variant's error can be
    read against what two MC runs differ by at the same particle count."""
    fields = [coarsen(mc.solve(prob, n, seed=s)[0], prob["per_cm"])
              for s in seeds]
    floor = float(np.mean([rel(fields[1], fields[0]), rel(fields[2], fields[0]),
                           rel(fields[2], fields[1])]))
    return fields[0], floor


# ------------------------------------------------- 1. cost vs batch size
def lever_batch(prob):
    """GMC's per-particle cost falls with the particle count; MC's does not.

    Every live particle is one row of one batched network call, so a solve
    with ten times the particles calls the same network with ten times the
    batch and amortises the per-call overhead over it.  This is free
    speedup that costs nothing anywhere else -- but it only helps up to the
    point where the matrix multiplies are already large.
    """
    smp = load_sampler()
    rows = []
    for n in COUNTS:
        _, mcs, t_mc = timed_mc(prob, n, SEED)
        _, gs = gmc_solve(prob, n, smp, seed=SEED + 7)
        rows.append({"particles": n, "t_mc": t_mc, "t_gmc": gs["wall"],
                     "mc_us_per_particle": t_mc / n * 1e6,
                     "gmc_us_per_particle": gs["wall"] / n * 1e6,
                     "gmc_us_per_crossing": gs["wall"] / max(gs["crossings"], 1) * 1e6,
                     "mean_batch": gs["mean_batch"],
                     "speedup": t_mc / gs["wall"]})
        print(f"   n {n:>7,}  MC {t_mc:7.3f}s  GMC {gs['wall']:8.2f}s  "
              f"{rows[-1]['gmc_us_per_crossing']:7.1f} us/crossing  "
              f"mean batch {gs['mean_batch']:.0f}", flush=True)
    return rows


# -------------------------------------------- 2. the thin-cell threshold
def lever_threshold(scales=SCALES, n=N_ACC):
    """Send cells thinner than `thresh` back to analog MC.

    thresh = 0 is the current code.  thresh = inf is the same host loop
    driving analog MC in every cell, which is the like-for-like cost the
    sampler must beat.  Accuracy is scored against MC either way, so a
    hybrid cannot be worse than the sampler alone -- the cells it takes
    back are the ones it does exactly.
    """
    inner = load_sampler()
    rows = []
    for sc in scales:
        prob = mc.lattice(sc)
        ref, floor = mc_reference(prob, n)
        _, mcs, t_mc = timed_mc(prob, n, SEED)
        for th in THRESHOLDS:
            smp = HybridSampler(inner, th) if th > 0 else inner
            smp.reset()
            g, gs = gmc_solve(prob, n, smp, seed=SEED + 7)
            st = smp.stats if th > 0 else dict(gs)
            cr = max(gs["crossings"], 1)
            rows.append({
                "scale": sc, "W": 1.0 * sc, "thresh": th, "particles": n,
                "t_mc": t_mc, "t_gmc": gs["wall"], "speedup": t_mc / gs["wall"],
                "l2_pct": rel(g, ref) * 100, "floor_pct": floor * 100,
                "crossings": gs["crossings"],
                "to_network": int(st.get("to_network", gs["to_network"])),
                "to_walk": int(st.get("to_walk", 0)),
                "network_frac": st.get("to_network", gs["to_network"]) / cr,
                "us_per_crossing": gs["wall"] / cr * 1e6})
            print(f"   scale {sc:>3}  thresh {th:>6.1f}  GMC {gs['wall']:8.2f}s "
                  f"speedup {t_mc/gs['wall']:8.4g}x  L2 {rel(g, ref)*100:6.2f}% "
                  f"(floor {floor*100:.2f}%)", flush=True)
    return rows


# ------------------------------------------------ 3. NFE x threshold
def lever_nfe(prob, n=N_ACC, ladder=(("euler", 1), ("euler", 2),
                                     ("heun", 5))):
    ref, floor = mc_reference(prob, n)
    _, mcs, t_mc = timed_mc(prob, n, SEED)
    rows = []
    for solver, steps in ladder:
        inner = load_sampler(steps, solver)
        for th in (0.0, 2.0):
            smp = HybridSampler(inner, th) if th > 0 else inner
            smp.reset()
            g, gs = gmc_solve(prob, n, smp, seed=SEED + 7)
            rows.append({"solver": solver, "steps": steps,
                         "nfe": steps * NFE_PER_STEP[solver], "thresh": th,
                         "t_gmc": gs["wall"], "t_mc": t_mc,
                         "speedup": t_mc / gs["wall"],
                         "l2_pct": rel(g, ref) * 100, "floor_pct": floor * 100})
            print(f"   {solver:>5} {steps} (nfe {rows[-1]['nfe']:>2}) "
                  f"thresh {th:>4.1f}  {gs['wall']:8.2f}s  "
                  f"speedup {t_mc/gs['wall']:8.4g}x  "
                  f"L2 {rows[-1]['l2_pct']:6.2f}%", flush=True)
    return rows


# --------------------------------------------------------- 4. thread fairness
def lever_threads(prob, n=N_ACC):
    rows = []
    for nt in sorted({1, torch.get_num_threads()}):
        torch.set_num_threads(nt)
        smp = load_sampler()
        _, gs = gmc_solve(prob, n, smp, seed=SEED + 7)
        rows.append({"torch_threads": nt, "t_gmc": gs["wall"],
                     "wall_net": gs["wall_net"], "wall_host": gs["wall_host"]})
        print(f"   torch threads {nt}: {gs['wall']:.2f}s "
              f"(net {gs['wall_net']:.2f}s)", flush=True)
    torch.set_num_threads(max(r["torch_threads"] for r in rows))
    return rows


# ------------------------------------------------------------- reporting
def report(d, path):
    b = d["batch"]
    th = d["threshold"]
    nf = d["nfe"]
    tr = d["threads"]

    b_tbl = "\n".join(
        f"  {r['particles']:>9,} {r['t_mc']:>9.3f} {r['t_gmc']:>9.2f} "
        f"{r['mc_us_per_particle']:>12.2f} {r['gmc_us_per_particle']:>13.1f} "
        f"{r['gmc_us_per_crossing']:>13.1f} {r['mean_batch']:>11,.0f} "
        f"{r['speedup']:>9.4g}x" for r in b)

    def label(r):
        if r["thresh"] == 0:
            return "pure GMC"
        if r["thresh"] > 1e8:
            return "analog MC"
        return f"walk < {r['thresh']:g} mfp"

    def th_block(sc):
        rs = [r for r in th if r["scale"] == sc]
        head = (f"  cells {rs[0]['W']:.0f} mfp wide, MC reference "
                f"{rs[0]['t_mc']:.3f} s, noise floor {rs[0]['floor_pct']:.2f}%\n"
                f"    variant            GMC (s)   speedup    L2 vs MC"
                f"   via network")
        body = "\n".join(
            f"    {label(r):<18} {r['t_gmc']:>8.2f} {r['speedup']:>9.4g}x"
            f" {r['l2_pct']:>9.2f} % {r['network_frac']*100:>11.1f} %"
            for r in rs)
        return head + "\n" + body

    th_tbl = "\n\n".join(th_block(sc) for sc in sorted({r["scale"] for r in th}))

    nf_tbl = "\n".join(
        f"  {r['solver']:>6} {r['steps']:>3} {r['nfe']:>5} {r['thresh']:>8.1f}"
        f" {r['t_gmc']:>9.2f} {r['speedup']:>9.4g}x {r['l2_pct']:>9.2f} %"
        for r in nf)

    tr_tbl = "\n".join(
        f"  {r['torch_threads']:>7} {r['t_gmc']:>9.2f} {r['wall_net']:>9.2f}"
        f" {r['wall_host']:>9.3f}" for r in tr)

    best = max(th, key=lambda r: r["speedup"])
    bestn = max(nf, key=lambda r: r["speedup"])

    txt = f"""\
======================================================================
SPEED PROTOTYPE -- DO THE FIXES RECOVER THE TIME?
======================================================================
Every variant is scored on the same lattice against the same Monte Carlo
field, so a faster number that costs accuracy shows up as a worse L2.
Nothing in mc.py, model.py, sampler.py or solve.py was modified.

----------------------------------------------------------------------
1. PARTICLE COUNT: GMC amortises, MC does not
----------------------------------------------------------------------
  particles   MC (s)   GMC (s)   MC us/part    GMC us/part  GMC us/cross\
   mean batch   speedup
{b_tbl}

MC costs the same per particle however many there are.  GMC's cost per
crossing falls as the live batch grows, because the batch IS the matrix
multiply.  Any speedup quoted without the particle count is not a number.

----------------------------------------------------------------------
2. THE THIN-CELL THRESHOLD: stop generating what is cheap to simulate
----------------------------------------------------------------------
"analog MC" is the same host loop with the walk done exactly, which is the
per-crossing cost the sampler has to beat.  Cells below the threshold are
walked; the rest are generated.

{th_tbl}

  best speedup seen: {best['speedup']:.4g}x at scale {best['scale']:g}, \
threshold {best['thresh']:g}, L2 {best['l2_pct']:.2f}%

----------------------------------------------------------------------
3. FEWER NETWORK EVALUATIONS, WITH AND WITHOUT THE THRESHOLD
----------------------------------------------------------------------
  solver steps   nfe   thresh   GMC (s)   speedup      L2 vs MC
{nf_tbl}

  best: {bestn['solver']} {bestn['steps']} at threshold {bestn['thresh']:g}, \
{bestn['speedup']:.4g}x, L2 {bestn['l2_pct']:.2f}% \
(floor {bestn['floor_pct']:.2f}%)

----------------------------------------------------------------------
4. THREADS
----------------------------------------------------------------------
  threads   GMC (s)   network      host
{tr_tbl}

The MC baseline uses every core through numba.  The GMC host loop is
single-threaded; only the network's matrix multiplies are threaded.  A
per-core comparison would move the crossover, and neither number is wrong
as long as it says which one it is.
======================================================================
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)
    return txt


def main():
    torch.set_num_threads(torch.get_num_threads())
    prob = mc.lattice()
    print(f"device {DEVICE}, torch threads {torch.get_num_threads()}")
    print("warming the batched walk kernel")
    walk_batch(*[np.full(64, v) for v in (1.0, 1.0, 0.5, 0.8, 0.1)])

    print("1. particle count")
    b = lever_batch(prob)
    print("2. thin-cell threshold")
    th = lever_threshold()
    print("3. network evaluations")
    nf = lever_nfe(prob)
    print("4. threads")
    tr = lever_threads(prob)

    d = {"batch": b, "threshold": th, "nfe": nf, "threads": tr,
         "thresholds": THRESHOLDS, "scales": SCALES, "n_acc": N_ACC}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speed_prototype.json").write_text(json.dumps(d, indent=1))
    print("\n" + report(d, OUT / "speed_prototype.txt"))
    print("wrote results/speed_prototype.txt and results/speed_prototype.json")


if __name__ == "__main__":
    main()
