#!/usr/bin/env python3
"""How well does this code actually use the hardware it is given?

The GMC cost model is one number: NFE/s, network function evaluations per
second.  Everything else -- ODE steps, solver order, particles per second,
the crossover against Monte Carlo -- is that number divided by how many
evaluations a sample needs.  So this script measures it directly, and
measures the two things that stop it reaching peak.

  1. THROUGHPUT vs BATCH.  A 1.7 M-parameter MLP is small.  On a GPU the
     matmuls for a few hundred samples finish faster than the kernels can
     be launched, so throughput climbs with batch size until the device
     saturates.  The batch where the curve flattens is the smallest batch
     worth sending, and the transport driver must be run with at least that
     many live particles or the GPU idles.

  2. LAUNCH OVERHEAD.  Each network evaluation issues ~50 small kernels
     (LayerNorm, SiLU, FiLM chunk, residual adds).  Comparing measured
     throughput against the same FLOPs done as one big matmul chain bounds
     how much is being lost to launch latency rather than arithmetic, and
     the torch.compile row shows how much of that a fused graph recovers.

  3. TRANSPORT OCCUPANCY.  In the end-to-end solve the live particle set
     shrinks at every cell crossing.  The sampler call is batched over
     whatever is still alive, so late crossings run at tiny batch -- back
     in the launch-bound regime of (1).  The occupancy table shows how fast
     the batch decays, which sets how large --n has to be.

Nothing here needs a trained checkpoint except --ckpt for the transport
occupancy section, which is skipped if the checkpoint is missing.

Usage:
  python scripts/bench_gpu.py                      # auto-detect device
  python scripts/bench_gpu.py --device cuda --compile
  python scripts/bench_gpu.py --device cpu --batches 1024 8192
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmc import VelocityField                       # noqa: E402
from gmc.device import pick_device, describe        # noqa: E402
from gmc.sampler import integrate, NFE_PER_STEP     # noqa: E402


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn, dev, warmup=3, reps=10, min_seconds=0.25):
    """Median-of-reps wall time, with warmup and enough reps to be stable."""
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
        if sum(ts) > min_seconds and len(ts) >= 3:
            break
    return float(np.median(ts))


def fmt(x):
    for u, d in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x/d:.2f} {u}"
    return f"{x:.1f} "


# ------------------------------------------------------------------ (1)+(2)
def throughput(model, dev, batches, nfe_flops, compiled=None):
    print("\n" + "=" * 78)
    print("1. RAW THROUGHPUT vs BATCH SIZE  (one network evaluation)")
    print("=" * 78)
    hdr = (f"{'batch':>9} {'ms/NFE':>9} {'NFE/s':>12} {'samples/s':>13} "
           f"{'eff. FLOP/s':>13} {'vs best':>8}")
    if compiled is not None:
        hdr += f" {'compiled NFE/s':>15} {'gain':>6}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for b in batches:
        x = torch.randn(b, model.x_dim, device=dev)
        t = torch.rand(b, 1, device=dev)
        c = torch.randn(b, model.c_dim, device=dev)
        with torch.no_grad():
            dt = timeit(lambda: model(x, t, c), dev)
            nfe_s = b / dt
            line = (f"{b:>9,} {dt*1e3:>9.3f} {fmt(nfe_s):>12} "
                    f"{fmt(nfe_s):>13} {fmt(nfe_s*nfe_flops):>13}")
            cg = None
            if compiled is not None:
                dtc = timeit(lambda: compiled(x, t, c), dev)
                cg = b / dtc
            rows.append((b, nfe_s, cg, line))
        del x, t, c
    best = max(r[1] for r in rows)
    for b, nfe_s, cg, line in rows:
        extra = ""
        if cg is not None:
            extra = f" {fmt(cg):>15} {cg/nfe_s:>5.2f}x"
        print(f"{line} {nfe_s/best*100:>7.1f}%{extra}")
    print(f"\n  peak {fmt(best)}NFE/s  ->  {fmt(best*nfe_flops)}FLOP/s effective")
    print("  'vs best' is the fraction of this device's own peak reached at")
    print("  that batch: anything below ~80% means the batch is too small and")
    print("  the device is waiting on kernel launches, not computing.")
    return rows, best


def launch_bound_probe(model, dev, batch, nfe_flops):
    """Same FLOPs as one NFE, but as a few big matmuls: an achievable ceiling."""
    print("\n" + "=" * 78)
    print(f"2. HOW MUCH IS LAUNCH OVERHEAD?  (batch {batch:,})")
    print("=" * 78)
    w = 256
    n_mat = max(1, int(round(nfe_flops / (2.0 * w * w))))
    A = torch.randn(batch, w, device=dev)
    Ws = [torch.randn(w, w, device=dev) for _ in range(min(n_mat, 32))]
    reps = max(1, n_mat // len(Ws))

    def dense():
        h = A
        for _ in range(reps):
            for W in Ws:
                h = h @ W
        return h

    with torch.no_grad():
        d_dense = timeit(dense, dev)
        x = torch.randn(batch, model.x_dim, device=dev)
        t = torch.rand(batch, 1, device=dev)
        c = torch.randn(batch, model.c_dim, device=dev)
        d_model = timeit(lambda: model(x, t, c), dev)
    fl = 2.0 * batch * w * w * len(Ws) * reps
    print(f"  pure matmul chain of equal size : {d_dense*1e3:8.3f} ms  "
          f"({fmt(fl/d_dense)}FLOP/s)")
    print(f"  the actual velocity field       : {d_model*1e3:8.3f} ms  "
          f"({fmt(nfe_flops*batch/d_model)}FLOP/s)")
    r = d_model / d_dense
    print(f"  ratio {r:.2f}x")
    if r > 1.6:
        print("  -> more than half the time is NOT in the matmuls: the small")
        print("     elementwise/norm kernels and their launch latency dominate.")
        print("     torch.compile (--compile) or CUDA graphs is the fix.")
    else:
        print("  -> arithmetic-bound; the layout is already close to what the")
        print("     hardware can do for a network this size.")


# ------------------------------------------------------------------ solvers
def solver_cost(model, dev, batch, steps_list):
    print("\n" + "=" * 78)
    print(f"3. SOLVER COST AT MATCHED NFE  (batch {batch:,})")
    print("=" * 78)
    print("  Cost is NFE, not steps: Euler 1/step, Heun 2, RK4 4.  Rows with")
    print("  the same NFE cost the same; whether they are equally ACCURATE is")
    print("  a separate question, answered by scripts/eval_boundary_model.py.")
    print(f"\n{'solver':>7} {'steps':>6} {'NFE':>5} {'ms/batch':>10} "
          f"{'samples/s':>12}")
    z = torch.randn(batch, model.x_dim, device=dev)
    c = torch.randn(batch, model.c_dim, device=dev)
    for solver in ("euler", "heun", "rk4"):
        for st in steps_list:
            nfe = st * NFE_PER_STEP[solver]
            d = timeit(lambda: integrate(model, z, c, st, solver), dev,
                       warmup=1, reps=4)
            print(f"{solver:>7} {st:>6} {nfe:>5} {d*1e3:>10.2f} "
                  f"{fmt(batch/d):>12}")


# ------------------------------------------------------------------ (3)
def transport_occupancy(ckpt, dev, n, ode_steps):
    print("\n" + "=" * 78)
    print("4. TRANSPORT OCCUPANCY: does the batch stay large enough?")
    print("=" * 78)
    sys.path.insert(0, str(ROOT / "scripts"))
    from gmc_end_to_end import load_sampler, birth_analog     # noqa: E402
    from mc2d.lattice import build_lattice                    # noqa: E402
    from gmc.transport import macro_problem, run_gmc_transport  # noqa: E402

    sampler = load_sampler(ckpt, ode_steps=ode_steps, device=dev)
    sizes, flow = [], []
    orig = sampler.sample

    def spy(W, *a, **k):
        out = orig(W, *a, **k)
        sizes.append(np.size(W))
        # only the collided particles go through the ODE; the uncollided
        # branch is analytic and costs no network evaluation, so counting
        # the whole batch would overstate NFE (and hence NFE/s)
        flow.append(int((~out["uncollided"]).sum()))
        return out

    sampler.sample = spy
    prob = build_lattice(5, pitch=1.0, cells_per_pitch=16,
                         background={"sig_s": 4.0, "sig_a": 0.0},
                         absorber={"sig_s": 2.0, "sig_a": 38.0})
    ss, sa, _ = macro_problem(prob, 1.0, 16)
    t0 = time.perf_counter()
    _, st = run_gmc_transport(ss, sa, 1.0, n, sampler, (3, 3), seed=7,
                              birth_sampler=birth_analog, return_stats=True)
    wall = time.perf_counter() - t0
    sizes, flow = np.array(sizes), np.array(flow)
    nfe = flow.sum() * ode_steps * NFE_PER_STEP[sampler.solver]
    print(f"  {n:,} particles, {len(sizes)} batched sampler calls, "
          f"{st['crossings_per_particle']:.1f} crossings/particle")
    print(f"  {sizes.sum():,} particle-crossings, of which "
          f"{flow.sum():,} ({flow.sum()/sizes.sum()*100:.1f}%) reached the "
          f"network;\n  the rest took the analytic uncollided branch and "
          f"cost no evaluation")
    print(f"  wall {wall:.2f} s -> {fmt(nfe/wall)}NFE/s in situ "
          f"(compare with the peak in section 1)")
    print(f"\n{'call':>5} {'live batch':>11} {'to network':>11} "
          f"{'% of start':>11}")
    for i, s in enumerate(sizes):
        if i < 6 or i % max(1, len(sizes) // 8) == 0 or i == len(sizes) - 1:
            print(f"{i:>5} {s:>11,} {flow[i]:>11,} {s/sizes[0]*100:>10.1f}%")
    small = sizes[sizes < 2048].sum() / sizes.sum()
    calls_small = float((sizes < 2048).mean())
    print(f"\n  {small*100:.1f}% of all particle-crossings are drawn in calls "
          f"with fewer\n  than 2048 live particles -- but those account for "
          f"{calls_small*100:.1f}% of the CALLS.")
    print("  Wall time at small batch is dominated by per-call overhead, not")
    print("  by batch size, so the call share is the one that costs you.")
    if calls_small > 0.3:
        print("  -> a long thin tail of surviving particles is being tracked a")
        print("     handful at a time.  Raising --n raises the whole occupancy")
        print("     curve; lowering --max-crossings or raising the weight")
        print("     cutoff truncates the tail instead.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[256, 1024, 4096, 16384, 65536, 262144])
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--compile", action="store_true",
                    help="also time a torch.compile'd copy of the network")
    ap.add_argument("--ckpt", default="models/boundary_v2")
    ap.add_argument("--transport-n", type=int, default=20000)
    ap.add_argument("--ode-steps", type=int, default=5)
    ap.add_argument("--skip-transport", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(describe(dev))
    model = VelocityField(width=args.width, depth=args.depth).to(dev).eval()
    npar = model.n_params()
    # forward FLOPs ~ 2 * params per sample (multiply-accumulate per weight)
    nfe_flops = 2.0 * npar
    print(f"velocity field: width {args.width} depth {args.depth}, "
          f"{npar:,} params -> ~{fmt(nfe_flops)}FLOP per evaluation per sample")

    compiled = None
    if args.compile:
        try:
            compiled = torch.compile(model, mode="max-autotune")
            print("torch.compile: enabled (first call includes compilation)")
        except Exception as e:                       # pragma: no cover
            print(f"torch.compile unavailable: {e}")

    rows, best = throughput(model, dev, args.batches, nfe_flops, compiled)
    launch_bound_probe(model, dev, max(args.batches), nfe_flops)
    solver_cost(model, dev, min(65536, max(args.batches)), [4, 8, 12])

    ck = ROOT / args.ckpt
    if args.skip_transport:
        pass
    elif (ck / "model.pt").exists():
        transport_occupancy(ck, dev, args.transport_n, args.ode_steps)
    else:
        print(f"\n(skipping transport occupancy: no checkpoint at {ck})")

    print("\n" + "=" * 78)
    print("READ THIS OFF THE TABLES")
    print("=" * 78)
    print("  * smallest batch reaching ~80% of peak  -> the minimum useful")
    print("    batch; run the transport driver with several times that many")
    print("    particles.")
    print("  * ratio in section 2 near 1 -> arithmetic-bound, nothing left to")
    print("    win from fusion; well above 1 -> try --compile.")
    print("  * in-situ NFE/s in section 4 well below peak in section 1 -> the")
    print("    end-to-end solve is limited by batch decay and per-crossing")
    print("    host/device synchronisation, not by the network.")


if __name__ == "__main__":
    main()
