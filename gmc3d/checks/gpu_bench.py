"""What the network costs on a GPU, and what that would do to whole runs.

Used by: run by hand on any machine with PyTorch, NumPy and Numba (OpenMC
is not needed).  From gmc3d/:

    python -m checks.gpu_bench --device cuda
    python -m checks.gpu_bench --device cuda --compile    # fused kernels
    python -m checks.gpu_bench --device cpu               # today's numbers

Two parts:

 1. COST PER DRAW of the network on `device`, at batch sizes (requests per
    backend call) from 1,000 to 1,000,000.  The time is the backend's whole
    call: radii to the device, the flow, (mu, s) back to the CPU.  Uses the
    trained network if data/models/flow_seed0/model.pt exists, otherwise
    an untrained one of the same size, which costs the same (only the
    weights differ).  `--compile` runs the flow through torch.compile,
    which fuses its many small operations into fewer kernels.

 2. PROJECTION onto the race (runs/2026-10-05/race.json, 4M histories on
    4 CPU cores).  For each problem the race measured a run with
    exact-walk balls at R* = 2: its transport time without the walk, its
    number of balls, the requests per backend call and its error bar.
    Putting the network in place of the walk at the cost measured in part
    1 gives the run time; the error bar stays, since the network is as
    accurate as the walk ball by ball (ballcheck.log).  Printed against
    OpenMC (analog, as in the race), at the race's own batch size and
    with 10 times more particles in flight per batch.

This is a projection, not a race: the transport loop is taken to run as it
did on the race's 4 CPU cores, and OpenMC is not rerun.  On a machine with
OpenMC installed, `python run.py race` with the network on the GPU is the
real measurement.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ball import data as D
from ball import network as N

RACE = Path(__file__).resolve().parent.parent / "runs" / "2026-10-05" / \
    "race.json"
BATCHES = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)


def load(device):
    """The trained network on `device`, or an untrained one of its size."""
    path = N.MODEL_DIR / "flow_seed0"
    if (path / "model.pt").exists():
        return N.Network.load(path, device=device), True
    info = {"r_min": N.R_MIN, "r_max": N.R_MAX}
    return N.Network(N.SplineFlow(), D.Norm(0.0, 1.0),
                     D.Norm(np.zeros(2), np.ones(2)), info,
                     device=device), False


def cost_per_draw(be, batch, repeats=5):
    """Seconds per draw at one batch size (best of `repeats`)."""
    R = np.full(batch, 5.0)
    be.sample(R)                                  # warm up / compile
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        be.sample(R)                              # ends with a copy to the
        best = min(best, time.perf_counter() - t)  # CPU, so it is synced
    return best / batch


def project(costs, log=print):
    """Race runs with the network in place of the exact walk."""
    b = np.log(np.array(sorted(costs)))
    c = np.log(np.array([costs[k] for k in sorted(costs)]))

    def cost_at(batch):
        return float(np.exp(np.interp(np.log(batch), b, c)))

    race = json.loads(RACE.read_text())
    log("\nProjected time to reach OpenMC's leakage error bar, relative to "
        "OpenMC (higher is better):")
    log(f"  {'problem':10s} {'race, CPU network':>18s} "
        f"{'this device':>12s} {'10x in flight':>14s} {'free network':>13s}")
    for name, rows in race.items():
        om = rows[0]
        ex = next(r for r in rows if r["label"].startswith("ours + exact"))
        now = next(r for r in rows
                   if r["label"] == "ours + network (R* = 3)")
        base = om["fom"]["leakage"]
        balls, calls = ex["counters"]["balls"], ex["counters"]["backend_calls"]
        cpu = now["fom"]["leakage"] / base
        if not balls:
            log(f"  {name:10s} {cpu:17.2f}x   no ball fits: no change")
            continue
        transport = ex["seconds"] - ex["backend_s"]
        fom = ex["fom"]["leakage"] / base * ex["seconds"]   # x seconds
        per_call = balls / calls
        ratios = [fom / (transport + balls * cost_at(k * per_call))
                  for k in (1, 10)] + [fom / transport]
        log(f"  {name:10s} {cpu:17.2f}x " + " ".join(
            f"{r:{w}.2f}x" for r, w in zip(ratios, (11, 13, 12))) +
            f"   ({per_call:,.0f} requests per call)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()
    be, trained = load(a.device)
    if a.compile:
        be.net.sample = torch.compile(be.net.sample, dynamic=True)
    name = torch.cuda.get_device_name(be.device) \
        if be.device.type == "cuda" else f"CPU, {torch.get_num_threads()} " \
        "threads"
    print(f"device: {name}; torch {torch.__version__}; "
          f"{'trained' if trained else 'untrained (same cost)'} network, "
          f"{sum(q.numel() for q in be.net.parameters()):,} weights"
          f"{'; compiled' if a.compile else ''}")
    costs = {}
    for batch in BATCHES:
        costs[batch] = cost_per_draw(be, batch)
        print(f"  batch {batch:>9,}: {1e9 * costs[batch]:8.0f} ns per draw")
    project(costs)


if __name__ == "__main__":
    main()
