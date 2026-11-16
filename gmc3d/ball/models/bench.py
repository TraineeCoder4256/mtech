"""Cost per draw of every trained family, with every speed option (m2.4).

    python -m ball.models.bench                       # CPU here
    python -m ball.models.bench --device cuda --out runs/<date>/gpu-bench

For each family, interface v1 (R -> mu, s) and v2 (R, g -> cos_n, mu,
phi, s, the family's own draw without the exact first scatter), and each
option the device allows, it times one backend call at batch sizes 1k,
10k, 100k and 1M and reports microseconds per draw: the whole call the
transport makes (encode on the host, copy over, draw, copy back, decode).

Options (fast.Engine): on the CPU "plain" and "compile"; on a GPU also
"graphs" (CUDA graphs on padded sizes), "fp16", "compile+fp16" and
"graphs+fp16".  Every variant is warmed up (compiled, graphs recorded)
before it is timed, and each time is the best of --repeats calls.

Accuracy is printed next to speed: the largest difference from the plain
fp32 draw on the same random numbers ("same noise" for the variants that
use them; graph variants pad the batch, so they draw other noise and show
"-").  Whether a variant is good enough is gate2's call, not this file's:
run `run.py gate2 <model> --device cuda` with the same options.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import names, load

SIZES = (1_000, 10_000, 100_000, 1_000_000)
CPU_OPTIONS = {"plain": {}, "compile": {"compile": True}}
GPU_OPTIONS = {"plain": {}, "compile": {"compile": True},
               "graphs": {"graphs": True}, "fp16": {"precision": "fp16"},
               "compile+fp16": {"compile": True, "precision": "fp16"},
               "graphs+fp16": {"graphs": True, "precision": "fp16"}}


def _inputs(n, interface, seed=1):
    rng = np.random.default_rng(seed)
    R = rng.uniform(1.0, 30.0, n)
    if interface == 1:
        return (R,)
    g = rng.uniform(0.0, 0.6, (n, 12)) * np.linspace(1, 0.2, 12)
    return R, g


def _call(be, interface, args):
    return be.sample(*args) if interface == 1 else be.draw(*args)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def time_backend(be, interface, sizes, repeats):
    out = {}
    for n in sizes:
        args = _inputs(n, interface)
        _call(be, interface, args)
        best = np.inf
        for _ in range(repeats):
            _sync(be.device)
            t = time.perf_counter()
            _call(be, interface, args)
            _sync(be.device)
            best = min(best, time.perf_counter() - t)
        out[n] = best / n * 1e6
    return out


def max_diff(be, ref, interface, n=10_000):
    """Largest |difference| from the reference backend on the same noise
    (both freshly seeded); None for padded (graph) variants."""
    if be.engine.graphs:
        return None
    args = _inputs(n, interface, seed=7)
    a = np.stack(_call(ref, interface, args))
    b = np.stack(_call(be, interface, args))
    return float(np.nanmax(np.abs(a - b)))


def run(device="cpu", families=None, seed=0, sizes=None, repeats=5,
        log=print):
    dev = torch.device(device)
    options = GPU_OPTIONS if dev.type == "cuda" else CPU_OPTIONS
    sizes = sizes or (SIZES if dev.type == "cuda" else SIZES[:3])
    rows = []
    if dev.type == "cuda":
        log(f"GPU {torch.cuda.get_device_name(dev)}, torch {torch.__version__}")
    log(f"{'model':12s} {'option':13s} " +
        " ".join(f"{n:>11,}" for n in sizes) + "   max diff vs plain")
    for fam in families or names():
        for interface in (1, 2):
            kw = {"interface": interface}
            if interface == 2:
                kw["iso"] = _Stub()
            for opt, o in options.items():
                be = load(fam, seed, device=device, **kw, **o)
                be.warmup(tuple(sizes))
                us = time_backend(be, interface, sizes, repeats)
                # accuracy: a freshly seeded pair, first call each, so both
                # use the same noise (compiling happens inside that call
                # and draws no random numbers)
                d = 0.0 if opt == "plain" else max_diff(
                    load(fam, seed, device=device, **kw, **o),
                    load(fam, seed, device=device, **kw), interface)
                name = f"{fam}{'_v2' if interface == 2 else ''}"
                log(f"{name:12s} {opt:13s} " +
                    " ".join(f"{us[n]:11.3f}" for n in sizes) +
                    ("   -" if d is None else f"   {d:.2e}"))
                rows.append({"model": name, "seed": seed, "option": opt,
                             "us_per_draw": {str(n): us[n] for n in sizes},
                             "max_diff": d, **be.engine.describe()})
    return rows


class _Stub:
    """v2 timing does not touch isotropic balls; no v1 model needed."""
    name = "stub"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--families", nargs="*")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sizes", nargs="*", type=int)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out")
    a = p.parse_args()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    rows = run(a.device, a.families, a.seed, a.sizes, a.repeats, log)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench_models.log").write_text("\n".join(lines) + "\n")
        (out / "bench_models.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
