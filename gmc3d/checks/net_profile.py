"""Anatomy of one backend call: where the network's time and traffic go.

Used by: run by hand, on any machine with PyTorch and NumPy (no OpenMC).
From gmc3d/:

    python -m checks.net_profile                      # this device
    python -m checks.net_profile --device cuda
    python -m checks.net_profile --device cuda --compile
    python -m checks.net_profile --device cuda --dtype float16

Uses: ball/network.py (the flow and the backend), ball/data.py (encoding).
Nothing here changes those files; everything is measured from outside.

checks/gpu_bench.py answers "what does a draw cost on this device".  This
file answers "and what is that cost MADE OF", which is what decides which
GPU optimisation is worth doing:

 1. INVENTORY.  Weights, matrix multiplies, multiply-adds and bytes of
    activation traffic per draw, counted by walking the real modules and by
    recording every tensor each operation produces.  Gives the two roofs a
    device has: arithmetic (MAC/s) and memory bandwidth (bytes/s).

 2. STAGES.  The backend call split into host encode (NumPy log and
    standardise), the copy to the device, the noise draw, the flow itself,
    the copy back, and the host decode (NumPy exp).  Only the flow moves to
    a GPU; everything else is the floor that stays behind.

 3. OPERATIONS.  The torch profiler's own count and ranking, so the
    expensive kinds of operation are named rather than guessed.

 4. VARIANTS.  float32 / float16 / bfloat16, eager / compiled, and a
    "weights only" run with the splines replaced by a plain affine map,
    which separates the cost of the small matrix multiplies from the cost
    of the spline arithmetic around them.

Read the printed roofs first: if measured bytes/s is near the device's
bandwidth, the flow is memory-bound and only less traffic helps; if
measured MAC/s is near the device's arithmetic peak, only fewer weights do.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ball import data as D
from ball import network as N

BATCHES = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


# ------------------------------------------------------------------ inventory
def macs_and_weights(net):
    """Multiply-adds and weights per draw, from the real Linear layers."""
    macs = weights = 0
    layers = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features
            weights += m.weight.numel() + (0 if m.bias is None
                                           else m.bias.numel())
            layers.append((name, m.in_features, m.out_features))
    total = sum(p.numel() for p in net.parameters())
    return {"mac_per_draw": macs, "linear_weights": weights,
            "all_weights": total, "n_linear": len(layers), "layers": layers}


class Traffic(torch.overrides.TorchFunctionMode):
    """Counts every torch operation and the bytes of tensor it produces."""

    def __init__(self):
        super().__init__()
        self.ops = Counter()
        self.bytes_out = 0
        self.n = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.n += 1
        self.ops[getattr(func, "__name__", str(func))] += 1
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if isinstance(t, torch.Tensor):
                self.bytes_out += t.numel() * t.element_size()
        return out


def inventory(net, batch=4096, device="cpu"):
    """Ops and produced bytes for one sample() call, per draw."""
    dev = torch.device(device)
    z = torch.randn(batch, 2, device=dev)
    c = torch.zeros(batch, 1, device=dev)
    with torch.no_grad(), Traffic() as t:
        net.sample(z, c)
    return {"ops_per_call": t.n, "ops_by_kind": t.ops.most_common(12),
            "bytes_written_per_draw": t.bytes_out / batch}


# --------------------------------------------------------------------- stages
def stages(be, batch, repeats=7):
    """Seconds per draw for each part of Network.sample, timed apart.

    Mirrors ball/network.py's sample() step by step; it is not called, so
    that file stays untouched.
    """
    R = np.full(batch, 5.0)
    dev = be.device
    cuda = dev.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def best(fn):
        fn()
        out = np.inf
        for _ in range(repeats):
            sync()
            t = time.perf_counter()
            fn()
            sync()
            out = min(out, time.perf_counter() - t)
        return out / batch

    # the pieces, each closed over what the previous one produces
    enc = lambda: be.cnorm(np.log(R)[:, None]).astype(np.float32)
    c_np = enc()
    to_dev = lambda: torch.from_numpy(c_np).to(dev)
    c_t = to_dev()
    noise = lambda: torch.randn(batch, 2, generator=be.gen, device=dev)
    z_t = noise()
    with torch.no_grad():
        flow = lambda: be.net.sample(z_t, c_t)
        y_t = flow()
        back = lambda: y_t.cpu().numpy().astype(np.float64)
        y_np = back()
        dec = lambda: D.decode(R, be.ynorm.undo(y_np))
        whole = lambda: be.sample(R)
        out = {"host encode": best(enc), "copy to device": best(to_dev),
               "noise": best(noise), "the flow": best(flow),
               "copy back": best(back), "host decode": best(dec),
               "WHOLE CALL": best(whole)}
    return out


# ----------------------------------------------------------------- profiler
def op_table(be, batch=100_000, top=12):
    """The torch profiler's ranking of operations in one call."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU]
    if be.device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    R = np.full(batch, 5.0)
    be.sample(R)
    with profile(activities=acts) as pr:
        be.sample(R)
    key = "self_cuda_time_total" if be.device.type == "cuda" \
        else "self_cpu_time_total"
    rows = []
    for e in pr.key_averages():
        rows.append((e.key, e.count, getattr(e, key, 0) / 1e3))
    rows.sort(key=lambda r: -r[2])
    total = sum(r[2] for r in rows if r[2] > 0)
    return rows[:top], total, sum(r[1] for r in rows)


# ----------------------------------------------------------------- variants
class Affine(nn.Module):
    """A coupling's spline replaced by a shift and a scale: the same matrix
    multiplies, none of the spline arithmetic."""

    def __init__(self, step):
        super().__init__()
        self.inner = step
        self.bend, self.keep = step.bend, step.keep

    def forward(self, x, emb, inverse=False):
        s = self.inner
        p = s.proj_out(s.norm(s.block(
            s.proj_in(x[:, s.keep:s.keep + 1]), emb)))
        out = x.clone()
        out[:, s.bend] = x[:, s.bend] * (1.0 + p[:, 0]) + p[:, 1]
        return out, p[:, 0]


def cost(be, batch, repeats=7):
    R = np.full(batch, 5.0)
    be.sample(R)
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        be.sample(R)
        best = min(best, time.perf_counter() - t)
    return best / batch


def variants(device, batch, compile_it=False):
    """Cost per draw for each dtype, and with the splines removed."""
    out = {}
    for name, dt in DTYPES.items():
        be, _ = load(device)
        try:
            be.net = be.net.to(dt)
            be.net.sample = _cast_wrap(be.net, dt)
            out[name] = cost(be, batch)
        except Exception as e:                       # unsupported on device
            out[name] = f"failed: {type(e).__name__}"
    be, _ = load(device)
    be.net.steps = nn.ModuleList(Affine(s) for s in be.net.steps)
    out["float32, no splines"] = cost(be, batch)
    if compile_it:
        be, _ = load(device)
        be.net.sample = torch.compile(be.net.sample, dynamic=True)
        out["float32, compiled"] = cost(be, batch)
    return out


def _cast_wrap(net, dt):
    inner = type(net).sample
    def sample(z, c):
        return inner(net, z.to(dt), c.to(dt)).float()
    return sample


def load(device):
    path = N.MODEL_DIR / "flow_seed0"
    if (path / "model.pt").exists():
        return N.Network.load(path, device=device), True
    info = {"r_min": N.R_MIN, "r_max": N.R_MAX}
    return N.Network(N.SplineFlow(), D.Norm(0.0, 1.0),
                     D.Norm(np.zeros(2), np.ones(2)), info,
                     device=device), False


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch", type=int, default=100_000,
                   help="batch for the stage and variant tables")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    be, trained = load(a.device)
    if a.compile:
        be.net.sample = torch.compile(be.net.sample, dynamic=True)
    dev = be.device
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else \
        f"CPU, {torch.get_num_threads()} threads"
    print(f"device: {name}; torch {torch.__version__}; "
          f"{'trained' if trained else 'untrained (same cost)'}"
          f"{'; compiled' if a.compile else ''}")

    inv = macs_and_weights(be.net)
    tr = inventory(be.net, 4096, a.device)
    print(f"\n1. INVENTORY per draw")
    print(f"  weights          {inv['all_weights']:,} "
          f"({inv['n_linear']} Linear layers)")
    print(f"  multiply-adds    {inv['mac_per_draw']:,} "
          f"= {2 * inv['mac_per_draw'] / 1e3:.0f} kFLOP")
    print(f"  torch operations {tr['ops_per_call']:,} per call")
    print(f"  tensor bytes written {tr['bytes_written_per_draw']:,.0f} "
          f"(reads are at least as much again)")
    print("  the operations, by how often: " + ", ".join(
        f"{k.split('.')[-1]} x{v}" for k, v in tr["ops_by_kind"][:8]))

    print(f"\n2. STAGES of one backend call, ns per draw "
          f"(batch {a.batch:,})")
    st = stages(be, a.batch)
    whole = st.pop("WHOLE CALL")
    for k, v in st.items():
        print(f"  {k:16s} {1e9 * v:8.1f}  ({100 * v / whole:4.1f}%)")
    print(f"  {'WHOLE CALL':16s} {1e9 * whole:8.1f}")
    host = st["host encode"] + st["host decode"]
    print(f"  -> host NumPy work is {1e9 * host:.0f} ns per draw: the floor "
          f"a faster device cannot pass")
    mac_rate = inv["mac_per_draw"] / st["the flow"]
    byte_rate = tr["bytes_written_per_draw"] / st["the flow"]
    print(f"  -> the flow runs at {mac_rate / 1e9:.1f} GMAC/s and writes "
          f"{byte_rate / 1e9:.1f} GB/s")

    print(f"\n3. OPERATIONS the profiler sees (batch {a.batch:,}, ms)")
    rows, total, calls = op_table(be, a.batch)
    print(f"  {'operation':28s} {'count':>7s} {'ms':>8s} {'share':>7s}")
    for k, n, ms in rows:
        print(f"  {k[:28]:28s} {n:7d} {ms:8.2f} {100 * ms / total:6.1f}%")
    print(f"  total {total:.2f} ms over {calls} recorded calls")

    print(f"\n4. COST PER DRAW by batch size, ns")
    by_batch = {}
    for b in BATCHES:
        by_batch[b] = cost(be, b)
        print(f"  batch {b:>9,}: {1e9 * by_batch[b]:8.0f}")

    print(f"\n5. VARIANTS at batch {a.batch:,}, ns per draw")
    var = variants(a.device, a.batch, a.compile)
    for k, v in var.items():
        print(f"  {k:22s} " + (f"{1e9 * v:8.0f}" if isinstance(v, float)
                               else str(v)))

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"device": name, "torch": torch.__version__,
             "inventory": {**inv, **tr}, "stages": st, "whole_call": whole,
             "by_batch": by_batch,
             "variants": {k: v for k, v in var.items()}},
            indent=1, default=str))
        print(f"\nwritten to {a.json}")


if __name__ == "__main__":
    main()
