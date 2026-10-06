"""Candidate rewrites of the flow's SAMPLING path, timed and checked.

Used by: run by hand (PyTorch and NumPy only, no OpenMC).  From gmc3d/:

    python -m checks.net_variants                     # this device
    python -m checks.net_variants --device cuda
    python -m checks.net_variants --device cuda --batches 1000 3000 10000

Uses: ball/network.py (reads its modules and weights; changes nothing).

WHY.  checks/net_profile.py showed that on a CPU only 22% of a draw is the
matrix multiplies (`aten::addmm`); the biggest single item is `aten::copy_`
at 25%, and the spline arithmetic is only 13%.  The copies come from how
one coupling writes its result:

    out = x.clone()                 # a whole new (n, 2) tensor
    out[:, self.bend] = yb          # ... then one column overwritten

plus the strided column slices `x[:, bend]` that feed the spline.  Six
couplings do that six times.  None of it is arithmetic the method needs.

Each variant below computes the SAME function with the same weights, and is
checked against the original on the same noise before it is timed, so a
speed-up can never come from quietly computing something else.  The
variants:

    columns      the two targets kept as two 1-D tensors, stacked once at
                 the end: no clone, no column write, no strided slice
    knots        + the spline's bin edges built with one cumulative sum
                 and a narrow instead of a pad and two subtractions
    compiled     + torch.compile (fuses the small elementwise chains)
    graphs       + torch.compile(mode="reduce-overhead"), which on a GPU
                 captures the call as a CUDA graph.  Needs a FIXED batch
                 size, so the batch is padded up to the next power of two
                 (the padding is thrown away; the waste is under 2x, and
                 under 1.5x with the sqrt(2) buckets `--buckets fine`)
    half         + float16 weights and activations
    bfloat16     + bfloat16

Everything prints cost per draw at each batch size, and the largest
difference from the original draw, so accuracy and speed are read together.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ball import data as D
from ball import network as N

BATCHES = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000)


# ----------------------------------------------------------------- variants
def _knots_narrow(raw, bound, minimum):
    """Same as network._knots, with one cumsum and a narrow instead of a
    pad plus two slices."""
    k = raw.shape[-1]
    size = minimum + (1 - minimum * k) * torch.softmax(raw, dim=-1)
    inner = torch.cumsum(size, dim=-1) * (2 * bound) - bound
    edges = torch.cat([torch.full_like(inner[..., :1], -bound),
                       inner[..., :-1],
                       torch.full_like(inner[..., :1], bound)], dim=-1)
    return edges[..., 1:] - edges[..., :-1], edges


def spline_1d(x, uw, uh, ud, bound, knots_fn):
    """network.rq_spline with inverse=True, for a 1-D x, returning only the
    value (sampling never needs the log-determinant)."""
    inside = (x >= -bound) & (x <= bound)
    xc = x.clamp(-bound, bound)
    w, xk = knots_fn(uw, bound, N.MIN_WIDTH)
    h, yk = knots_fn(uh, bound, N.MIN_HEIGHT)
    shift = math.log(math.expm1(1 - N.MIN_SLOPE))
    d = F.pad(N.MIN_SLOPE + F.softplus(ud + shift), (1, 1), value=1.0)
    b = (xc[..., None] >= yk[..., 1:-1]).sum(-1, keepdim=True)
    g = lambda t: t.gather(-1, b)[..., 0]
    x0, y0, wb, hb, d0, d1 = g(xk), g(yk), g(w), g(h), g(d), g(d[..., 1:])
    s = hb / wb
    dy = xc - y0
    a = hb * (s - d0) + dy * (d1 + d0 - 2 * s)
    bq = hb * d0 - dy * (d1 + d0 - 2 * s)
    c = -s * dy
    disc = (bq * bq - 4 * a * c).clamp_min(0)
    t = (2 * c) / (-bq - torch.sqrt(disc))
    return torch.where(inside, x0 + t * wb, x)


def sample_columns(net, z, c, knots_fn=N._knots):
    """net.sample with the two targets as separate 1-D tensors."""
    emb = net.embed(c)
    y = [z[:, 0].contiguous(), z[:, 1].contiguous()]
    for step in reversed(net.steps):
        bend, keep = step.bend, step.keep
        p = step.proj_out(step.norm(step.block(
            step.proj_in(y[keep][:, None]), emb)))
        k = step.bins
        y[bend] = spline_1d(y[bend], p[:, :k], p[:, k:2 * k], p[:, 2 * k:],
                            step.bound, knots_fn)
    return torch.stack(y, dim=1)


# ------------------------------------------------------------------ harness
def load(device):
    path = N.MODEL_DIR / "flow_seed0"
    if (path / "model.pt").exists():
        return N.Network.load(path, device=device), True
    info = {"r_min": N.R_MIN, "r_max": N.R_MAX}
    return N.Network(N.SplineFlow(), D.Norm(0.0, 1.0),
                     D.Norm(np.zeros(2), np.ones(2)), info,
                     device=device), False


def bucket(n, fine=False):
    """The fixed size a batch of n is padded up to."""
    if n <= 1:
        return 1
    step = math.sqrt(2.0) if fine else 2.0
    e = math.ceil(math.log(n, step))
    return int(round(step ** e))


def padded(fn, fine=False):
    """Wrap a sampler so it always sees one of a few fixed batch sizes."""
    def call(z, c):
        n = z.shape[0]
        m = bucket(n, fine)
        if m == n:
            return fn(z, c)
        zp = torch.cat([z, z[:1].expand(m - n, -1)], 0)
        cp = torch.cat([c, c[:1].expand(m - n, -1)], 0)
        return fn(zp, cp)[:n]
    return call


def timed(fn, z, c, repeats=7, cuda=False):
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    with torch.no_grad():
        fn(z, c)
        sync()
        best = np.inf
        for _ in range(repeats):
            sync()
            t = time.perf_counter()
            fn(z, c)
            sync()
            best = min(best, time.perf_counter() - t)
    return best / z.shape[0]


def build(net, device, which, fine=False):
    """{name: sampler} for the variants asked for."""
    dt = {"half": torch.float16, "bfloat16": torch.bfloat16}
    out = {}
    base = lambda z, c: net.sample(z, c)
    out["original"] = base
    if "columns" in which:
        out["columns"] = lambda z, c: sample_columns(net, z, c)
    if "knots" in which:
        out["knots"] = lambda z, c: sample_columns(net, z, c, _knots_narrow)
    if "compiled" in which:
        f = torch.compile(lambda z, c: sample_columns(net, z, c,
                                                      _knots_narrow),
                          dynamic=True)
        out["compiled"] = f
    if "graphs" in which:
        f = torch.compile(lambda z, c: sample_columns(net, z, c,
                                                      _knots_narrow),
                          mode="reduce-overhead", dynamic=False)
        out["graphs (padded)"] = padded(f, fine)
    for name in ("half", "bfloat16"):
        if name in which:
            cl = N.SplineFlow(len(net.steps),
                              net.steps[0].proj_in.out_features,
                              net.steps[0].bins,
                              net.embed[0].out_features)
            cl.load_state_dict(net.state_dict())
            cl = cl.to(device).to(dt[name]).eval()
            out[name] = (lambda m, d: lambda z, c: sample_columns(
                m, z.to(d), c.to(d), _knots_narrow).float())(cl, dt[name])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--batches", type=int, nargs="*", default=list(BATCHES))
    p.add_argument("--which", nargs="*",
                   default=["columns", "knots", "compiled", "graphs",
                            "half", "bfloat16"])
    p.add_argument("--buckets", choices=("coarse", "fine"), default="coarse")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    be, trained = load(a.device)
    dev = be.device
    cuda = dev.type == "cuda"
    name = torch.cuda.get_device_name(dev) if cuda else \
        f"CPU, {torch.get_num_threads()} threads"
    print(f"device: {name}; torch {torch.__version__}; "
          f"{'trained' if trained else 'untrained (same cost)'} network, "
          f"{sum(q.numel() for q in be.net.parameters()):,} weights")
    variants = build(be.net, dev, a.which, a.buckets == "fine")

    # ---- same answer?
    g = torch.Generator(device=dev).manual_seed(4)
    z = torch.randn(4096, 2, generator=g, device=dev)
    c = torch.zeros(4096, 1, device=dev)
    with torch.no_grad():
        ref = variants["original"](z, c).double()
        print("\nlargest difference from the original draw "
              "(same noise, same weights):")
        diffs = {}
        for k, f in variants.items():
            if k == "original":
                continue
            d = (f(z, c).double() - ref).abs().max().item()
            diffs[k] = d
            print(f"  {k:18s} {d:.2e}")

    print("\ncost per draw, ns")
    hdr = f"  {'batch':>9s} " + " ".join(f"{k[:14]:>15s}"
                                        for k in variants)
    print(hdr)
    table = {}
    for b in a.batches:
        z = torch.randn(b, 2, generator=g, device=dev)
        c = torch.zeros(b, 1, device=dev)
        row = {}
        for k, f in variants.items():
            row[k] = timed(f, z, c, cuda=cuda)
        table[b] = row
        print(f"  {b:>9,} " + " ".join(f"{1e9 * row[k]:15.0f}"
                                      for k in variants))
    print("\nspeed-up over the original")
    print(hdr)
    for b, row in table.items():
        print(f"  {b:>9,} " + " ".join(
            f"{row['original'] / row[k]:14.2f}x" for k in variants))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"device": name, "torch": torch.__version__, "diffs": diffs,
             "ns_per_draw": {str(b): {k: 1e9 * v for k, v in r.items()}
                             for b, r in table.items()}}, indent=1))
        print(f"\nwritten to {a.json}")


if __name__ == "__main__":
    main()
