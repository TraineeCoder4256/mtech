"""What does a draw cost as the problem grows more inputs and outputs?

Used by: run by hand.  From gmc3d/:

    python -m checks.feature_cost --device cpu
    python -m checks.feature_cost --device cuda --compile     # on Colab

Today's flow is the smallest the ball formulation allows: ONE condition
(the radius in mean free paths) and TWO targets (exit cosine, path length).
Stage 2 and beyond add both.  Anisotropic scattering adds the incident
direction as a condition and a second exit angle as a target.  Energy adds
the entry energy, the material and the anisotropy as conditions, and the
exit energy plus a PER-GROUP path length as targets -- the track-length
estimator needs the path resolved by group, not one scalar.

The question this answers is not "does it cost more" (it does) but WHICH
growth costs: conditions, targets, or the depth needed to mix them.  So the
three are swept separately, with everything else held at today's value.

Nothing here is trained.  Untrained weights cost exactly what trained ones
cost, and `checks/net_profile.py` showed the draw is dominated by
framework overhead (772 operations, only 22% of them matrix multiplies, the
flow at 8.4% of this machine's matmul peak), so timing is the whole point
and accuracy is not on trial.  What a given size can LEARN is a separate
question and this script says nothing about it.
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ball import network as N

# today's flow, for reference: 1 condition, 2 targets, 6 couplings
TODAY = dict(cond=1, dim=2, layers=N.LAYERS)


class GenCoupling(nn.Module):
    """One coupling step for `dim` targets instead of exactly two.

    Half the targets (by parity) are left alone and supply the spline
    parameters for the other half, which is what Coupling in
    ball/network.py does for dim = 2.  At dim = 2 the two agree in shape
    and in weight count.
    """

    def __init__(self, parity, dim, width, emb, bins, bound):
        super().__init__()
        idx = torch.arange(dim)
        self.register_buffer("keep_idx", idx[idx % 2 == parity])
        self.register_buffer("bend_idx", idx[idx % 2 != parity])
        self.bins, self.bound = bins, bound
        self.n_bend = int(self.bend_idx.numel())
        self.proj_in = nn.Linear(int(self.keep_idx.numel()), width)
        self.block = N.FiLMBlock(width, emb)
        self.norm = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, self.n_bend * (3 * bins - 1))
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, emb):
        p = self.proj_out(self.norm(self.block(
            self.proj_in(x[:, self.keep_idx]), emb)))
        p = p.view(x.shape[0], self.n_bend, 3 * self.bins - 1)
        k = self.bins
        out = x.clone()
        for j in range(self.n_bend):
            q = p[:, j]
            yb, _ = N.rq_spline(x[:, self.bend_idx[j]], q[:, :k],
                                q[:, k:2 * k], q[:, 2 * k:],
                                inverse=True, bound=self.bound)
            out[:, self.bend_idx[j]] = yb
        return out


class GenFlow(nn.Module):
    """p(y | c) for `dim` targets and `cond` conditions."""

    def __init__(self, cond, dim, layers, width=N.WIDTH, bins=N.BINS,
                 emb=N.EMB, bound=N.BOUND):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(cond, emb), nn.SiLU(),
                                   nn.Linear(emb, emb), nn.SiLU())
        self.steps = nn.ModuleList(
            GenCoupling(i % 2, dim, width, emb, bins, bound)
            for i in range(layers))

    def sample(self, z, c):
        emb = self.embed(c)
        for step in reversed(self.steps):
            z = step(z, emb)
        return z


def time_one(cond, dim, layers, batch, device, compile_=False, reps=15):
    """Seconds per draw, and the weight count."""
    torch.manual_seed(0)
    net = GenFlow(cond, dim, layers).to(device).eval()
    weights = sum(p.numel() for p in net.parameters())
    fn = net.sample
    if compile_:
        fn = torch.compile(net.sample, dynamic=False)
    z = torch.randn(batch, dim, device=device)
    c = torch.rand(batch, cond, device=device) * 2 - 1
    with torch.no_grad():
        for _ in range(3):
            fn(z, c)
        if device != "cpu":
            torch.cuda.synchronize()
        best = math.inf
        for _ in range(reps):
            t = time.perf_counter()
            fn(z, c)
            if device != "cpu":
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
    return best / batch, weights


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batches", type=int, nargs="*",
                   default=[12853, 100000])
    a = p.parse_args()
    torch.set_num_threads(torch.get_num_threads())

    ref = {}

    def row(label, cond, dim, layers, is_ref=False):
        cells = []
        for b in a.batches:
            ns, w = time_one(cond, dim, layers, b, a.device, a.compile)
            if is_ref:
                ref[b] = ns
            x = ns / ref[b] if ref.get(b) else float("nan")
            cells.append(f"{1e9 * ns:8.0f} {x:5.2f}x")
        print(f"  {label:36s} {w:>9,} " + " ".join(cells), flush=True)

    head = (f"  {'variant':36s} {'weights':>9s} "
            + " ".join(f"{str(b) + ' req':>14s}" for b in a.batches))
    print(f"\ndevice {a.device}, compile {a.compile}; nanoseconds per draw "
          f"and the ratio to the first row of each block.\nThe absolute "
          f"level drifts on a shared CPU (the same shape measured in four\n"
          f"blocks spread 1.6x), so read the RATIOS, which are taken inside "
          f"one block.\n")

    print("== more CONDITIONS (inputs), targets and depth held at today's")
    print(head)
    for c in (1, 2, 5, 10, 20):
        row(f"{c} cond, 2 targets, 6 steps"
            + ("   <- today" if c == 1 else ""), c, 2, N.LAYERS,
            is_ref=(c == 1))

    print("\n== more TARGETS (outputs), conditions and depth held")
    print(head)
    for d in (2, 4, 8, 16, 24):
        row(f"1 cond, {d} targets, 6 steps"
            + ("    <- today" if d == 2 else ""), 1, d, N.LAYERS,
            is_ref=(d == 2))

    print("\n== more DEPTH (couplings), width unchanged")
    print(head)
    for L in (6, 8, 12, 16, 24):
        row(f"1 cond, 2 targets, {L} steps"
            + ("   <- today" if L == N.LAYERS else ""), 1, 2, L,
            is_ref=(L == N.LAYERS))

    print("\n== the two realistic Stage 2+ shapes, against today")
    print(head)
    row("today: 1 cond, 2 targets, 6", *TODAY.values(), is_ref=True)
    row("anisotropic: 2 cond, 4 targets, 8", 2, 4, 8)
    row("energy, 20 groups: 5 cond, 24 t, 16", 5, 24, 16)


if __name__ == "__main__":
    main()
