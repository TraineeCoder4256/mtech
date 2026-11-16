"""Drawing from a trained family at speed, on a GPU or the CPU (m2.4).

Used by: backend.py (every backend draws through an Engine).
Uses: the family's noise() and draw_from() (base.py), PyTorch.

A backend call is: encode the conditions on the host, send them to the
device, draw, bring the samples back, decode on the host.  With the
defaults (fp32, no compile, no graphs) the Engine does exactly what
backend.py always did -- `family.draw(net, c, generator)` -- so every
accepted model draws the same numbers as before, bit for bit.

The options, each its own switch so the benchmark can show what each buys
and gate2 can judge each for accuracy:

  precision="fp16"  run the network under autocast in half precision: the
                    matrix products go to the GPU's tensor cores, while
                    exp, log, softmax and sums stay in float32 (mixed
                    precision).  GPU only.  ("fp32" is the default; "tf32"
                    lets A100-class GPUs use TensorFloat-32 products.)
  compile=True      torch.compile of the draw: the dozens of small
                    operations in each layer are fused into a few larger
                    kernels.  On a T4 the Speed thread measured 4-5x for
                    the flow from this alone (checks/net_variants.py).
  graphs=True       CUDA graphs (torch.compile mode "reduce-overhead"): the
                    whole draw is recorded once per batch size and then
                    replayed as one launch, removing the CPU cost of
                    launching hundreds of kernels.  A graph needs a fixed
                    batch size, so a batch of n is padded up to the next
                    size in a ladder of powers of sqrt(2) (at most 41%
                    padding, about 17% on average; nothing below 256) and
                    the padding is thrown away.  The kernels are compiled
                    once for every size; only the cheap recording is per
                    size.  GPU only; implies compile.

Note: precision="tf32" / "fp32" set torch's global TF32 switch for matrix
products on a GPU (it is process-wide in torch).

What stays the same with any option: the noise is drawn eagerly from the
backend's own seeded generator before the compiled part runs (so
compiling never changes WHICH random numbers are used, only how the
network turns them into samples), and the decode back to physical
outputs stays in float64 on the host.

Host-device traffic: conditions are staged in pinned (page-locked) host
memory and copied with non_blocking=True, and results come back the same
way, so a copy never goes through an extra bounce buffer.  An analogy for
the whole file: a courier who packs the parcels before the van arrives,
drives one fixed route that was planned once, and unloads straight onto
the shelf.
"""

import math

import numpy as np
import torch

PRECISIONS = ("fp32", "tf32", "fp16")


def bucket(n):
    """The batch size a graph is recorded for: n rounded up to the ladder
    1, 2, 3, 4, 6, 8, 12, 16, 23, 32, 46, 64, ... (powers of sqrt(2),
    rounded up)."""
    if n <= 1:
        return 1
    k = max(0, math.ceil(2 * math.log2(n)) - 1)
    m = math.ceil(2.0 ** (k / 2))
    return m if m >= n else math.ceil(2.0 ** ((k + 1) / 2))


MIN_GRAPH = 256     # graphs pad small batches at least up to this: a tiny
                    # draw costs the same on a GPU, and fewer sizes means
                    # fewer graphs to record


class Engine:
    """family + network + options -> a function from encoded conditions
    (numpy float32, (n, c_dim)) to encoded samples (numpy float64)."""

    def __init__(self, family, net, c_dim, x_dim, device="cpu",
                 precision="fp32", compile=False, graphs=False, seed=12345):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.family, self.c_dim, self.x_dim = family, c_dim, x_dim
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError("device cuda asked for, but torch sees no GPU")
        if (graphs or precision == "fp16") and not self.cuda:
            raise ValueError("graphs and fp16 need a GPU (device='cuda')")
        self.precision = precision
        self.compile = bool(compile or graphs)
        self.graphs = bool(graphs)
        self.net = net.to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.plain = not (self.compile or precision != "fp32" or self.cuda)
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        fn = lambda c, *z: family.draw_from(self.net, c, *z)   # noqa: E731
        if self.compile:
            # dynamic=True: the kernels are compiled ONCE for any batch
            # size (dynamic=False would recompile per size and, past
            # dynamo's recompile limit, quietly fall back to eager).  With
            # graphs, one CUDA graph is recorded per padded bucket size;
            # recording is cheap next to compiling.
            fn = torch.compile(fn, dynamic=True,
                               mode="reduce-overhead" if self.graphs
                               else None)
        self._fn = fn
        self._slots = [_Slot(), _Slot()]   # two calls may be in flight
        self._next = 0
        self.recorded = set()             # batch sizes with a graph

    def describe(self):
        return {"device": str(self.device), "precision": self.precision,
                "compile": self.compile, "graphs": self.graphs}

    @torch.no_grad()
    def __call__(self, c):
        """Encoded conditions -> encoded samples (numpy float64)."""
        return self.collect(self.submit(c))

    @torch.no_grad()
    def submit(self, c):
        """Start a draw and return at once with a ticket; collect(ticket)
        gives the samples.  On a GPU the work runs while the caller does
        something else (transport overlaps it with tracking).  Up to two
        tickets may be outstanding; tickets are collected in any order.
        The random numbers are taken at submit time, so a sequence of
        submits draws exactly what the same sequence of calls would."""
        c = np.ascontiguousarray(c, np.float32)
        n = c.shape[0]
        if self.plain:                     # the original path, bit for bit
            ct = torch.from_numpy(c).to(self.device)
            y = self.family.draw(self.net, ct, self.gen)
            return _Done(y.cpu().numpy().astype(np.float64))
        if n == 0:
            return _Done(np.zeros((0, self.x_dim)))
        slot = self._slots[self._next]
        self._next ^= 1
        if slot.busy:
            raise RuntimeError("more than two draws in flight: collect a "
                               "ticket before submitting another")
        ct = slot.stage(c, self.device) if self.cuda else torch.from_numpy(c)
        m = bucket(max(n, MIN_GRAPH)) if self.graphs else n
        if m > n:
            ct = torch.cat([ct, ct[:1].expand(m - n, -1)], 0)
        if self.graphs:
            self.recorded.add(m)
            torch.compiler.cudagraph_mark_step_begin()
        z = self.family.noise(self.net, m, self.gen, self.device)
        with torch.autocast(self.device.type, dtype=torch.float16,
                            enabled=self.precision == "fp16"):
            y = self._fn(ct, *z)
        y = y[:n].float()
        if not self.cuda:
            return _Done(y.numpy().astype(np.float64))
        return slot.send_back(y)

    @staticmethod
    def collect(ticket):
        """Wait for a ticket's draw; numpy float64 (n, x_dim)."""
        return ticket.result()

    def warmup(self, sizes=(1024, 4096, 16384, 65536)):
        """Compile (and, with graphs, record) ahead of the first real call,
        so a timed run does not pay for it.  Uses the generator, so call it
        before drawing anything that must be reproducible."""
        if self.plain:
            return
        for n in sizes:
            for _ in range(3 if self.graphs else 1):
                self(np.zeros((n, self.c_dim), np.float32))


class _Done:
    """A ticket whose result is already on the host."""

    def __init__(self, y):
        self.y = y

    def result(self):
        return self.y


class _Slot:
    """One pinned host buffer pair for a call in flight, and the CUDA event
    that says when the call's results have landed in it."""

    def __init__(self):
        self.inp = self.out = self.event = None
        self.busy = False                 # a ticket is still to collect

    def stage(self, c, device):
        n, d = c.shape
        if self.inp is None or self.inp.shape[0] < n or self.inp.shape[1] != d:
            self.inp = torch.empty((max(n, 1024), d),
                                   dtype=torch.float32).pin_memory()
        buf = self.inp[:n]
        buf.numpy()[:] = c
        return buf.to(device, non_blocking=True)

    def send_back(self, y):
        n, d = y.shape
        if self.out is None or self.out.shape[0] < n or self.out.shape[1] != d:
            self.out = torch.empty((max(n, 1024), d),
                                   dtype=torch.float32).pin_memory()
        out = self.out[:n]
        out.copy_(y, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record()
        self.busy = True
        return _Pending(self, out)


class _Pending:
    """A ticket whose draw is still running on the GPU."""

    def __init__(self, slot, out):
        self.slot, self.out = slot, out

    def result(self):
        self.slot.event.synchronize()
        y = self.out.numpy().astype(np.float64)     # a copy: slot is free
        self.slot.busy = False
        return y
