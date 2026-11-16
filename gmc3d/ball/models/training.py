"""The training loop every family shares.

Used by: base.py (Family.fit), so every family that trains by "one scalar
loss per batch" -- the flow, flow matching, a VAE, diffusion.
Uses: PyTorch only.

From the outside all those families train the same way: draw a random
batch, compute ONE number that says how wrong the network is, take a
gradient step, repeat.  They differ only in the network and in that number
(a negative log-likelihood, a velocity error, an evidence bound).  So a
family writes those two things and hands them to fit_loop(); everything
else is here, in one copy, so that no family can win the comparison by
being given a better optimiser, a longer warm-up or a kinder validation:

  optimiser   AdamW, weight decay 1e-5, gradient norm clipped to 1
  schedule    linear warm-up, then cosine decay to zero at the last step
  EMA         a moving average of the weights; it is what gets saved,
              validated and sampled, because it is smoother than the raw
              weights
  validation  every val_every steps on 65,536 held-out walks
  time cap    stop early when the wall clock passes time_cap; the step
              reached is recorded in the facts

This is the loop that was inside ball/network.py's train() (gmc3d
v1.6.1), operation for operation and random number for random number: the
seed, then the network build, then the EMA and optimiser, then one
torch.Generator for the batch and validation indices.  So the flow trains
bit-identically to before (shown before this file was pushed).  It is also
the same recipe as gmc2d's generators/training.py.

A GAN does NOT fit this shape (two networks, two optimisers, two losses):
a GAN family overrides Family.fit and copies these conventions by hand,
using the device helpers below.

ON A GPU (m2.3).  device="cuda" changes WHERE the work runs, not the
recipe (same batch, steps, optimiser, schedule, EMA), so the fairness
rules still hold.  What changes, and why:

  rows on the GPU   the training rows are copied to the GPU once (v2:
                    37M x 17 float32 = 2.5 GB) and each batch is picked
                    there, so no step waits for a host-to-GPU copy.  If
                    they do not fit, they stay in pinned host memory and
                    each batch is copied without blocking.
  indices           drawn by a torch.Generator on the same device as the
                    rows, seeded with the run's seed.  So a GPU run is
                    reproducible on that GPU but is NOT the CPU run's random
                    numbers: a GPU-trained model is a new model.
  no waiting        the loss is only read back to the host (.item(), which
                    makes the CPU wait for the GPU) at validation steps; the
                    smoothed training loss is kept on the GPU in between.
  fused kernels     AdamW(fused=True) and one multi-tensor EMA update
                    instead of a Python loop over weight tensors.
  precision         "fp32" (default, exact); "tf32" lets matrix products
                    use TensorFloat-32 on GPUs that have it (A100 and
                    newer; a T4 does not); "fp16" runs the forward pass in
                    half precision under autocast with a gradient scaler
                    (mixed precision: weights, optimiser and the
                    exp/log/sum reductions stay float32).
  compile           torch.compile of the loss: the network's small
                    operations are fused into fewer, larger kernels.  The
                    first steps are slower while it compiles.

With device="cpu", precision="fp32" and compile=False (the defaults) the
loop is the one above, operation for operation: CPU runs reproduce every
model trained before m2.3 bit for bit.
"""

import copy
import time

import numpy as np
import torch

VAL_ROWS = 65536
PRECISIONS = ("fp32", "tf32", "fp16")


class Device:
    """Where a training run executes, and the speed options that go with it.

    The CPU defaults change nothing; see the module docstring."""

    def __init__(self, device="cpu", precision="fp32", compile=False,
                 log=print):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.dev = torch.device(device)
        self.cuda = self.dev.type == "cuda"
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError("device cuda asked for, but torch sees no GPU "
                               "(torch.cuda.is_available() is False)")
        self.precision, self.compile, self.log = precision, compile, log
        self.amp = precision == "fp16" and self.cuda
        if precision == "fp16" and not self.cuda:
            log("precision fp16 needs a GPU; training in fp32")
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
            torch.backends.cudnn.allow_tf32 = precision == "tf32"
        self.on_device = False

    @property
    def plain(self):
        """True for the original CPU loop (bit-identical to m2.2)."""
        return not self.cuda and not self.compile

    def describe(self):
        d = {"device": str(self.dev), "precision": self.precision,
             "compile": bool(self.compile)}
        if self.cuda:
            d["gpu"] = torch.cuda.get_device_name(self.dev)
            d["rows_on_gpu"] = self.on_device
        return d

    def rows(self, *arrays):
        """numpy arrays -> tensors where the batches will be drawn: on the
        GPU if they fit (keeping 2 GB free for the network), else pinned
        host memory."""
        ts = [torch.from_numpy(a) for a in arrays]
        if not self.cuda:
            return ts
        need = sum(t.numel() * t.element_size() for t in ts)
        free, _ = torch.cuda.mem_get_info(self.dev)
        self.on_device = need + (2 << 30) < free
        if self.on_device:
            self.log(f"rows   {need / 2**30:.2f} GB copied to the GPU "
                     f"({free / 2**30:.1f} GB free)")
            return [t.to(self.dev) for t in ts]
        self.log(f"rows   {need / 2**30:.2f} GB kept in pinned host memory "
                 f"(GPU has {free / 2**30:.1f} GB free)")
        return [t.pin_memory() for t in ts]

    def generator(self, seed):
        """The batch/validation index generator: on the rows' device."""
        if self.on_device:
            return torch.Generator(device=self.dev).manual_seed(seed)
        return torch.Generator().manual_seed(seed)

    def batch(self, n, size, gen, *rows):
        """`size` random rows of each tensor in `rows`, on the device."""
        if self.on_device:
            idx = torch.randint(0, n, (size,), generator=gen,
                                device=self.dev)
            return [r[idx] for r in rows]
        idx = torch.randint(0, n, (size,), generator=gen)
        if self.cuda:
            return [r[idx].pin_memory().to(self.dev, non_blocking=True)
                    for r in rows]
        return [r[idx] for r in rows]

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16,
                              enabled=self.amp)

    def scaler(self):
        return torch.amp.GradScaler("cuda", enabled=self.amp)

    def optimizer(self, params, **kw):
        """AdamW, fused into one kernel on a GPU."""
        cls = kw.pop("cls", torch.optim.AdamW)
        if self.cuda:
            kw["fused"] = True
        return cls(params, **kw)

    def wrap(self, fn):
        if not self.compile:
            return fn
        return torch.compile(fn)


class EMA:
    """Moving average of the weights; sampling uses it, training does not."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.shadow.parameters(), model.parameters()):
            a.mul_(self.decay).add_(b, alpha=1 - self.decay)
        for a, b in zip(self.shadow.buffers(), model.buffers()):
            a.copy_(b)                  # buffers are constants (e.g. Fourier
                                        # frequencies); none in the flow

    @torch.no_grad()
    def update_fused(self, model):
        """The same update as one multi-tensor kernel per step (GPU)."""
        a = list(self.shadow.parameters())
        torch._foreach_mul_(a, self.decay)
        torch._foreach_add_(a, list(model.parameters()),
                            alpha=1 - self.decay)
        for x, y in zip(self.shadow.buffers(), model.buffers()):
            x.copy_(y)


def fit_loop(build, loss_fn, data, *, steps, batch, lr, warmup, ema_decay,
             val_every, seed, time_cap=float("inf"), log=print,
             loss_name="loss", device="cpu", precision="fp32",
             compile=False):
    """Train build() on data with loss_fn.  Returns (EMA network, facts).

    build()             -> a fresh torch.nn.Module, called right after
                           torch.manual_seed(seed), so the seed fixes its
                           starting weights
    loss_fn(net, y, c)  -> a scalar loss on a batch of standardised targets
                           y and conditions c.  It may draw its own random
                           numbers (noise, a time) from torch's global
                           generator, which the seed also fixes
    data                the dict encoding.prepare() returns
    device, precision, compile   where and how fast (module docstring);
                           the network comes back on the CPU either way
    """
    dv = Device(device, precision, compile, log)
    if not dv.plain:
        return _fit_loop_device(dv, build, loss_fn, data, steps=steps,
                                batch=batch, lr=lr, warmup=warmup,
                                ema_decay=ema_decay, val_every=val_every,
                                seed=seed, time_cap=time_cap, log=log,
                                loss_name=loss_name)
    torch.manual_seed(seed)
    y, c = torch.from_numpy(data["y_train"]), torch.from_numpy(data["c_train"])
    yv, cv = torch.from_numpy(data["y_val"]), torch.from_numpy(data["c_val"])
    net = build()
    n_params = sum(p.numel() for p in net.parameters())
    ema = EMA(net, ema_decay)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule(warmup, steps))
    gen = torch.Generator().manual_seed(seed)
    hist, smooth, t1, done = [], None, time.time(), 0
    for step in range(1, steps + 1):
        idx = torch.randint(0, y.shape[0], (batch,), generator=gen)
        loss = loss_fn(net, y[idx], c[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(net)
        smooth = loss.item() if smooth is None else \
            0.98 * smooth + 0.02 * loss.item()
        done = step
        capped = time.time() - t1 > time_cap
        if step % val_every == 0 or step == steps or capped:
            with torch.no_grad():
                v = torch.randint(0, yv.shape[0], (VAL_ROWS,), generator=gen)
                vloss = loss_fn(ema.shadow, yv[v], cv[v]).item()
            hist.append((step, smooth, vloss))
            log(f"step {step:6d}  train {loss_name} {smooth:.4f}  "
                f"val {loss_name} (EMA) {vloss:.4f}  {time.time() - t1:.0f} s")
        if capped:
            log(f"time cap of {time_cap:.0f} s reached at step {step}")
            break
    return ema.shadow.eval(), {
        "steps": steps, "steps_done": done, "seed": seed,
        "params": int(n_params), "train_seconds": time.time() - t1,
        "history": hist, "loss_name": loss_name}


def schedule(warmup, steps):
    """Linear warm-up, then cosine decay to zero at the last step."""
    return lambda s: min(1.0, (s + 1) / warmup) * \
        0.5 * (1 + np.cos(np.pi * min(1.0, s / steps)))


def _fit_loop_device(dv, build, loss_fn, data, *, steps, batch, lr, warmup,
                     ema_decay, val_every, seed, time_cap, log, loss_name):
    """fit_loop on a GPU (or compiled): same recipe, nothing waits."""
    torch.manual_seed(seed)
    y, c, yv, cv = dv.rows(data["y_train"], data["c_train"],
                           data["y_val"], data["c_val"])
    net = build().to(dv.dev)
    n_params = sum(p.numel() for p in net.parameters())
    ema = EMA(net, ema_decay)
    opt = dv.optimizer(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule(warmup, steps))
    scaler = dv.scaler()
    gen = dv.generator(seed)
    lossc = dv.wrap(loss_fn)
    log(f"device {dv.describe()}")
    hist, smooth, t1, done = [], None, time.time(), 0
    for step in range(1, steps + 1):
        yb, cb = dv.batch(y.shape[0], batch, gen, y, c)
        with dv.autocast():
            loss = lossc(net, yb, cb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update_fused(net) if dv.cuda else ema.update(net)
        ld = loss.detach().float()
        smooth = ld if smooth is None else 0.98 * smooth + 0.02 * ld
        done = step
        capped = time.time() - t1 > time_cap
        if step % val_every == 0 or step == steps or capped:
            with torch.no_grad(), dv.autocast():
                vy, vc = dv.batch(yv.shape[0], VAL_ROWS, gen, yv, cv)
                vloss = loss_fn(ema.shadow, vy, vc).item()
            sm = smooth.item()
            hist.append((step, sm, vloss))
            log(f"step {step:6d}  train {loss_name} {sm:.4f}  "
                f"val {loss_name} (EMA) {vloss:.4f}  {time.time() - t1:.0f} s")
        if capped:
            log(f"time cap of {time_cap:.0f} s reached at step {step}")
            break
    return ema.shadow.cpu().eval(), {
        "steps": steps, "steps_done": done, "seed": seed,
        "params": int(n_params), "train_seconds": time.time() - t1,
        "history": hist, "loss_name": loss_name, **dv.describe()}
