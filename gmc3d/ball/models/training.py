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
a GAN family overrides Family.fit and copies these conventions by hand.
"""

import copy
import time

import numpy as np
import torch

VAL_ROWS = 65536


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


def fit_loop(build, loss_fn, data, *, steps, batch, lr, warmup, ema_decay,
             val_every, seed, time_cap=float("inf"), log=print,
             loss_name="loss"):
    """Train build() on data with loss_fn.  Returns (EMA network, facts).

    build()             -> a fresh torch.nn.Module, called right after
                           torch.manual_seed(seed), so the seed fixes its
                           starting weights
    loss_fn(net, y, c)  -> a scalar loss on a batch of standardised targets
                           y and conditions c.  It may draw its own random
                           numbers (noise, a time) from torch's global
                           generator, which the seed also fixes
    data                the dict encoding.prepare() returns
    """
    torch.manual_seed(seed)
    y, c = torch.from_numpy(data["y_train"]), torch.from_numpy(data["c_train"])
    yv, cv = torch.from_numpy(data["y_val"]), torch.from_numpy(data["c_val"])
    net = build()
    n_params = sum(p.numel() for p in net.parameters())
    ema = EMA(net, ema_decay)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) *
        0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
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
