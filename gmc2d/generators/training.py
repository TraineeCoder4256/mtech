"""The training loop every regression-style family shares.

Flow matching, diffusion, a normalizing flow and a VAE are all trained the
same way from the outside: draw a random batch, compute ONE scalar loss,
take a gradient step, repeat.  They differ only in the network and in that
loss (a velocity error, a noise error, a negative log-likelihood, an ELBO).
So a new family writes two things -- a function that builds its network and
a function that returns its loss on a batch -- and hands them to fit_loop().
Everything else lives here, in one copy, so that no family can win the
comparison by being given a better optimiser, a longer warm-up or a kinder
validation check than another:

  optimiser   AdamW, weight decay 1e-5, gradient norm clipped to 1
  schedule    linear warm-up, then cosine decay to zero at the last step
  EMA         a moving average of the weights; it is what gets sampled and
              validated, because it is smoother than the raw weights
  validation  every val_every steps on 16,384 held-out rows (held out by
              ENTRY CONDITION in data.py, so it measures generalisation to
              unseen cells, not memory)
  time cap    stop early when the wall clock passes time_cap; the step
              reached is recorded, and run.py's report says so
  records     log.txt, loss.png, and the facts dict saved as train_info.json

This is the loop that was inside generators/cfm.py, moved here operation for
operation: the same seed, then the same network build, then the same EMA and
optimiser, then one torch.Generator for the batch indices.  Random numbers
are drawn in exactly the old order, so cfm trains bit-identically
(check_refactor.py tests it).

A GAN does NOT fit this shape -- it trains two networks against each other,
with two optimisers and two losses -- so a GAN family keeps its own loop and
should copy the conventions above by hand.
"""
import pathlib
import time

import numpy as np
import torch

from model import EMA


def fit_loop(build, loss_fn, data, out_dir, time_cap, *, steps, batch, lr,
             warmup, ema_decay, val_every, seed, device, loss_name="loss"):
    """Train build() on data with loss_fn; return (EMA network, facts).

    build()                 -> a fresh torch.nn.Module.  Called right after
                               torch.manual_seed(seed), so its initial weights
                               are fixed by the seed.
    loss_fn(net, y, c)      -> scalar loss on a batch of targets y (n, 6) and
                               conditions c (n, 5).  It may draw its own
                               random numbers (noise, time) from torch's
                               global generator, as cfm_loss does.
    data                    the dict data.load_dataset() returns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    y, c, yv, cv = (torch.from_numpy(data[k]).to(device) for k in
                    ("y_train", "c_train", "y_val", "c_val"))
    net = build().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"model  {n_params:,} params")

    ema = EMA(net, ema_decay)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(       # warmup, then cosine
        opt, lambda s: min(1.0, (s + 1) / warmup) *
        0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))

    per_epoch = max(1, y.shape[0] // batch)
    print(f"train  {steps:,} steps at batch {batch:,} "
          f"= {steps / per_epoch:.0f} nominal epochs\n")

    gen = torch.Generator().manual_seed(seed)
    hist, smooth, t0 = [], None, time.time()
    log = open(out_dir / "log.txt", "w")
    done = 0

    for step in range(1, steps + 1):
        idx = torch.randint(0, y.shape[0], (batch,), generator=gen).to(device)
        loss = loss_fn(net, y[idx], c[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(net)
        smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
        done = step
        capped = time.time() - t0 > time_cap

        if step % val_every == 0 or step == steps or capped:
            with torch.no_grad():
                v = torch.randint(0, yv.shape[0], (16384,), generator=gen).to(device)
                vloss = loss_fn(ema.shadow, yv[v], cv[v]).item()
            line = (f"step {step:6d} (ep {step / per_epoch:5.1f})  "
                    f"train {smooth:.4f}  val(EMA) {vloss:.4f}  "
                    f"{step * batch / (time.time() - t0):,.0f} samp/s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            hist.append((step, smooth, vloss))
        if capped:
            print(f"\ntime cap of {time_cap:.0f} s reached at step {step}")
            break
    log.close()

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(h[:, 0], h[:, 1], label="train")
    ax.plot(h[:, 0], h[:, 2], label="validation (EMA)")
    ax.set_xlabel("step"); ax.set_ylabel(loss_name); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)
    return ema.shadow.eval(), {
        "steps_done": done, "train_seconds": time.time() - t0,
        "final_train_loss": hist[-1][1], "final_val_loss": hist[-1][2],
        "seed": seed, "params": int(n_params)}
