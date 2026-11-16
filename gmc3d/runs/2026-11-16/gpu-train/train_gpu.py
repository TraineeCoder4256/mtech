"""Train the v2 models (interface v2, anisotropic balls) on a GPU.

    python runs/2026-11-16/gpu-train/train_gpu.py --seeds 0 1
    python runs/2026-11-16/gpu-train/train_gpu.py --seeds 0 --families cvae
    python runs/2026-11-16/gpu-train/train_gpu.py --device cpu --steps 30   # check

ball/models/training.fit_loop trains on the CPU only.  This script runs a
copy of it that puts the network and the training rows on --device and
otherwise follows it step for step: same seed, build, EMA, AdamW, warm-up
and cosine, batch indices from a CPU generator, validation on the EMA
network.  The trained network comes back to the CPU before it is saved,
so model.pt loads anywhere.  With --device cpu it gives the same losses as
fit_loop (checked when this script was written).  On a GPU the numbers
differ slightly (other kernels), so the weights are new models, not
bit-for-bit copies of m2.1.

Data: pipeline.aniso_data() as for m2.1 (hash e7a34544f0048944, about
9.3 GB of RAM to build, CPU).  Weights: data/models/<family>_v2_seed<k>/.
Logs: next to this script.  cGAN has its own loop and is not covered.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from ball import models                                     # noqa: E402
from ball.models import base, pipeline, training            # noqa: E402

DEVICE = "cuda"


def fit_loop_device(build, loss_fn, data, *, steps, batch, lr, warmup,
                    ema_decay, val_every, seed, time_cap=float("inf"),
                    log=print, loss_name="loss"):
    """training.fit_loop with the network and the rows on DEVICE."""
    dev = torch.device(DEVICE)
    torch.manual_seed(seed)
    y = torch.from_numpy(data["y_train"]).to(dev)
    c = torch.from_numpy(data["c_train"]).to(dev)
    yv = torch.from_numpy(data["y_val"]).to(dev)
    cv = torch.from_numpy(data["c_val"]).to(dev)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = build().to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    ema = training.EMA(net, ema_decay)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) *
        0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    gen = torch.Generator().manual_seed(seed)
    hist, smooth, t1, done = [], None, time.time(), 0
    for step in range(1, steps + 1):
        idx = torch.randint(0, y.shape[0], (batch,), generator=gen).to(dev)
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
                v = torch.randint(0, yv.shape[0], (training.VAL_ROWS,),
                                  generator=gen).to(dev)
                vloss = loss_fn(ema.shadow, yv[v], cv[v]).item()
            hist.append((step, smooth, vloss))
            log(f"step {step:6d}  train {loss_name} {smooth:.4f}  "
                f"val {loss_name} (EMA) {vloss:.4f}  "
                f"{time.time() - t1:.0f} s  [{DEVICE}]")
        if capped:
            log(f"time cap of {time_cap:.0f} s reached at step {step}")
            break
    return ema.shadow.eval().cpu(), {
        "steps": steps, "steps_done": done, "seed": seed,
        "params": int(n_params), "train_seconds": time.time() - t1,
        "history": hist, "loss_name": loss_name, "device": DEVICE}


def logger(path):
    f = open(path, "w")

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def main():
    global DEVICE
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--families", nargs="+", default=["nflow", "cfm", "cvae"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=None,
                   help="default: the shared budget (24,000)")
    a = p.parse_args()
    DEVICE = a.device
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("torch sees no GPU: install a CUDA build of PyTorch")
    if "cgan" in a.families:
        sys.exit("cgan trains in its own loop; this script does not cover it")
    base.fit_loop = fit_loop_device         # what Family.fit calls
    pipeline.aniso_data(logger(HERE / "data.log"))
    for s in a.seeds:
        for fam in a.families:
            models.train(fam, s, steps=a.steps, interface=2,
                         log=logger(HERE / f"train_{fam}_v2_seed{s}.log"))


if __name__ == "__main__":
    main()
