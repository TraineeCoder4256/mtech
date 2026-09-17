#!/usr/bin/env python3
"""Step 2: train the boundary model.  Run: python train.py

Conditional flow matching -- the objective is in model.py, the encodings
and the leak-free split are in data.py.

Writes models/model.pt, models/norm.json, models/loss.png, models/log.txt.
"""
import pathlib
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset, save_norm
from model import VelocityField, cfm_loss, EMA

# ---- settings ----------------------------------------------------------
DATA = pathlib.Path("data/cells.npz")
OUT = pathlib.Path("models")
STEPS, BATCH, LR = 20000, 4096, 2e-3
WIDTH, DEPTH = 256, 5
WARMUP, EMA_DECAY, VAL_EVERY, VAL_FRAC = 500, 0.999, 500, 0.05
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    print(f"device {DEVICE}")

    ds = load_dataset(DATA, VAL_FRAC, SEED)
    i = ds["info"]
    print(f"data   {i['rows']:,} rows -> {i['collided']:,} collided, "
          f"{i['conditions']:,} conditions")
    print(f"split  {i['train']:,} train / {i['val']:,} val "
          f"({i['val_conditions']:,} held-out conditions)")

    y, c, yv, cv = (torch.from_numpy(ds[k]).to(DEVICE) for k in
                    ("y_train", "c_train", "y_val", "c_val"))
    net = VelocityField(y.shape[1], c.shape[1], WIDTH, DEPTH).to(DEVICE)
    print(f"model  width {WIDTH} depth {DEPTH}, {net.n_params():,} params")

    ema = EMA(net, EMA_DECAY)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(       # warmup, then cosine
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        0.5 * (1 + np.cos(np.pi * min(1.0, s / STEPS))))

    per_epoch = max(1, y.shape[0] // BATCH)
    print(f"train  {STEPS:,} steps at batch {BATCH:,} "
          f"= {STEPS / per_epoch:.0f} nominal epochs\n")

    gen = torch.Generator().manual_seed(SEED)
    hist, smooth, t0 = [], None, time.time()
    log = open(OUT / "log.txt", "w")

    for step in range(1, STEPS + 1):
        idx = torch.randint(0, y.shape[0], (BATCH,), generator=gen).to(DEVICE)
        loss = cfm_loss(net, y[idx], c[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(net)
        smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()

        if step % VAL_EVERY == 0 or step == STEPS:
            with torch.no_grad():
                v = torch.randint(0, yv.shape[0], (16384,), generator=gen).to(DEVICE)
                vloss = cfm_loss(ema.shadow, yv[v], cv[v]).item()
            line = (f"step {step:6d} (ep {step / per_epoch:5.1f})  "
                    f"train {smooth:.4f}  val(EMA) {vloss:.4f}  "
                    f"{step * BATCH / (time.time() - t0):,.0f} samp/s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            hist.append((step, smooth, vloss))
    log.close()

    config = {"width": WIDTH, "depth": DEPTH, "x_dim": int(y.shape[1]),
              "c_dim": int(c.shape[1]), "steps": STEPS, "batch": BATCH,
              "lr": LR, "seed": SEED}
    torch.save({"ema": {k: v.cpu() for k, v in ema.shadow.state_dict().items()},
                "config": config}, OUT / "model.pt")
    save_norm(OUT / "norm.json", ds["ynorm"], ds["cnorm"], config)

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(h[:, 0], h[:, 1], label="train")
    ax.plot(h[:, 0], h[:, 2], label="validation (EMA)")
    ax.set_xlabel("step"); ax.set_ylabel("CFM loss"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "loss.png", dpi=150)
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
