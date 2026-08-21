#!/usr/bin/env python3
"""Train the boundary model.  Run: python scripts/train.py

Conditional flow matching -- see gmc/model.py for the objective.  Two
preprocessing rules do real work, both in gmc/data.py: uncollided particles
are dropped (their exit is a Dirac delta a smooth flow cannot represent, and
they are handled analytically at sampling time), and the train/validation
split is by entry condition rather than by row, so validation cannot be
satisfied by memorising a sibling sample.

Writes model.pt, normalizers.json, loss_curve.png and train_log.txt.
"""
import pathlib
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmc import VelocityField, cfm_loss, EMA          # noqa: E402
from gmc import load_dataset, save_normalizers        # noqa: E402

# ---- settings ----------------------------------------------------------
DATA = ROOT / "data" / "singlecell.npz"
OUT = ROOT / "models" / "boundary"
STEPS, BATCH, LR = 20000, 4096, 2e-3
WIDTH, DEPTH = 256, 5
WARMUP, EMA_DECAY, VAL_EVERY, VAL_FRAC = 500, 0.999, 500, 0.05
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device {DEVICE}")

    ds = load_dataset(DATA, val_frac=VAL_FRAC, seed=SEED)
    i = ds["info"]
    print(f"data   {i['n_total']:,} rows -> {i['n_collided']:,} collided, "
          f"{i['n_conditions']:,} conditions")
    print(f"split  {i['n_train']:,} train / {i['n_val']:,} val "
          f"({i['n_val_conditions']:,} held-out conditions)")

    y_tr, c_tr, y_va, c_va = (torch.from_numpy(ds[k]).to(DEVICE) for k in
                              ("y_train", "c_train", "y_val", "c_val"))
    model = VelocityField(y_tr.shape[1], c_tr.shape[1], WIDTH, DEPTH).to(DEVICE)
    print(f"model  width {WIDTH} depth {DEPTH}, {model.n_params():,} params")

    ema = EMA(model, EMA_DECAY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    # linear warmup, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) *
        0.5 * (1.0 + np.cos(np.pi * min(1.0, s / STEPS))))

    n = y_tr.shape[0]
    per_epoch = max(1, n // BATCH)
    print(f"train  {STEPS:,} steps at batch {BATCH:,} "
          f"= {STEPS / per_epoch:.0f} nominal epochs\n")

    gen = torch.Generator().manual_seed(SEED)
    hist, smooth, t0 = [], None, time.time()
    log = open(OUT / "train_log.txt", "w")

    for step in range(1, STEPS + 1):
        idx = torch.randint(0, n, (BATCH,), generator=gen).to(DEVICE)
        loss = cfm_loss(model, y_tr[idx], c_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()

        if step % VAL_EVERY == 0 or step == STEPS:
            with torch.no_grad():
                v = torch.randint(0, y_va.shape[0], (16384,), generator=gen).to(DEVICE)
                vloss = cfm_loss(ema.shadow, y_va[v], c_va[v]).item()
            line = (f"step {step:6d} (ep {step / per_epoch:5.1f})  "
                    f"train {smooth:.4f}  val(EMA) {vloss:.4f}  "
                    f"{step * BATCH / (time.time() - t0):,.0f} samp/s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            hist.append((step, smooth, vloss))
    log.close()

    config = {"width": WIDTH, "depth": DEPTH, "x_dim": int(y_tr.shape[1]),
              "c_dim": int(c_tr.shape[1]), "steps": STEPS, "batch": BATCH,
              "lr": LR, "data": str(DATA), "seed": SEED}
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.cpu() for k, v in ema.shadow.state_dict().items()},
                "config": config}, OUT / "model.pt")
    save_normalizers(OUT / "normalizers.json", ds["ynorm"], ds["cnorm"], config)

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(h[:, 0], h[:, 1], label="train (running mean)")
    ax.plot(h[:, 0], h[:, 2], label="validation (EMA weights)")
    ax.set_xlabel("step"); ax.set_ylabel("CFM loss"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png", dpi=150)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
