#!/usr/bin/env python3
"""Train the boundary model with conditional flow matching.

The objective in one paragraph.  Pick a training exit state y and a
Gaussian noise sample z, pick a time t in [0,1], and place a point on the
straight line between them, x_t = (1-t) z + t y.  Ask the network what
velocity would carry a particle along that line, and the answer is just
y - z.  Regress on that.  At sampling time, integrate the learned velocity
field from t = 0 to t = 1 starting from noise, and you land on a draw from
the conditional distribution.  gmc/cfm.py is nine lines; that is all of it.

What comes out (in --out, default models/boundary/):
    model.pt         raw + EMA weights, and the config needed to rebuild
    normalizers.json encoding statistics, needed to sample
    loss_curve.png   train and validation loss
    train_log.txt    the same numbers as text

Usage:
  python scripts/train.py                                # sensible defaults
  python scripts/train.py --device cuda --batch 16384 --lr 3e-3
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmc import VelocityField, cfm_loss, EMA            # noqa: E402
from gmc.data import load_dataset, save_normalizers     # noqa: E402
from gmc.device import pick_device, describe            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/singlecell.npz")
    ap.add_argument("--out", default="models/boundary")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(describe(dev))
    torch.manual_seed(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(ROOT / args.data, val_frac=args.val_frac, seed=args.seed)
    i = ds["info"]
    print(f"data:    {i['n_total']:,} rows -> {i['n_collided']:,} collided, "
          f"{i['n_conditions']:,} entry conditions")
    print(f"split:   {i['n_train']:,} train / {i['n_val']:,} val "
          f"({i['n_val_conditions']:,} held-out conditions -- split is by "
          f"CONDITION, not by row)")

    y_tr = torch.from_numpy(ds["y_train"]).to(dev)
    c_tr = torch.from_numpy(ds["c_train"]).to(dev)
    y_va = torch.from_numpy(ds["y_val"]).to(dev)
    c_va = torch.from_numpy(ds["c_val"]).to(dev)

    model = VelocityField(x_dim=y_tr.shape[1], c_dim=c_tr.shape[1],
                          width=args.width, depth=args.depth).to(dev)
    print(f"model:   width {args.width} depth {args.depth}, "
          f"{model.n_params():,} params")
    ema = EMA(model, decay=args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        0.5 * (1.0 + np.cos(np.pi * min(1.0, s / args.steps))))

    n = y_tr.shape[0]
    per_epoch = max(1, n // args.batch)
    print(f"train:   {args.steps:,} steps at batch {args.batch:,} "
          f"= {args.steps / per_epoch:.0f} nominal epochs\n")

    g = torch.Generator().manual_seed(args.seed)
    hist, run_loss, t0 = [], None, time.time()
    log = open(out / "train_log.txt", "w")

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), generator=g).to(dev)
        loss = cfm_loss(model, y_tr[idx], c_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        run_loss = loss.item() if run_loss is None else \
            0.98 * run_loss + 0.02 * loss.item()

        if step % args.val_every == 0 or step == args.steps:
            with torch.no_grad():
                v = torch.randint(0, y_va.shape[0], (16384,),
                                  generator=g).to(dev)
                vloss = cfm_loss(ema.shadow, y_va[v], c_va[v]).item()
            line = (f"step {step:6d} (ep {step / per_epoch:5.1f})  "
                    f"train {run_loss:.4f}  val(EMA) {vloss:.4f}  "
                    f"{step * args.batch / (time.time() - t0):,.0f} samp/s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            hist.append((step, run_loss, vloss))
    log.close()

    config = {"width": args.width, "depth": args.depth,
              "x_dim": int(y_tr.shape[1]), "c_dim": int(c_tr.shape[1]),
              "steps": args.steps, "batch": args.batch, "lr": args.lr,
              "data": str(args.data), "seed": args.seed}
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.cpu() for k, v in ema.shadow.state_dict().items()},
                "config": config}, out / "model.pt")
    save_normalizers(out / "normalizers.json", ds["ynorm"], ds["cnorm"], config)
    print(f"\nsaved to {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(h[:, 0], h[:, 1], label="train (running mean)")
    ax.plot(h[:, 0], h[:, 2], label="validation (EMA weights)")
    ax.set_xlabel("step")
    ax.set_ylabel("CFM loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    print(f"wrote {out / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
