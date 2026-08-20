#!/usr/bin/env python3
"""Train the boundary GMC model (conditional flow matching).

Reads a single-cell dataset produced by make_lattice_dataset.py, applies the
preprocessing of docs/boundary_model_training.md (uncollided branch dropped,
circular p encoding, log path length, condition-wise split), and trains the
velocity field with the CFM objective.

--s-param selects how the path length is encoded (see gmc/data.py):
  detour  u = log(s / s_min(p))   default; the straight-line bound s >= s_min
                                  is then structural and the sampler never
                                  has to clamp
  logW    u = log(s / W~)         the original v1 encoding, kept so the
                                  earlier checkpoint stays reproducible

Outputs (to --out, default models/boundary_v1/):
  model.pt           raw + EMA weights and config
  normalizers.json   encoding statistics (needed to sample)
  loss_curve.png     train/val CFM loss
  train_log.txt      per-interval losses

Usage:
  python scripts/train_boundary_model.py [--data data/lattice_singlecell_coverage.npz]
      [--steps 20000] [--batch 4096] [--width 256] [--depth 5] [--lr 2e-3]
      [--out models/boundary_v1] [--seed 0]
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmc import VelocityField, cfm_loss, EMA  # noqa: E402
from gmc.data import load_boundary_dataset, save_normalizers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lattice_singlecell_coverage.npz")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--s-param", default="detour", choices=["detour", "logW"],
                    help="path-length encoding for the 6th target component")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/boundary_v1")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | mps; default: cuda if available")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    ds = load_boundary_dataset(ROOT / args.data, val_frac=args.val_frac,
                               seed=args.seed, s_param=args.s_param)
    info = ds["info"]
    print(f"dataset: {info['n_total']} rows -> {info['n_collided']} collided "
          f"(k>0), {info['n_conditions']} entry conditions")
    print(f"split:   {info['n_train']} train / {info['n_val']} val rows "
          f"({info['n_val_conditions']} val conditions, condition-wise)")
    print(f"s_param: {args.s_param}")

    y_tr = torch.from_numpy(ds["y_train"]).to(dev)
    c_tr = torch.from_numpy(ds["c_train"]).to(dev)
    y_va = torch.from_numpy(ds["y_val"]).to(dev)
    c_va = torch.from_numpy(ds["c_val"]).to(dev)

    model = VelocityField(x_dim=y_tr.shape[1], c_dim=c_tr.shape[1],
                          width=args.width, depth=args.depth).to(dev)
    print(f"model:   width={args.width} depth={args.depth} "
          f"({model.n_params():,} params) on {dev}")
    ema = EMA(model, decay=args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        0.5 * (1.0 + np.cos(np.pi * min(1.0, s / args.steps))))

    n = y_tr.shape[0]
    steps_per_epoch = max(1, n // args.batch)
    print(f"epochs:  {args.steps / steps_per_epoch:.1f} equivalent "
          f"({steps_per_epoch} steps/epoch at batch {args.batch}; batches are "
          f"sampled with replacement, so epochs are nominal)")
    g = torch.Generator().manual_seed(args.seed)
    log_path = out / "train_log.txt"
    hist = []
    t0 = time.time()
    run_loss = None

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
                vidx = torch.randint(0, y_va.shape[0], (16384,),
                                     generator=g).to(dev)
                vloss = cfm_loss(ema.shadow, y_va[vidx], c_va[vidx]).item()
            rate = step * args.batch / (time.time() - t0)
            line = (f"step {step:6d} (ep {step / steps_per_epoch:5.1f})  "
                    f"train {run_loss:.4f}  val(EMA) {vloss:.4f}  "
                    f"{rate:,.0f} samp/s")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            hist.append((step, run_loss, vloss))

    # ---- save ----------------------------------------------------------
    config = {"width": args.width, "depth": args.depth,
              "x_dim": int(y_tr.shape[1]), "c_dim": int(c_tr.shape[1]),
              "steps": args.steps, "batch": args.batch, "lr": args.lr,
              "data": str(args.data), "seed": args.seed,
              "s_param": args.s_param}
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.cpu() for k, v in ema.shadow.state_dict().items()},
                "config": config}, out / "model.pt")
    save_normalizers(out / "normalizers.json", ds["ynorm"], ds["cnorm"],
                     config)
    print(f"saved model + normalizers to {out}")

    # ---- loss curve ----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(h[:, 0], h[:, 1], label="train (running)")
    ax.plot(h[:, 0], h[:, 2], label="val (EMA)")
    ax.set_xlabel("step")
    ax.set_ylabel("CFM loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=150)
    print(f"wrote {out / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
