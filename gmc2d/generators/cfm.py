"""Conditional flow matching -- the model this project started with.

This file is the old train.py loop and the old sampler.integrate(), moved
behind the Generator interface without changing a single operation.  The
check that nothing changed is check_refactor.py: given the same data and
seed, the weights this trains and the samples it draws are bit-identical to
the code at commit 81d8340.

How it draws: start from Gaussian noise z and integrate dx/dt = v(x, t, c)
from t = 0 to t = 1 with a FIXED number of steps.  Fixed matters -- a fixed
step count is a fixed, predictable cost, which is the whole selling point
of the method.  An adaptive solver would bring back the thickness-dependent
cost that the method exists to remove.  The objective and the network are
in model.py.

Cost: steps x NFE_PER_STEP[solver] network evaluations per sample; the
shipped setting, heun with 5 steps, is 10.
"""
import json
import pathlib
import time

import numpy as np
import torch

from model import VelocityField, cfm_loss, EMA
from generators.base import Generator

NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}

# ---- training settings (were the constants at the top of train.py) -----
STEPS, BATCH, LR = 20000, 4096, 2e-3
WIDTH, DEPTH = 256, 5
WARMUP, EMA_DECAY, VAL_EVERY = 500, 0.999, 500
SEED = 0


@torch.no_grad()
def integrate(model, x, c, steps, solver):
    """Fixed-step flow ODE.  Cost is steps * NFE_PER_STEP[solver]."""
    dt = 1.0 / steps
    # t only takes values on a half-step grid, so build them once
    ts = [torch.full((x.shape[0], 1), i / (2.0 * steps), device=x.device)
          for i in range(2 * steps + 1)]
    for i in range(steps):
        if solver == "euler":
            x = x + dt * model(x, ts[2 * i], c)
        elif solver == "heun":
            v0 = model(x, ts[2 * i], c)
            v1 = model(x + dt * v0, ts[2 * i + 2], c)
            x = x + 0.5 * dt * (v0 + v1)
        elif solver == "rk4":
            k1 = model(x, ts[2 * i], c)
            k2 = model(x + 0.5 * dt * k1, ts[2 * i + 1], c)
            k3 = model(x + 0.5 * dt * k2, ts[2 * i + 1], c)
            k4 = model(x + dt * k3, ts[2 * i + 2], c)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown solver {solver!r}")
    return x


class CFM(Generator):
    name = "cfm"

    def __init__(self, net=None, steps=5, solver="heun", device="cpu"):
        self.net = None if net is None else net.to(device).eval()
        self.steps, self.solver, self.device = steps, solver, device
        self.nfe = steps * NFE_PER_STEP[solver]
        self.config = {}

    # ------------------------------------------------------------ training
    def fit(self, data, out_dir, time_cap=float("inf")):
        """The train.py loop, operation for operation, so the RNG streams
        line up: seed, build the net, build the EMA, then one generator for
        the batch indices.  The only addition is the time cap, which reads
        the clock and draws no random numbers."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(SEED)
        dev = self.device

        y, c, yv, cv = (torch.from_numpy(data[k]).to(dev) for k in
                        ("y_train", "c_train", "y_val", "c_val"))
        net = VelocityField(y.shape[1], c.shape[1], WIDTH, DEPTH).to(dev)
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
        log = open(out_dir / "log.txt", "w")
        done = 0

        for step in range(1, STEPS + 1):
            idx = torch.randint(0, y.shape[0], (BATCH,), generator=gen).to(dev)
            loss = cfm_loss(net, y[idx], c[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(net)
            smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
            done = step
            capped = time.time() - t0 > time_cap

            if step % VAL_EVERY == 0 or step == STEPS or capped:
                with torch.no_grad():
                    v = torch.randint(0, yv.shape[0], (16384,), generator=gen).to(dev)
                    vloss = cfm_loss(ema.shadow, yv[v], cv[v]).item()
                line = (f"step {step:6d} (ep {step / per_epoch:5.1f})  "
                        f"train {smooth:.4f}  val(EMA) {vloss:.4f}  "
                        f"{step * BATCH / (time.time() - t0):,.0f} samp/s")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                hist.append((step, smooth, vloss))
            if capped:
                print(f"\ntime cap of {time_cap:.0f} s reached at step {step}")
                break
        log.close()

        self.net = ema.shadow.eval()
        self.config = {"width": WIDTH, "depth": DEPTH, "x_dim": int(y.shape[1]),
                       "c_dim": int(c.shape[1]), "steps": STEPS, "batch": BATCH,
                       "lr": LR, "seed": SEED}

        h = np.array(hist)
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.plot(h[:, 0], h[:, 1], label="train")
        ax.plot(h[:, 0], h[:, 2], label="validation (EMA)")
        ax.set_xlabel("step"); ax.set_ylabel("CFM loss"); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "loss.png", dpi=150)
        plt.close(fig)
        return {"steps_done": done, "train_seconds": time.time() - t0,
                "final_train_loss": hist[-1][1], "final_val_loss": hist[-1][2]}

    # ------------------------------------------------------------- drawing
    def sample(self, c, generator, cond=None):
        z = torch.randn(c.shape[0], self.net.x_dim, generator=generator).to(self.device)
        return integrate(self.net, z, c.to(self.device), self.steps,
                         self.solver).cpu()

    # ---------------------------------------------------------- checkpoints
    def save(self, out_dir):
        """model.pt in exactly the format train.py has always written, so
        old checkpoints load and new ones load in old code."""
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"ema": {k: v.cpu() for k, v in self.net.state_dict().items()},
                    "config": self.config}, out_dir / "model.pt")

    @classmethod
    def load(cls, out_dir, steps=5, solver="heun", device="cpu"):
        st = torch.load(pathlib.Path(out_dir) / "model.pt", map_location="cpu",
                        weights_only=True)
        c = st["config"]
        net = VelocityField(c["x_dim"], c["c_dim"], c["width"], c["depth"])
        net.load_state_dict(st["ema"])
        g = cls(net, steps, solver, device)
        g.config = c
        return g

    def describe(self):
        d = {"name": self.name, "nfe": self.nfe, "steps": self.steps,
             "solver": self.solver, **self.config}
        if self.net is not None:
            d["params"] = int(sum(p.numel() for p in self.net.parameters()))
        return d
