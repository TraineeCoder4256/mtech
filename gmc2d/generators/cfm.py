"""Conditional flow matching -- the model this project started with.

This file is the old sampler.integrate() and the flow-matching half of the
old train.py, behind the Generator interface.  The other half of train.py,
the loop that every family shares, now lives in generators/training.py.
Neither move changed a single operation: check_refactor.py shows that, given
the same data and seed, the weights this trains and the samples it draws are
bit-identical to the code at commit 81d8340.

How it draws: start from Gaussian noise z and integrate dx/dt = v(x, t, c)
from t = 0 to t = 1 with a FIXED number of steps.  Fixed matters -- a fixed
step count is a fixed, predictable cost, which is the whole selling point
of the method.  An adaptive solver would bring back the thickness-dependent
cost that the method exists to remove.  The objective and the network are
in model.py.

Cost: steps x NFE_PER_STEP[solver] network evaluations per sample; the
shipped setting, heun with 5 steps, is 10.
"""
import pathlib

import torch

from model import VelocityField, cfm_loss
from generators.base import Generator
from generators.training import fit_loop

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

    def __init__(self, net=None, steps=5, solver="heun", device="cpu", seed=SEED):
        self.net = None if net is None else net.to(device).eval()
        self.steps, self.solver, self.device, self.seed = steps, solver, device, seed
        self.nfe = steps * NFE_PER_STEP[solver]
        self.config = {}

    # ------------------------------------------------------------ training
    def fit(self, data, out_dir, time_cap=float("inf")):
        """The shared loop (generators/training.py) with the flow-matching
        loss.  The settings are read from this module's constants at call
        time, so check_refactor.py can shorten a run by setting them."""
        x_dim, c_dim = data["y_train"].shape[1], data["c_train"].shape[1]
        print(f"cfm    width {WIDTH} depth {DEPTH}")
        self.net, facts = fit_loop(
            lambda: VelocityField(x_dim, c_dim, WIDTH, DEPTH), cfm_loss,
            data, out_dir, time_cap, steps=STEPS, batch=BATCH, lr=LR,
            warmup=WARMUP, ema_decay=EMA_DECAY, val_every=VAL_EVERY,
            seed=self.seed, device=self.device, loss_name="CFM loss")
        self.config = {"width": WIDTH, "depth": DEPTH, "x_dim": int(x_dim),
                       "c_dim": int(c_dim), "steps": STEPS, "batch": BATCH,
                       "lr": LR, "seed": self.seed}
        return facts

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
        d = {"name": self.name, "nfe": self.nfe, "ode_steps": self.steps,
             "solver": self.solver, "train_config": self.config}
        if self.net is not None:
            d["params"] = int(sum(p.numel() for p in self.net.parameters()))
        return d
