"""Family 2: conditional flow matching -- the model this project started with.

Ported from gmc2d (generators/cfm.py and model.py) and cut down to the
ball problem: 1 condition (log R), 2 targets.  Nothing about the method
changed; only the sizes.

HOW IT DRAWS.  Start from Gaussian noise z and move it along a velocity
field v(x, t, c) from time t = 0 to t = 1:  dx/dt = v.  Where it lands is
the sample.  The network IS that velocity field.  The integration uses a
FIXED number of steps (Heun's method, 5 steps = 10 network passes, as in
gmc2d), because a fixed step count is a fixed, predictable cost; an
adaptive solver would make the cost depend on the ball.

HOW IT TRAINS.  Draw noise z and a real walk y, pick a time t, put a point
on the straight line between them, x_t = (1 - t) z + t y.  Along that line
the velocity is y - z, so regress the network on it (mean squared error).
Each x_t lies on many such lines, so the network learns the AVERAGE
velocity there -- and that averaged field is exactly the one whose flow
carries noise onto the data.  An analogy: watch many people walk from the
station to their homes; the average direction of travel at each street
corner, followed from the station, spreads people out just as the homes
are spread.

Three choices carried over from gmc2d, none obvious:
  * Fourier features for t: a raw scalar through a linear layer can only
    vary slowly with t, but the field changes character along the path.
  * FiLM conditioning rather than concatenating c: concatenated
    conditioning fades with depth, and R is the whole point here.
  * Zero-initialised output layer: the flow starts as "do not move".

COST.  nfe = 10 network passes per draw against the flow's 1, so expect
about ten times the flow's cost per draw at the same size.  It is here for
the comparison (accuracy per weight, per training step); it will only
reach transport if it passes Innovation's gate, cost per draw included.

SIZE.  Width 64, 7 FiLM blocks, embedding 32, 32 Fourier features:
149,090 weights, +1.7% on the flow's 146,554 (fairness budget +-20%).
Tuning trials: none yet (budget fairness.TUNING_RUNS).
"""

import math

import torch
import torch.nn as nn

from .encoding import C_DIM, X_DIM
from .base import Family
from .nflow import FiLMBlock

WIDTH, DEPTH, EMB, T_DIM = 64, 7, 32, 32
ODE_STEPS, SOLVER = 5, "heun"
NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


class FourierTime(nn.Module):
    """t -> sines and cosines of t at fixed random frequencies."""

    def __init__(self, dim=T_DIM, scale=16.0):
        super().__init__()
        self.register_buffer("freq", torch.randn(dim // 2) * scale)

    def forward(self, t):
        a = 2.0 * math.pi * t * self.freq[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class VelocityField(nn.Module):
    """v(x, t, c): (n, X_DIM) + (n, 1) + (n, C_DIM) -> (n, X_DIM)."""

    def __init__(self, width=WIDTH, depth=DEPTH, emb=EMB, t_dim=T_DIM,
                 x_dim=X_DIM, c_dim=C_DIM):
        super().__init__()
        self.x_dim = x_dim
        self.time = FourierTime(t_dim)
        self.embed = nn.Sequential(nn.Linear(t_dim + c_dim, emb), nn.SiLU(),
                                   nn.Linear(emb, emb), nn.SiLU())
        self.proj_in = nn.Linear(x_dim, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, emb)
                                    for _ in range(depth))
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, x_dim)
        nn.init.zeros_(self.proj_out.weight)      # start at zero velocity
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, t, c):
        emb = self.embed(torch.cat([self.time(t), c], dim=1))
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.proj_out(self.norm_out(h))


@torch.no_grad()
def integrate(net, x, c, steps=ODE_STEPS, solver=SOLVER):
    """Fixed-step flow ODE from t = 0 to 1.  Cost: steps * NFE_PER_STEP."""
    dt = 1.0 / steps
    ts = [torch.full((x.shape[0], 1), i / (2.0 * steps), device=x.device)
          for i in range(2 * steps + 1)]
    for i in range(steps):
        if solver == "euler":
            x = x + dt * net(x, ts[2 * i], c)
        elif solver == "heun":
            v0 = net(x, ts[2 * i], c)
            v1 = net(x + dt * v0, ts[2 * i + 2], c)
            x = x + 0.5 * dt * (v0 + v1)
        elif solver == "rk4":
            k1 = net(x, ts[2 * i], c)
            k2 = net(x + 0.5 * dt * k1, ts[2 * i + 1], c)
            k3 = net(x + 0.5 * dt * k2, ts[2 * i + 1], c)
            k4 = net(x + dt * k3, ts[2 * i + 2], c)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown solver {solver!r}")
    return x


class CFM(Family):
    name = "cfm"
    nfe = ODE_STEPS * NFE_PER_STEP[SOLVER]
    loss_name = "CFM loss"
    CONFIG = {"width": WIDTH, "depth": DEPTH, "emb": EMB, "t_dim": T_DIM,
              "ode_steps": ODE_STEPS, "solver": SOLVER}

    @classmethod
    def build(cls, config):
        return VelocityField(config["width"], config["depth"], config["emb"],
                             config["t_dim"], config.get("x_dim", X_DIM),
                             config.get("c_dim", C_DIM))

    @staticmethod
    def loss(net, y, c):
        t = torch.rand(y.shape[0], 1, device=y.device)
        z = torch.randn_like(y)
        return torch.mean((net((1 - t) * z + t * y, t, c) - (y - z)) ** 2)

    @classmethod
    def draw(cls, net, c, generator):
        z = torch.randn(c.shape[0], net.x_dim, generator=generator,
                        device=c.device)
        return integrate(net, z, c, ODE_STEPS, SOLVER)
