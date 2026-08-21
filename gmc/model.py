"""The velocity field v(x, t, c) -- the only neural network in the project.

WHAT IT HAS TO DO
-----------------
Flow matching turns "sample from a complicated distribution" into "solve an
ODE".  The network's whole job is to answer one question:

    "I am at point x, at time t along the path from noise to data, and the
     cell I am describing has properties c.  Which way do I move next?"

It returns a velocity in the same 6-D space as the data.  Integrating that
velocity from t=0 (pure noise) to t=1 lands on a sample.  See gmc/cfm.py for
how it is trained and gmc/sampler.py for how it is integrated.

    input   x (n, 6)   where we are now, in the 6-D exit-state space
            t (n, 1)   how far along the noise -> data path, 0 to 1
            c (n, 5)   the cell + entry condition being asked about
    output    (n, 6)   the velocity to move along

THREE DESIGN CHOICES, AND WHY
-----------------------------
1. FOURIER FEATURES FOR TIME.  A raw scalar t entering a linear layer can
   only produce effects that vary smoothly and slowly with t.  But the
   velocity field genuinely changes character over the path -- early on it
   is pushing noise into roughly the right region, late on it is placing
   fine detail.  Expanding t into many sines and cosines of different
   frequencies gives the network a basis in which sharp changes with t are
   easy to express.

2. FiLM CONDITIONING, NOT CONCATENATION.  The obvious way to use c is to
   glue it onto the input and let the first layer sort it out.  The problem
   is that by the third or fourth layer the network has largely reshaped
   its representation and the conditioning has faded.  FiLM instead lets c
   produce a scale and a shift applied inside EVERY block, so the condition
   modulates the computation all the way down.  For us c is the whole point
   -- a cell of optical width 0.1 and one of width 20 have completely
   different exit distributions -- so it must not fade.

3. ZERO-INITIALISED OUTPUT LAYER.  At initialisation the network returns
   exactly zero, i.e. "do not move".  The ODE then starts as the identity
   map, which is a stable, sensible thing to be before any training has
   happened, and the loss descends from there instead of from whatever
   random field the initial weights happened to encode.

The network is small on purpose: the target is 6-D and the conditioning
5-D, so ~1.7 M parameters is ample, and staying small keeps it trainable
on a CPU and cheap to evaluate -- which matters, because inference cost is
the entire argument for the method.
"""

import math

import torch
import torch.nn as nn


class FourierTime(nn.Module):
    """Scalar flow time t -> a vector of sines and cosines of t.

    The frequencies are drawn once at construction and never trained; they
    are registered as a buffer so they travel with the checkpoint.  A
    trained frequency set is not needed -- a fixed random spread of scales
    already spans the range of behaviours in t.
    """

    def __init__(self, dim=64, scale=16.0):
        super().__init__()
        # `scale` sets the spread of frequencies: bigger means the features
        # can resolve faster variation in t.
        self.register_buffer("freq", torch.randn(dim // 2) * scale)

    def forward(self, t):                     # t: (n, 1) in [0, 1]
        angles = 2.0 * math.pi * t * self.freq[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class FiLMBlock(nn.Module):
    """One residual MLP block whose behaviour is modulated by the condition.

    FiLM = Feature-wise Linear Modulation: the conditioning embedding
    predicts a per-feature scale and shift, which are applied to the
    normalised activations before the block's own MLP runs.
    """

    def __init__(self, width, emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)      # widen
        self.fc2 = nn.Linear(width * 2, width)      # and back
        self.film = nn.Linear(emb_dim, width * 2)   # -> (scale, shift)

    def forward(self, h, emb):
        # one linear layer produces both halves at once, then split them
        scale, shift = self.film(emb).chunk(2, dim=1)
        # normalise first, then let the condition rescale and offset.
        # (1 + scale) so that an untrained film layer is the identity.
        x = self.norm(h) * (1.0 + scale) + shift
        x = self.fc2(torch.nn.functional.silu(self.fc1(x)))
        # residual: the block learns a correction, not a replacement, which
        # is what keeps a stack of them trainable
        return h + x


class VelocityField(nn.Module):
    """v(x, t, c): (n, x_dim) + (n, 1) + (n, c_dim) -> (n, x_dim)."""

    def __init__(self, x_dim=6, c_dim=5, width=256, depth=5,
                 emb_dim=128, t_dim=64):
        super().__init__()
        self.x_dim, self.c_dim = x_dim, c_dim

        # time -> Fourier features
        self.time = FourierTime(t_dim)
        # (time features, condition) -> one embedding vector that every
        # block will consult
        self.embed = nn.Sequential(
            nn.Linear(t_dim + c_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
        )

        # the trunk: lift the 6-D point to `width`, refine it `depth` times,
        # project back down to a 6-D velocity
        self.proj_in = nn.Linear(x_dim, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, emb_dim)
                                    for _ in range(depth))
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, x_dim)

        # start by predicting zero velocity (see design choice 3 above)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, t, c):
        # 1. fold time and condition into a single embedding
        emb = self.embed(torch.cat([self.time(t), c], dim=1))
        # 2. lift the current point into the trunk's width
        h = self.proj_in(x)
        # 3. refine it, with the condition steering every block
        for blk in self.blocks:
            h = blk(h, emb)
        # 4. read off the velocity
        return self.proj_out(self.norm_out(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
