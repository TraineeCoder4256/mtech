"""Free: a "model" that costs nothing.  The COST goalpost.

sample() returns its input noise untouched -- no arithmetic at all, which is
what the zero velocity field (FreeField) in speed_profile.py did inside the
ODE.  Its exit states are physically meaningless, so its accuracy must never
be read.  Its wall time is the point: everything the transport loop costs
EXCEPT the model.  No model, however cheap, can make a solve faster than
this row, so it bounds every speed claim from below.

One caveat worth stating in a write-up: meaningless exits change how many
cell crossings a particle makes, so this is the floor for the loop as
driven by noise, not exactly the loop as driven by a good model.  The
crossing count is reported beside the time so the two can be compared.
"""
import torch

from generators.base import Generator


class Free(Generator):
    name = "free"
    nfe = 0
    trainable = False

    def __init__(self, ynorm=None):
        pass

    def sample(self, c, generator, cond=None):
        return torch.randn(c.shape[0], 6, generator=generator)
