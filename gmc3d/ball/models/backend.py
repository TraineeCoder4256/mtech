"""Any trained family, behind the face core/transport.py already uses.

Used by: run.py (through models.load), ball/network.py (the flow's old
`Network` name is this class with the flow inside), compare.py.
Uses: encoding.py, store.py, the family's draw().

The transport loop asks every backend the same thing: "here are the radii
of the balls waiting; give me (mu, s) for each".  This class does the
part that is the same for every family -- encode R, call the family's draw,
decode back to (mu, s) -- so a family only ever writes its draw in the
encoded space.  It is ball/network.py's `Network` class (v1.6.1) with the
flow-specific line (the draw) handed to the family; the operations and the
random numbers are unchanged, so the flow's draws are bit-identical.

`device` is where the network runs: "cpu" (default) or "cuda".  The
transport stays on the CPU either way; each call sends the batch of radii
over and gets (mu, s) back.  The draw uses its own seeded torch generator
(the particles' streams are not needed), so a run is reproducible on a
given device (CPU and GPU draw different noise).
"""

import numpy as np
import torch

from . import encoding as E


class ModelBackend:

    def __init__(self, family, net, cnorm, ynorm, info, seed=12345,
                 device="cpu", name=None):
        self.family = family
        self.name = name or family.name
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.cnorm, self.ynorm, self.info = cnorm, ynorm, info
        self.r_min, self.r_max = info["r_min"], info["r_max"]
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.draws = 0

    @torch.no_grad()
    def sample(self, R, seeds=None, idx=None):
        """(mu, s) for each R."""
        R = np.asarray(R, np.float64)
        c = torch.from_numpy(self.cnorm(E.condition(R)).astype(
            np.float32)).to(self.device)
        y = self.family.draw(self.net, c, self.gen)
        y = y.cpu().numpy().astype(np.float64)
        self.draws += R.size
        return E.decode(R, self.ynorm.undo(y))

    def describe(self):
        return {"name": self.name, "family": self.family.name,
                "nfe": self.family.nfe,
                **{k: v for k, v in self.info.items() if k != "history"}}
