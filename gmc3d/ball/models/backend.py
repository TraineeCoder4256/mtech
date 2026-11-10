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


class AnisoBackend:
    """Interface v2 (m2.0): the backend face core/transport.py uses when a
    problem has anisotropic materials.

        aniso = True                  tells the driver to ask for both
        sample(R, seeds, idx)         isotropic balls: (mu, s), handed to
                                      the accepted v1 model unchanged
        sample_aniso(R, mats, ang, seeds, idx)
                                      anisotropic balls: (cos_n, mu, phi, s)
                                      from the v2 model

    The v2 model needs each ball's law as 12 Legendre moments.  The driver
    hands over the material index of each ball and the packed laws `ang`
    (core/scattering.pack); the moments of every row of `ang` are worked
    out once (core.scattering.moments, the same function the training
    pool used) and looked up per ball.  So isotropic accuracy never depends
    on the v2 model, which is what Innovation and Development agreed.
    """

    aniso = True

    def __init__(self, family, net, cnorm, ynorm, info, iso, seed=12345,
                 device="cpu", name=None):
        self.family, self.iso = family, iso
        self.name = name or f"{family.name}_v2"
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.cnorm, self.ynorm, self.info = cnorm, ynorm, info
        self.r_min, self.r_max = info["r_min"], info["r_max"]
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.draws = 0
        self._ang_key, self._g = None, None

    def sample(self, R, seeds=None, idx=None):
        return self.iso.sample(R, seeds, idx)

    def moments(self, ang):
        """(materials, 12) moments of every law in ang, cached per ang."""
        key = (ang.shape, hash(np.ascontiguousarray(ang).tobytes()))
        if key != self._ang_key:
            self._g = np.array([law_moments(row) for row in ang])
            self._ang_key = key
        return self._g

    @torch.no_grad()
    def draw(self, R, g):
        """(cos_n, mu, phi, s) for radii R (n,) and moments g (n, 12)."""
        R = np.asarray(R, np.float64)
        c = torch.from_numpy(self.cnorm(E.condition_aniso(R, g)).astype(
            np.float32)).to(self.device)
        y = self.family.draw(self.net, c, self.gen)
        y = y.cpu().numpy().astype(np.float64)
        self.draws += R.size
        return E.decode_aniso(R, self.ynorm.undo(y))

    def sample_aniso(self, R, mats, ang, seeds=None, idx=None):
        return self.draw(R, self.moments(ang)[np.asarray(mats)])

    def describe(self):
        return {"name": self.name, "family": self.family.name,
                "nfe": self.family.nfe, "interface": 2,
                "isotropic_balls": getattr(self.iso, "name", "?"),
                **{k: v for k, v in self.info.items() if k != "history"}}


def law_moments(row):
    """One packed law (a row of core/scattering.pack's `ang`) -> its
    moments g_1 .. g_12, as ball/laws.moments gives them for the law."""
    from core import scattering as SC
    from .. import interface as I
    kind, n = int(row[0]), int(row[1])
    if kind == SC.ISO:
        return np.zeros(I.N_MOMENTS)
    H = SC.HEAD
    if kind in (SC.TAB, SC.HIST):
        v = (row[H:H + n], row[H + n:H + 2 * n])
    else:
        v = (row[H:H + n], row[2], row[3])
    return SC.moments(kind, v, I.N_MOMENTS)[1:]
