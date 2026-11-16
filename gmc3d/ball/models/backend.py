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

`precision`, `compile`, `graphs` are the speed options of fast.Engine
(m2.4): fp16 mixed precision, torch.compile, CUDA graphs on padded batch
sizes.  With their defaults a backend draws exactly what it always drew.
"""

import numpy as np

from . import encoding as E
from .fast import Engine


class ModelBackend:

    def __init__(self, family, net, cnorm, ynorm, info, seed=12345,
                 device="cpu", name=None, precision="fp32", compile=False,
                 graphs=False):
        self.family = family
        self.name = name or family.name
        self.engine = Engine(family, net, E.C_DIM, E.X_DIM, device,
                             precision, compile, graphs, seed)
        self.device, self.net = self.engine.device, self.engine.net
        self.gen = self.engine.gen
        self.cnorm, self.ynorm, self.info = cnorm, ynorm, info
        self.r_min, self.r_max = info["r_min"], info["r_max"]
        self.draws = 0

    def sample(self, R, seeds=None, idx=None):
        """(mu, s) for each R."""
        return self.collect(self.submit(R, seeds, idx))

    def submit(self, R, seeds=None, idx=None):
        """Start sample(R) and return a ticket at once (the GPU works while
        the caller tracks other particles); collect(ticket) -> (mu, s), the
        same numbers sample() gives for the same calls in the same order.
        At most two tickets may be outstanding."""
        R = np.asarray(R, np.float64)
        t = self.engine.submit(self.cnorm(E.condition(R)).astype(np.float32))
        self.draws += R.size
        return R, t

    def collect(self, ticket):
        R, t = ticket
        return E.decode(R, self.ynorm.undo(self.engine.collect(t)))

    def warmup(self, sizes=(1024, 4096, 16384, 65536)):
        """Compile / record graphs before timing (fast.Engine.warmup)."""
        self.engine.warmup(sizes)

    def describe(self):
        return {"name": self.name, "family": self.family.name,
                "nfe": self.family.nfe, **self.engine.describe(),
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

    SINGLE-SCATTER EXITS ARE WALKED EXACTLY (m2.1).  A walk that scatters
    once stays in one plane, so it leaves with phi exactly pi: a point
    mass (up to half the walks at R = 1) that no smooth model can draw.
    So sample_aniso first walks every ball exactly up to its second
    collision (walk.first_scatter_many_aniso, Innovation's, with the
    particles' own random streams, like the exact walk).  The balls that
    exit there are done, exactly; for the others the walk's state is
    thrown away and the family draws from what it learned, p(exit | at
    least 2 scatters, R, law).  That mixture is exact apart from the
    family itself: the share that exits after one scatter is the walk's
    own, and the family is asked exactly what it was trained on.  An
    analogy: a shop assistant who hands over the common items from the
    counter and only sends to the warehouse for the rest.  A model trained
    on all rows (m2.0, info without k_min) is used as before, without this.

    The v2 model needs each ball's law as 12 Legendre moments.  The driver
    hands over the material index of each ball and the packed laws `ang`
    (core/scattering.pack); the moments of every row of `ang` are worked
    out once (core.scattering.moments, the same function the training
    pool used) and looked up per ball.  So isotropic accuracy never depends
    on the v2 model, which is what Innovation and Development agreed.
    """

    aniso = True

    def __init__(self, family, net, cnorm, ynorm, info, iso, seed=12345,
                 device="cpu", name=None, precision="fp32", compile=False,
                 graphs=False):
        self.family, self.iso = family, iso
        self.name = name or f"{family.name}_v2"
        self.engine = Engine(family, net, E.C_DIM_ANISO, E.X_DIM_ANISO,
                             device, precision, compile, graphs, seed)
        self.device, self.net = self.engine.device, self.engine.net
        self.gen = self.engine.gen
        self.cnorm, self.ynorm, self.info = cnorm, ynorm, info
        self.r_min, self.r_max = info["r_min"], info["r_max"]
        self.draws = 0
        self.single = 0                  # balls finished by the first scatter
        self.k_min = int(info.get("k_min", 1))
        self._seed, self._used = seed, 0
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

    def draw(self, R, g):
        """The FAMILY's draw alone: (cos_n, mu, phi, s) for radii R (n,)
        and moments g (n, 12).  For an m2.1 model this is the k >= 2 part
        only; whole balls come from sample_aniso."""
        return self._draw_collect(self._draw_submit(R, g))

    def _draw_submit(self, R, g):
        R = np.asarray(R, np.float64)
        t = self.engine.submit(self.cnorm(E.condition_aniso(R, g)).astype(
            np.float32))
        self.draws += R.size
        return R, t

    def _draw_collect(self, ticket):
        R, t = ticket
        return E.decode_aniso(R, self.ynorm.undo(self.engine.collect(t)))

    def sample_aniso(self, R, mats, ang, seeds=None, idx=None):
        """(cos_n, mu, phi, s) for balls of radius R in materials mats of
        the packed laws ang.  seeds/idx: the particles' random streams, as
        the exact walk takes them; without them the backend uses its own
        (seeded once, never reused)."""
        return self.collect(self.submit_aniso(R, mats, ang, seeds, idx))

    def submit_aniso(self, R, mats, ang, seeds=None, idx=None):
        """sample_aniso in two halves: the exact first-scatter walk runs
        now (CPU), the family's draw for the rest is started (GPU) and a
        ticket returned; collect(ticket) waits and fills them in.  Same
        numbers as sample_aniso for the same calls in the same order."""
        R = np.ascontiguousarray(R, np.float64)
        mats = np.ascontiguousarray(mats, np.int64)
        g = self.moments(ang)
        if self.k_min < 2:
            return ("all", self._draw_submit(R, g[mats]))
        from core import rng
        from core import scattering as SC
        from .. import walk
        if R.size and np.any(np.asarray(ang)[mats, 0] == SC.ISO):
            # the driver sends isotropic materials to sample(); the exact
            # first-scatter walk does not return on an ISO row (found
            # 16 Nov 2026), so refuse rather than hang
            raise ValueError("sample_aniso got a ball in an isotropic "
                             "material; isotropic balls go to sample()")
        if seeds is None:
            seeds = rng.streams(np.uint64(self._seed), self._used, R.size)
            idx = np.arange(R.size)
            self._used += R.size
        ex, cn, mu, phi, s = walk.first_scatter_many_aniso(
            R, mats, np.ascontiguousarray(ang, np.float64), seeds,
            np.ascontiguousarray(idx, np.int64))
        rest = np.flatnonzero(~ex)
        self.single += R.size - rest.size
        t = self._draw_submit(R[rest], g[mats[rest]]) if rest.size else None
        return ("rest", (cn, mu, phi, s, rest, t))

    def submit(self, R, seeds=None, idx=None):
        """Isotropic balls, started (the v1 model's submit if it has one)."""
        if hasattr(self.iso, "submit"):
            return ("iso", self.iso.submit(R, seeds, idx))
        return ("done", self.iso.sample(R, seeds, idx))

    def collect(self, ticket):
        """The result of submit() or submit_aniso()."""
        kind, t = ticket
        if kind == "iso":
            return self.iso.collect(t)
        if kind == "done":
            return t
        if kind == "all":
            return self._draw_collect(t)
        cn, mu, phi, s, rest, dt = t
        if dt is not None:
            cn[rest], mu[rest], phi[rest], s[rest] = self._draw_collect(dt)
        return cn, mu, phi, s

    def warmup(self, sizes=(1024, 4096, 16384, 65536)):
        """Compile / record graphs before timing (fast.Engine.warmup)."""
        self.engine.warmup(sizes)

    def describe(self):
        return {"name": self.name, "family": self.family.name,
                "nfe": self.family.nfe, "interface": 2,
                **self.engine.describe(),
                "first_scatter_exact": self.k_min >= 2,
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
