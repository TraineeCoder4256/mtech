"""The plug-in face every model family implements.

Used by: every family file in this folder, pipeline.py, backend.py.
Uses: training.py (the shared loop), encoding.py (the sizes, which
follow ball/interface.py).

Every family does one job: given a ball's radius, turn random noise into
one exit (mu, s).  What differs between families -- a normalizing flow,
flow matching, a VAE, a GAN -- is only HOW noise becomes the exit, and how
the network is trained.  Everything around that draw must be identical for
every family, or a comparison measures the surrounding code instead of the
models.  So the line is drawn here, and it is thin on purpose:

  IN   c   (n, C_DIM) conditions, already encoded and standardised
  OUT  y   (n, X_DIM) targets in the same encoded, standardised space

A family never sees R, mu or s in physical units: encoding.py does that,
in one copy, for every family.  The sizes C_DIM and X_DIM come from
encoding.py, which derives them from ball/interface.py (Innovation's
file).  When Innovation changes the interface (v2 adds anisotropy, stage
3 adds energy), every family resizes by itself; what the model side then
does is retrain and re-tune.

WHAT A NEW FAMILY WRITES (one file, plus one line in __init__.py):

  name, nfe   a label, and network evaluations per draw -- its cost class
              (1 for anything that draws in one pass, 10 for flow matching
              with Heun x 5).  The cost per draw decides whether a ball
              step pays at all, so it goes into every result.
  CONFIG      the network's shape (widths, depths ...), saved with it
  build()     CONFIG -> a fresh torch.nn.Module
  loss()      (net, y, c) -> one scalar to minimise on a batch
  draw()      (net, c, generator) -> y: the sample.  Every random number
              must come from `generator`, so a run is reproducible.

and inherits fit(), which runs the shared loop.  A family that cannot
train by one loss per batch (a GAN) overrides fit() and copies the shared
loop's conventions by hand.  The fairness rules are in fairness.py.
"""

from .encoding import C_DIM, X_DIM
from . import fairness
from .training import fit_loop


class Family:
    name = "base"
    nfe = 0
    loss_name = "loss"
    CONFIG = {}
    TRAIN = fairness.TRAIN          # a family may override within its
                                    # tuning budget (fairness.TUNING_RUNS)

    @classmethod
    def build(cls, config):
        raise NotImplementedError

    @staticmethod
    def loss(net, y, c):
        raise NotImplementedError

    @classmethod
    def draw(cls, net, c, generator):
        raise NotImplementedError

    @classmethod
    def fit(cls, data, *, seed, steps=None, time_cap=fairness.TIME_CAP,
            log=print):
        """Train on encoding.prepare() output.
        Returns (net, config, facts)."""
        cfg = dict(cls.CONFIG)
        tr = dict(cls.TRAIN)
        if steps:
            tr["steps"] = steps
        net, facts = fit_loop(lambda: cls.build(cfg), cls.loss, data,
                              seed=seed, time_cap=time_cap, log=log,
                              loss_name=cls.loss_name, **tr)
        facts["train"] = tr
        return net, cfg, facts

    @classmethod
    def describe(cls, config=None):
        return {"name": cls.name, "nfe": cls.nfe,
                "config": dict(config or cls.CONFIG),
                "x_dim": X_DIM, "c_dim": C_DIM}
