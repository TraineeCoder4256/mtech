"""The model families, behind one plug-in face (the model side of gmc3d).

Used by: run.py (`train --family`, `compare`, and any --backend name that
is a family), ball/network.py (re-exports the flow for older code).

    base.py       the face every family implements, and why it is thin
    nflow.py      family 1: normalizing flow (one pass per draw)
    cfm.py        family 2: conditional flow matching (10 passes per draw)
    cvae.py       family 3: conditional VAE (one decoder pass per draw)
    training.py   the training loop every family shares
    fairness.py   the rules of a fair comparison (data, steps, size, seeds)
    encoding.py   physical (R, mu, s) <-> what a network sees
    backend.py    any trained family behind the transport's backend face
    store.py      where trained models live, and their manifest.json
    compare.py    families side by side against the exact walk
    compare_aniso.py  the same at interface v2 (anisotropic balls, m2.0)
    CHANGELOG.md  model-side versions (m1.0 ...), each naming the interface
                  version it targets

ADDING A FAMILY = one new file here (a Family subclass: name, nfe, CONFIG,
build, loss, draw -- see base.py) + one line in REGISTRY below.  Then

    python run.py train --family <name> --seed 0      (and --seed 1)
    python run.py compare                              every family
    python run.py solve slab --backend <name>          in transport

work with no other change.

INTERFACE V2 (m2.0, anisotropic balls).  Every family also trains at
interface v2 with no family-specific code beyond an optional CONFIG_ANISO:

    models.train("<name>", seed, interface=2)
    models.load("<name>", seed, interface=2)        an AnisoBackend

WHO OWNS WHAT.  This folder is the model side.  What a model must learn
(the inputs and outputs in physical units, and the exact walk that makes
the training data) is Innovation's, in ball/interface.py, ball/walk.py and
ball/data.py.  Which model transport uses by default (`network`) changes
only when Innovation's acceptance gate in checks/ passes and Innovation
accepts it; any family can still be picked by name for experiments.

The registry holds "module:Class" strings and imports a family only when
asked, so importing this package (e.g. for encoding.py) costs nothing.
"""

import importlib

REGISTRY = {
    "nflow": "nflow:NFlow",
    "cfm": "cfm:CFM",
    "cvae": "cvae:CVAE",
}


def names():
    return list(REGISTRY)


def get(name):
    """The Family class registered as `name`."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model family {name!r}; known: "
                       f"{', '.join(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)


def train(family, seed=0, steps=None, out_dir=None, log=print, shared=True,
          interface=1):
    """Train one family on the shared data, save it, return the backend.
    interface=2: the anisotropic version (m2.0)."""
    from .pipeline import train as _train
    return _train(family, seed, steps, out_dir, log, shared, interface)


def load(family, seed=0, device="cpu", out_dir=None, name=None,
         interface=1, iso=None):
    """A trained family as a transport backend (see backend.py).
    interface=2: AnisoBackend, isotropic balls going to `iso` (a v1
    backend; default the flow, seed 0)."""
    from .pipeline import load as _load
    return _load(family, seed, device, out_dir, name, interface, iso)
