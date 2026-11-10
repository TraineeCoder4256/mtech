"""Train a family on the shared data, and load a trained one.

Used by: models/__init__.py (train, load), which run.py calls.
Uses: ball/data.py (Innovation's walks), encoding.py, fairness.py,
store.py, backend.py.

train() is the same for every family, which is the point: the data, the
split, the encoding, the step budget and the time cap are fixed here, so
the only thing that differs between two trained models is the family.
"""

import time

import numpy as np

from .. import data as D
from .. import interface as I
from . import fairness, get, store
from .backend import AnisoBackend, ModelBackend
from . import encoding as E


def train(family, seed=0, steps=None, out_dir=None, log=print, shared=True,
          interface=1):
    if interface == I.ANISO_INTERFACE_VERSION:
        return _train_aniso(family, seed, steps, out_dir, log, shared)
    fam = get(family)
    out_dir = out_dir or store.MODEL_DIR / store.folder(family, seed)
    t0 = time.time()
    data = D.dataset(fairness.N_TRAIN, fairness.N_VAL, I.R_MIN, I.R_MAX,
                     seed=fairness.DATA_SEED)
    data["seed"] = fairness.DATA_SEED
    log(f"data   {fairness.N_TRAIN:,} + {fairness.N_VAL:,} walks, R in "
        f"[{I.R_MIN:g}, {I.R_MAX:g}], mean {data['mean_scatters']:.1f} "
        f"scatters each, {time.time() - t0:.1f} s  (hash {data['sha256']})")
    log(f"model  {fam.name} (interface v{I.INTERFACE_VERSION}, {fam.nfe} "
        f"network pass(es) per draw), seed {seed}, {fam.CONFIG}")
    net, cfg, facts = fam.fit(data, seed=seed, steps=steps, log=log)
    log(f"       {facts['params']:,} weights, {facts['steps_done']:,} steps"
        f" in {facts['train_seconds']:.0f} s")
    warn = fairness.check_size(facts["params"])
    if warn:
        log(f"FAIRNESS: {warn}")
    store.save(out_dir, fam, cfg, net, data, facts, shared=shared)
    return load(family, seed, out_dir=out_dir)


def load(family, seed=0, device="cpu", out_dir=None, name=None,
         interface=1, iso=None):
    """A trained family as a backend.  interface=2 gives the anisotropic
    backend (AnisoBackend); its isotropic balls go to `iso`, a v1 backend
    (default: the flow, seed 0, the model gmc3d accepted)."""
    if interface == I.ANISO_INTERFACE_VERSION:
        return _load_aniso(family, seed, device, out_dir, name, iso)
    path = out_dir or store.locate(family, seed)
    if path is None:
        raise FileNotFoundError(
            f"no trained {family} (seed {seed}): run  python run.py train "
            f"--family {family} --seed {seed}  first")
    fam_name, cfg, state, info, cnorm, ynorm = store.read(path)
    if fam_name != family:
        raise ValueError(f"{path} holds a {fam_name}, not a {family}")
    fam = get(family)
    net = fam.build(cfg)
    net.load_state_dict(state)
    return ModelBackend(fam, net, cnorm, ynorm, info, device=device,
                        name=name)


def count_params(family):
    fam = get(family)
    return sum(p.numel() for p in fam.build(dict(fam.CONFIG)).parameters())



# ------------------------------------------------------ interface v2 (m2.0)
_ANISO_CACHE = {}


def iso_rows(n, r_min, r_max, seed=1, cache=True):
    """n walks with ISOTROPIC scattering, walked by Innovation's
    walk_many_aniso (so cos_n and phi come out as for any law) in the
    make_aniso() layout: g_pool is one row of zeros, every law index 0.
    The law is a flat 33-point table (packed by hand: core.scattering.law
    would call it ISO, which the anisotropic walk does not take).  Own
    random streams (master seed 1000 + seed), so no number is shared with
    the anisotropic rows."""
    from core import rng, scattering as SC
    from .. import data as D, walk
    path = D.DATA_DIR / f"iso_aniso_walks_{n}_{r_min:g}_{r_max:g}_{seed}.npz"
    if cache and path.exists():
        return dict(np.load(path))
    flat = (np.full(SC.DEFAULT_NMU, 0.5),
            np.linspace(0.0, 1.0, SC.DEFAULT_NMU))
    ang = SC.pack([(SC.TAB, flat)])
    gen = np.random.default_rng([seed, 0x150])
    R = np.exp(gen.uniform(np.log(r_min), np.log(r_max), n))
    law = np.zeros(n, np.int64)
    state = rng.streams(np.uint64(1000 + seed), 0, n)
    cn, mu, phi, s, k = walk.walk_many_aniso(R, law, ang, state,
                                             np.arange(n))
    d = {"g_pool": np.zeros((1, I.N_MOMENTS)), "R": R, "cos_n": cn,
         "mu": mu, "phi": phi, "s": s, "k": k, "law": law}
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **d)
    return d


def aniso_data(log=print):
    """The v2 training set every family gets (fairness.py): Innovation's
    rows plus the isotropic slice, encoded and standardised.  Kept in
    memory, so training several models in one process builds it once."""
    if "data" in _ANISO_CACHE:
        return _ANISO_CACHE["data"]
    t0 = time.time()
    na, ni = (fairness.N_TRAIN_ANISO + fairness.N_VAL_ANISO,
              fairness.N_TRAIN_ISO + fairness.N_VAL_ISO)
    raw = D.dataset_aniso(na, I.R_MIN, I.R_MAX, seed=fairness.DATA_SEED)
    iso = iso_rows(ni, I.R_MIN, I.R_MAX, seed=fairness.DATA_SEED)
    hashes = [D.data_hash(raw), D.data_hash(iso)]
    data = E.prepare_aniso([(raw, fairness.N_TRAIN_ANISO),
                            (iso, fairness.N_TRAIN_ISO)],
                           fairness.N_TRAIN_ANISO + fairness.N_TRAIN_ISO,
                           I.R_MIN, I.R_MAX, hashes)
    data["seed"] = fairness.DATA_SEED
    del raw, iso
    log(f"data   {na:,} anisotropic + {ni:,} isotropic walks (interface "
        f"v2), mean {data['mean_scatters']:.1f} scatters, "
        f"{time.time() - t0:.0f} s  (hash {data['sha256']}; parts "
        f"{hashes[0][:16]}, {hashes[1][:16]})")
    _ANISO_CACHE["data"] = data
    return data


def _train_aniso(family, seed, steps, out_dir, log, shared):
    v = I.ANISO_INTERFACE_VERSION
    fam = get(family)
    out_dir = out_dir or store.MODEL_DIR / store.folder(family, seed, v)
    data = aniso_data(log)
    cfg = fam.config_for(v)
    log(f"model  {fam.name} (interface v{v}, {fam.nfe} network pass(es) "
        f"per draw), seed {seed}, {cfg}")
    net, cfg, facts = fam.fit(data, seed=seed, steps=steps, log=log,
                              interface=v)
    log(f"       {facts['params']:,} weights, {facts['steps_done']:,} steps"
        f" in {facts['train_seconds']:.0f} s")
    warn = fairness.check_size(facts["params"], v)
    if warn:
        log(f"FAIRNESS: {warn}")
    store.save(out_dir, fam, cfg, net, data, facts, shared=shared,
               interface=v)
    return load(family, seed, out_dir=out_dir, interface=v)


def _load_aniso(family, seed, device, out_dir, name, iso):
    v = I.ANISO_INTERFACE_VERSION
    path = out_dir or store.locate(family, seed, v)
    if path is None:
        raise FileNotFoundError(
            f"no trained {family} for interface v2 (seed {seed}): train it "
            f"with models.train({family!r}, {seed}, interface=2) first")
    fam_name, cfg, state, info, cnorm, ynorm = store.read(path, expect=v)
    if fam_name != family:
        raise ValueError(f"{path} holds a {fam_name}, not a {family}")
    fam = get(family)
    net = fam.build(cfg)
    net.load_state_dict(state)
    if iso is None:
        iso = load("nflow", 0, device=device, name="network")
    return AnisoBackend(fam, net, cnorm, ynorm, info, iso, device=device,
                        name=name)
