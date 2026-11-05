"""Train a family on the shared data, and load a trained one.

Used by: models/__init__.py (train, load), which run.py calls.
Uses: ball/data.py (Innovation's walks), encoding.py, fairness.py,
store.py, backend.py.

train() is the same for every family, which is the point: the data, the
split, the encoding, the step budget and the time cap are fixed here, so
the only thing that differs between two trained models is the family.
"""

import time

from .. import data as D
from .. import interface as I
from . import fairness, get, store
from .backend import ModelBackend


def train(family, seed=0, steps=None, out_dir=None, log=print, shared=True):
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


def load(family, seed=0, device="cpu", out_dir=None, name=None):
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

