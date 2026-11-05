"""Backend 3 of 3: a trained generative network -- now any family in models/.

Since model version m1.0 the network code lives in ball/models/ (the model
side): the flow is models/nflow.py, flow matching models/cfm.py, the
shared training loop models/training.py.  This file only keeps the names
older code imports (checks/gpu_bench.py, net_profile.py, net_variants.py,
precision_check.py, feature_cost.py, and run.py's `network` backend), all
pointing at the flow, unchanged:

    Network      the flow behind the backend face (models/backend.py)
    SplineFlow, Coupling, FiLMBlock, rq_spline, _knots, EMA, the shape and
                 training constants
    train()      train the flow, as before
    MODEL_DIR    gmc3d/data/models; a trained flow is in nflow_seed<k>/

Moving the code changed no operation: the flow trains to the same weights
from the same seed and draws the same samples from the same weights.
"""

from pathlib import Path

from . import data as D
from . import interface as I
from .models import fairness, store
from .models.backend import ModelBackend
from .models.nflow import (BINS, BOUND, EMB, LAYERS, MIN_HEIGHT,  # noqa: F401
                           MIN_SLOPE, MIN_WIDTH, WIDTH, Coupling, FiLMBlock,
                           NFlow, SplineFlow, _knots, rq_spline)
from .models.training import EMA  # noqa: F401

MODEL_DIR = store.MODEL_DIR

R_MIN, R_MAX = I.R_MIN, I.R_MAX
N_TRAIN, N_VAL = fairness.N_TRAIN, fairness.N_VAL
STEPS, BATCH, LR = (fairness.TRAIN[k] for k in ("steps", "batch", "lr"))
WARMUP, EMA_DECAY, VAL_EVERY = (fairness.TRAIN[k] for k in
                                ("warmup", "ema_decay", "val_every"))
SEED = 0


class Network(ModelBackend):
    """The flow as a backend (the name run.py's `network` has always had)."""

    def __init__(self, net, cnorm, ynorm, info, seed=12345, device="cpu"):
        super().__init__(NFlow, net, cnorm, ynorm, info, seed, device,
                         name="network")

    @classmethod
    def load(cls, out_dir, seed=12345, device="cpu"):
        _, cfg, state, info, cnorm, ynorm = store.read(out_dir)
        net = NFlow.build(cfg)
        net.load_state_dict(state)
        return cls(net, cnorm, ynorm, info, seed, device)


def train(out_dir=None, steps=STEPS, seed=SEED, r_min=R_MIN, r_max=R_MAX,
          n_train=N_TRAIN, n_val=N_VAL, log=print):
    """Make (or reuse) the data, train the flow, save.  Returns Network."""
    out_dir = Path(out_dir or MODEL_DIR / store.folder("nflow", seed))
    data = D.dataset(n_train, n_val, r_min, r_max, seed=fairness.DATA_SEED)
    data["seed"] = fairness.DATA_SEED
    net, cfg, facts = NFlow.fit(data, seed=seed, steps=steps, log=log)
    store.save(out_dir, NFlow, cfg, net, data, facts)
    return Network.load(out_dir)
