"""The rules of a fair comparison between families (gmc2d's, carried over).

Used by: base.py (the shared training settings), pipeline.py (applies the
data and time rules, flags the weight budget), compare.py (prints them).

A family can only be said to beat another if neither was given an
advantage the other did not get.  So, for every family:

  data      the same walks, the same train/validation split and the same
            encoding (pipeline.py builds them; a family never reads data)
  loop      training.fit_loop with TRAIN below, unless the family cannot
            use it (a GAN); then its own loop copies the conventions
  steps     the same step budget, TRAIN["steps"], and the same wall-clock
            cap, TIME_CAP
  size      a weight count within +-20% of the reference family's
            (REFERENCE, the flow that gmc3d shipped with: 146,554 weights)
  tuning    at most TUNING_RUNS short trials of a family's own settings
            before the run that is scored, listed in its file's docstring
  seeds     every family is scored at SEEDS, so no ranking rests on one
            lucky initialisation
  scoring   compare.py, unchanged, for every family

The step budget is the flow's existing one (8,000 steps at batch 4,096,
learning rate 2e-3, warm-up 300), so the flow trains exactly as it did.
"""

REFERENCE = "nflow"
BUDGET = 0.20                    # +-20% of the reference's weights
TIME_CAP = 3 * 3600              # seconds of training, per model
TUNING_RUNS = 4
SEEDS = (0, 1)

# the data every family trains on (as ball/network.py v1.6.1)
N_TRAIN, N_VAL = 4_000_000, 500_000
DATA_SEED = 1

TRAIN = {"steps": 8000, "batch": 4096, "lr": 2e-3, "warmup": 300,
         "ema_decay": 0.999, "val_every": 500}

# interface v2 (m2.0, anisotropic balls), agreed with Innovation 10 Nov 2026.
# Data: Innovation's data.dataset_aniso rows (pool of 20,000 training laws,
# seed 1) plus isotropic rows (g = 0) walked by the same walk_aniso, so the
# model meets the isotropic end of the family it will border on.  The
# problem is harder (4 outputs, 13 inputs, 20,000 laws), so 3x the steps;
# everything else as v1.  Size: within +-20% of the v2 flow.
N_TRAIN_ANISO, N_VAL_ANISO = 40_000_000, 500_000
N_TRAIN_ISO, N_VAL_ISO = 1_000_000, 12_500          # 2.5% of the rows
TRAIN_ANISO = {**TRAIN, "steps": 24_000, "val_every": 1000}


def reference_params(interface=1):
    from . import get
    fam = get(REFERENCE)
    return sum(p.numel() for p in
               fam.build(fam.config_for(interface)).parameters())


def check_size(params, interface=1):
    """None if params is inside the budget, else a one-line warning."""
    ref = reference_params(interface)
    lo, hi = (1 - BUDGET) * ref, (1 + BUDGET) * ref
    if lo <= params <= hi:
        return None
    return (f"{params:,} weights is outside the fairness budget "
            f"{lo:,.0f}-{hi:,.0f} ({REFERENCE} +-{100 * BUDGET:.0f}%)")
