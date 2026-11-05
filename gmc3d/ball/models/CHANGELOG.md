# Model-side changelog (ball/models/)

Model versions are numbered m1.0, m1.1, ... separately from gmc3d's own
VERSION, which Innovation owns. Each entry names the interface version
(ball/interface.py) it targets. A model only becomes transport's default
(`network`) when Innovation's acceptance gate passes and Innovation records
the acceptance in the top-level CHANGELOG.

## m1.0 (interface v1) - 5 Nov 2026

The pluggable pipeline from gmc2d, rebuilt for the ball step.

- `ball/models/`: plug-in face (`base.py`), lazy registry (`__init__.py`),
  shared training loop (`training.py`), fairness rules (`fairness.py`),
  encoding (`encoding.py`, moved out of `ball/data.py`), trained-model
  store with `manifest.json` and a shared copy in
  `/mnt/project-files/models/` (`store.py`), backend wrapper (`backend.py`),
  side-by-side comparison (`compare.py`).
- Family 1, `nflow`: the flow from `ball/network.py`, moved unchanged.
  Trains to identical weights from the same seed and draws identical
  samples from the same weights (checked against v1.6.1 before pushing);
  an old v1.6.1 checkpoint loads and draws identically.
- Family 2, `cfm`: conditional flow matching ported from gmc2d, 1 condition
  and 2 targets, 149,090 weights (+1.7% on the flow's 146,554), Heun with 5
  steps (10 network passes per draw).
- `ball/network.py` and `ball/data.py` keep every name older code imports.
- `run.py`: any family is a backend by name (`cfm`, `cfm_seed1`);
  `network` still means the flow at seed 0. `train --family <name>`,
  new `compare`. Trained flows now live in `data/models/nflow_seed<k>/`
  (was `flow_seed0/`).
- `tests/test_models_m10.py`: every registered family is tested by being
  registered (face, fairness budget, train/save/load/draw, interface
  refusal).

Not yet: trained models at two seeds and the first comparison (next).
