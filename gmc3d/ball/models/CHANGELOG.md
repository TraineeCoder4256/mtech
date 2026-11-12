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

## m1.1 (interface v1) - 5 Nov 2026

- First comparison: nflow and cfm, seeds 0 and 1, trained on the full data
  (`runs/2026-11-05/models/README.md`). Flow worst |z| 3.3-4.2 vs the
  exact walk, flow matching 5.7-7.4; flow matching costs 3.5-7x the flow
  per draw. Weights in `/mnt/project-files/models/`.
- `store.py`: a manifest's "+dirty" flag now looks at code only (run
  outputs and data/ no longer count).

## m1.2 (interface v1) - 5 Nov 2026

- Family 3, `cvae`: conditional VAE, encoder 3 / decoder 4 FiLM blocks,
  latent 2, 148,776 weights (+1.5%), one decoder pass per draw. Trained at
  seeds 0 and 1 (code 3f96133). Worst |z| 3.9 / 4.7 (only the R = 29,
  kappa 0.1 tail), -ELBO 2.164 below the flow's NLL 2.183, 2-3x cheaper
  per draw than the flow. Results in `runs/2026-11-05/models/README.md`.

## m2.0 (interface v2) - 10 Nov 2026

Every family at interface v2 (anisotropic balls: 13 inputs R, g_1..g_12;
4 outputs cos_n, mu, phi, s), agreed with Innovation on 10 Nov 2026 and
built on its handoff (0b232f4). Interface v1 models are untouched.

- Families are dimension-agnostic: sizes come from the config (`x_dim`,
  `c_dim`; v1's when absent). `Family.config_for(interface)`;
  `CONFIG_ANISO` per family (cVAE: latent 4). The flow's couplings bend
  half the targets and read the other half, cycling through every split
  (`nflow.masks`): for d = 2 that is exactly v1's alternation.
  Checked before pushing: the three saved v1 models draw identically with
  the old and new code, and 300 training steps (nflow, cVAE) give
  identical weights (166 arrays, all equal).
- Encoding (`encoding.py`): c = (log R, g_1..g_12); y = (logit((1+cos_n)/2),
  logit mu, logit(phi/pi), log(s/R - 1)), standardised. `prepare_aniso`
  builds the 41M-row set in float32.
- Data (`fairness.py`, `pipeline.aniso_data`): Innovation's
  `dataset_aniso(40.5M, 1, 30, seed 1)` (hash f444267f...) plus 1.01M
  isotropic rows walked by `walk_many_aniso` with a flat table
  (`pipeline.iso_rows`, own streams; hash db43737b...); 40M + 1M train,
  0.5M + 12.5k validation. Steps 24,000 (3x v1), all else as v1.
- Sizes: nflow 165,652 (the v2 reference), cfm 149,732 (-9.6%), cVAE
  149,936 (-9.5%).
- Store: v2 models in `<family>_v2_seed<k>/`, manifest interface 2;
  `store.read(..., expect=2)`; a v1 model never loads as v2 or back.
- Backend (`backend.AnisoBackend`): `aniso = True`; `sample()` goes to the
  accepted v1 model (default the flow, seed 0); `sample_aniso(R, mats,
  ang, ...)` works out each material's moments once from the packed laws
  (`law_moments`, equal to `laws.moments` on every held-out law).
- `compare_aniso.py`: model-side check against the exact walk on the 7
  held-out laws and the isotropic law at R = 1, 3, 5, 10, 30, plus
  Innovation's P3 reference point and cost per draw. Innovation's gate
  decides acceptance.
- `models.train/load(..., interface=2)`. `tests/test_models_m20.py`.

## m2.1 (interface v2) - 10 Nov 2026

The single-scatter fix, agreed with Innovation (its v2.1.3 2278ce3 gives
`walk.first_scatter_many_aniso`).

- Why: m2.0 failed Innovation's gate at every R <= 3 for every family. A
  walk that scatters once stays in one plane and exits with phi = pi (to
  rounding): 37-53% of walks at R = 1, a point mass no smooth model can
  draw (runs/2026-11-10/models-v2/README.md).
- `AnisoBackend.sample_aniso` walks every ball exactly to its second
  collision on the particles' own streams; balls that exit there are the
  walk's own (exact), the rest are drawn by the family. Exact mixture, no
  extra weights, the same for every family. Without streams the backend
  uses its own (seeded) ones. `draw(R, g)` is the family's k >= 2 part.
- Every family trains on k >= 2 rows only (`fairness.K_MIN_ANISO = 2`,
  `prepare_aniso(..., k_min)`, recorded in info and manifest). An m2.0
  model (no k_min) answers as before. m2.0 weights moved to
  `/mnt/project-files/models/m2.0/`.
- `compare_aniso` scores whole balls on streams of its own; cost per draw
  is per whole ball, streams made outside the timer.
- Tests: the first-scatter balls equal the walk's k = 1 walks on the same
  streams, outputs identical, phi = pi; k_min filter.

## m2.2 (interfaces v1 and v2) - 12 Nov 2026

- Family 4, `cgan` (`cgan.py`, one registry line), approved by Phani 12 Nov
  and agreed with Innovation. Generator + discriminator (FiLM blocks of
  width 64, 4 + 3, noise 8), non-saturating loss with R1 penalty, its own
  loop copying fit_loop's conventions (EMA of the generator, same steps,
  batch, warm-up and cosine, validation, time cap). 149,955 weights at v1
  (generator 85,666, the draw cost), 150,981 at v2 (generator 86,180).
  Settings and the 3 short tuning trials are listed in its docstring.
- At v2 it gets the exact first scatter through AnisoBackend and trains on
  k >= 2 rows, like the others.
