# Interface v2 models with exact single-scatter exits (m2.1, 10 Nov 2026)

Same families, sizes, steps and data as m2.0 (`../models-v2/README.md`), with one change, the same for every family. The backend walks each ball exactly up to its second collision. Innovation's `walk.first_scatter_many_aniso` (v2.1.3) does this on the particles' own random streams. Walks that exit there are done, exactly, with phi = pi. The family is asked only for the rest, so every family is trained on the k >= 2 rows only.

- Code: fb89e55.
- Training data: 37,024,760 + 462,904 rows kept of 41.5M, hash e7a34544f0048944.
- Run: `train_seed.py 0`, `train_seed.py 1`, then `judge.py`.
- Gate: Innovation's v2.1.4 (bdba9f0), which now scores whole balls through `sample_aniso` on streams of its own.
- Weights: `/mnt/project-files/models/<family>_v2_seed<k>/`. The m2.0 weights moved to `.../models/m2.0/`.

## Training

| model | weights | final val loss | train time |
|---|---|---|---|
| nflow seed 0 / 1 | 165,652 | NLL 4.675 / 4.668 | 1984 / 2012 s |
| cfm seed 0 / 1 | 149,732 | CFM 1.316 / 1.312 | 910 / 910 s |
| cvae seed 0 / 1 | 149,936 | -ELBO 4.689 / 4.701 | 973 / 983 s |

These losses are on k >= 2 walks only, so they do not compare with m2.0's. The single-scatter rows were very sharp and lowered the average NLL a lot. The two seeds now agree closely; m2.0's differed by 0.1 nats. The flow and the cVAE's bound are within 0.03 nats of each other.

## Innovation's gate (343 checks: 7 held-out laws x 7 radii)

| model | failing checks, m2.0 -> m2.1 | rows failing, R <= 3 | rows failing, R >= 5 | worst weight z | worst KS | us per ball |
|---|---|---|---|---|---|---|
| nflow seed 0 | 91 -> **11** | 2 / 21 | 2 / 28 | 6.2 | 0.015 | 6.0 |
| nflow seed 1 | 88 -> **18** | 2 / 21 | 5 / 28 | 7.6 | 0.018 | 6.0 |
| cfm seed 0 | 111 -> 22 | 4 / 21 | 6 / 28 | 6.9 | 0.017 | 25.4 |
| cfm seed 1 | 76 -> 25 | 6 / 21 | 4 / 28 | 5.5 | 0.027 | 25.9 |
| cvae seed 0 | 151 -> 21 | 3 / 21 | 8 / 28 | 7.2 | 0.021 | 1.9 |
| cvae seed 1 | 188 -> 27 | 5 / 21 | 6 / 28 | 8.1 | 0.021 | 2.0 |
| exact walk | 0 | | | | | 1.6-1.9 |

- **The point mass is gone.** Every KS distance is now inside the gate's 0.03, on every row and model. In m2.0 the worst was 0.53, and every R <= 3 row failed.
- **Every remaining failure is a surviving-weight z between 3.3 and 8.1.** These are small mean shifts in path length that 100,000 samples can detect. For example, nflow seed 0's 11 failures are the clipped-P3 law at R = 5 and 10 (z +4 to +6.2) and two R = 1 rows (|z| 3.4-4.0).
- **The gate still fails all six,** because its weight rule is |z| <= 3.29 with no size tolerance. Innovation will put a two-condition rule to Phani: fail only if |z| > 3.29 AND the absorbed share is off by more than a tolerance he picks, with 1% suggested. Innovation will report both readings with these models.
- **Ranking:**
  - Accuracy: nflow ahead of cfm and cvae, which are close to each other.
  - Cost per whole ball: cvae 1.9-2.0 us, about the exact walk's 1.6-1.9; nflow 6.0 (7.6-7.9 in m2.0, since the network now runs only for walks that go on); cfm 25.

## Model-side check (`compare.log`)

Whole balls, 5 radii x 8 laws (the 7 held-out laws plus isotropic), 100k each:

| model | worst \|z\|, held-out | worst \|z\|, isotropic | KS excess, held-out |
|---|---|---|---|
| nflow | 5.3 / 6.8 | 2.6 / 2.3 | +0.006 / +0.008 |
| cfm | 7.6 / 5.6 | 5.3 / 3.0 | +0.013 / +0.019 |
| cvae | 7.9 / 8.2 | 6.0 / 6.0 | +0.014 / +0.017 |

Innovation's P3 point at R = 5 (cos_n 0.382, mu 0.778, phi 1.864, s 8.91): every model is within 0.003 / 0.002 / 0.004 / 0.03. In m2.0 the spread was 0.013 / 0.004 / 0.04 / 0.2.

## Limits

- One sample of 100k per row; z moves by about 1 between runs.
- No tuning trials yet (budget 4 per family).
- 24,000 steps is untested as an optimum.
- CPU timings only, on a container whose timings spread 8-15%.
