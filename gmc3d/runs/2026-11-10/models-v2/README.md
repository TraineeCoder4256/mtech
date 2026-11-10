# First interface v2 models (m2.0, 10 Nov 2026)

These are anisotropic balls: 13 inputs (R, g_1..g_12) and 4 outputs (cos_n, mu, phi, s). All three families were trained at seeds 0 and 1 with code cefcd57 under the fairness rules.

Data:
- 40,000,000 + 500,000 rows from Innovation's `dataset_aniso` (hash f444267f...).
- 1,000,000 + 12,500 isotropic rows (hash db43737b...).
- Combined hash 5a07321cf122e667.

Training: 24,000 steps at batch 4,096, on 4 CPU cores. Weights are in `/mnt/project-files/models/<family>_v2_seed<k>/`.

Files:
- Run: `train_all.py`. It hit the 2 h background limit during cvae seed 1, so `finish.sh` retrained that model from scratch; its first 1,000 steps are digit-identical to the killed run in `train_cvae_v2_seed1_killed.log`. `finish.sh` also ran the comparisons.
- Model-side check: `compare.log` / `compare.json` (`ball/models/compare_aniso.py`).
- Innovation's acceptance gate: `gate/gate2_*.{log,json}`, from `python run.py gate2 <family>_v2_seed<k>`, v2.1.2 e12d6cb.

## Training

| model | weights | final val loss | train time |
|---|---|---|---|
| nflow seed 0 / 1 | 165,652 | NLL 2.158 / 2.265 | 2110 / 2011 s |
| cfm seed 0 / 1 | 149,732 | CFM 1.095 / 1.101 | 931 / 946 s |
| cvae seed 0 / 1 | 149,936 | -ELBO 2.965 / 2.834 | 1076 / 994 s |

The flow fits the held-out walks much better than the cVAE here (NLL 2.16-2.27 against an upper bound of 2.83-2.97). That reverses v1, where the cVAE's bound was below the flow's NLL.

## Innovation's gate: every model fails

The gate compares 7 held-out laws at 7 radii, 100k walks against 100k draws each. A row passes only if every weight |z| <= 3.29, every KS <= 0.03 and no draw is out of range.

| model | failing checks / 343 | rows failing, R <= 3 | rows failing, R >= 5 | worst weight z, R >= 5 | us per draw (gate) |
|---|---|---|---|---|---|
| nflow seed 0 | 91 | 21 / 21 | 8 / 28 | 5.3 | 7.6 |
| nflow seed 1 | 88 | 21 / 21 | 12 / 28 | 7.2 | 7.9 |
| cfm seed 0 | 111 | 21 / 21 | 25 / 28 | 8.3 | 36.3 |
| cfm seed 1 | 76 | 21 / 21 | 16 / 28 | 7.3 | 36.2 |
| cvae seed 0 | 151 | 21 / 21 | 25 / 28 | 11.5 | 2.8 |
| cvae seed 1 | 188 | 21 / 21 | 22 / 28 | 11.1 | 2.8 |
| exact walk | 0 (Innovation's self-test) | | | | 1.8-2.0 |

### Cause 1, R <= 3: phi has point masses no continuous model can make

A walk that scatters exactly once stays in one plane: the first flight runs along u0, then there is one turn, then the exit. So n, u0 and the exit direction are coplanar, and phi is exactly 0 or pi.

Measured on the held-out laws, 100k exact walks each: the share of walks with phi at 0 or pi equals the share with k = 1, to the digit:

| R | share with phi at 0 or pi |
|---|---|
| 1 | 37-53% |
| 3 | 6-13% |
| 10 and 30 | 0 |

Every family draws phi from a smooth density, so its KS distance for phi is just the size of that point mass. That is why KS phi is the same for all six models (0.480 for p1 at R = 1, 0.534 for p5_order2). The same walks also give s and cos_n a sharp single-scatter piece, which is where most of the R <= 1.5 s and weight z (up to 36 for the cVAE) come from.

This is not a family's fault. A model-side fix inside the interface is to give every family a small "one scatter or more" head: draw k = 1 with a learned probability, then for k = 1 draw (cos_n, mu, s) and phi in {0, pi}. This is proposed to Innovation and not built.

### Cause 2, R >= 5: ordinary accuracy misses

There are no point masses at R >= 5.
- **nflow** is closest: 8 and 12 failing rows of 28, worst weight |z| 5.3 / 7.2, KS within 0.03 everywhere but once.
- **cfm** has a path-length bias at R = 20-29 (s z about +5).
- **cvae** blurs phi (KS up to 0.056) and has weight z up to 11 on the backward law.

### Isotropic rows and the P3 reference

At g = 0, compared with the isotropic law at R = 1-30 (`compare.log`, "isotropic" rows):
- The models learn the lean of cos_n at R >= 5: mean cos_n |z| <= 3.1 for all six.
- At R <= 3 it is mixed: cfm |z| <= 1.5, nflow up to 6.8, cvae seed 1 up to 17.7.
- They share the R = 1 phi point mass.

Innovation's P3 point at R = 5 (cos_n 0.382, mu 0.778, phi 1.864, s 8.91): every model is within 0.013 / 0.004 / 0.04 / 0.2.

## Cost

These are the gate's numbers at batch 100,000, on the same CPU. The exact walk costs 1.8-2.0 us per draw.
- The cVAE is the cheapest network (2.8 us), but still 1.4-1.5x the exact walk.
- The flow is 7.6-7.9 us.
- Flow matching is 36 us (10 network passes).

As at v1, no network beats the exact walk on this CPU.

## Limits

- One sample of 100k per (law, R); |z| moves by about 1 between runs.
- No tuning trials yet (budget 4 per family).
- 24,000 steps is my 3x guess from v1, not a tested optimum.
- CPU timings only.
