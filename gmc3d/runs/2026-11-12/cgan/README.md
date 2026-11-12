# m2.2: the conditional GAN (cGAN), the fourth model family

Code: `ball/models/cgan.py` (c5c869d).  Approved by Phani on 12 Nov 2026,
agreed with the Innovation thread first.

## What was run

| step | command | output |
|---|---|---|
| tuning, 4 trials of 1,500 steps | scratch runs, scored with ball_metrics at R = 1.5, 5, 20 | listed in the cgan.py docstring |
| v1 seeds 0 and 1, 8,000 steps each | `v1.sh` (`run.py train --family cgan --seed s`) | `train_cgan_seed{0,1}.log` |
| v1 comparison of all four families | `run.py compare` (same command as m1.x) | `compare_v1.log`, `compare.json` |
| v2 seeds 0 and 1, 24,000 steps each | `train_v2.py <seed>` | `train_cgan_v2_seed{0,1}.log` |
| v2 comparison (beside cvae_v2_seed0, nflow_v2_seed0) and gate2 on each cGAN seed | `judge_v2.py` | `compare_v2.log`, `compare_v2.json`, `gate/` |

Settings scored: trial 3 (generator lr 2.5e-4, discriminator 1e-3, Adam
betas (0, 0.99), 2 discriminator steps per generator step, R1 gamma 0.1,
noise 8).  Weights 149,955 at v1 (generator alone 85,666), inside the
fairness band (146,554 +-20%).

## v1 results (exact walk as reference, same session, same draws)

| backend | weights | worst \|z\| | KS excess | us/draw @10k |
|---|---|---|---|---|
| nflow / seed 1 | 146,554 | 3.3 / 4.2 | +0.002 / +0.004 | 3.1 / 2.5 |
| cfm / seed 1 | 149,090 | 7.4 / 5.7 | +0.008 / +0.005 | 12.5 / 14.8 |
| cvae / seed 1 | 148,776 | 3.9 / 4.7 | +0.004 / +0.005 | 0.73 / 0.76 |
| **cgan / seed 1** | 149,955 | **24.8 / 23.2** | **+0.023 / +0.032** | 0.78 / 0.80 |
| oracle | - | 2.2 | 0.000 | 0.86 |

The cGAN is as cheap as the cVAE (one generator pass) but the least
accurate family by a wide margin:

* mean path length s is short at every radius: -1.3% to -4.6% (seed 0),
  -0.8% to -2.1% (seed 1), z down to -25.
* the weight a ball passes on in an absorbing medium (kappa = 0.1) is
  low in big balls: -25% at R = 20 and -44% at R = 29 (seed 0), -23% and
  -10% (seed 1).  That is the short-path tail being under-drawn, the
  failure the proposal predicted for GANs ("mode dropping").
* the exit direction mu is close (|z| <= 5).

## Diagnosis

The discriminator has stopped seeing anything.  Its loss reaches log 4 =
1.3863 by step 4,000 on both seeds and stays there; at the end its output
on 65,536 validation rows has a spread of 0.004 (seed 0) and 0.002
(seed 1), so it scores every walk, real or fake, the same.  The
generator's marginal moments match to 0.01 standard deviation.

Why: in the encoded space the remaining errors are about 0.02 standard
deviation, which is the size of the sampling noise of a 4,096-row batch
(1/sqrt(4096) = 0.016).  A discriminator is a classifier, so it learns
only from differences it can tell apart in a batch; below that, its
gradient is noise and the generator gets no push.  The likelihood
families (flow, cVAE) are scored on every single walk instead, so they
keep improving below that level.  This is a property of the method, not
a tuning gap: trial 4 (no R1 penalty) gave the same picture, and the
4-trial tuning budget is spent.

## v2 results (anisotropic balls, exact first scatter, k >= 2 rows)

Same data as m2.1 (hash e7a34544f0048944), 24,000 steps, 150,981 weights
(generator alone 86,180).  Training took 2,664 s and 2,937 s; the
discriminator loss sat at log 4 from about step 6,000 on both seeds.

Model-side comparison (`compare_v2.log`, 7 held-out laws + isotropic,
R = 1, 3, 5, 10, 30, 100,000 draws each, same session):

| backend | held-out laws worst \|z\| | KS excess | isotropic worst \|z\| | us/draw @10k |
|---|---|---|---|---|
| cgan_v2_seed0 | 21.2 | +0.030 | 6.4 | 1.20 |
| cgan_v2_seed1 | 12.7 | +0.022 | 8.7 | 1.01 |
| cvae_v2_seed0 (accepted) | 7.9 | +0.014 | 6.0 | 1.41 |
| nflow_v2_seed0 | 5.3 | +0.006 | 2.6 | 3.62 |

Innovation's gate2 (whole balls, final rule: |z| > 3.29 AND absorbed
share off by > 2%):

| model | failing checks of 343 | weight |z| alone | at 0.5% / 1% | model / exact walk us per ball |
|---|---|---|---|---|
| cgan_v2_seed0 | **10** (9 weight + 1 KS s 0.032) | 75 | 62 / 34 | 1.82 / 1.81 |
| cgan_v2_seed1 | **4** (all weight) | 45 | 30 / 16 | 1.97 / 1.95 |
| cvae_v2_seed0 (m2.1, accepted) | 0 | | | 1.75 / 1.35 |

Every failing weight check has the model absorbing 2.1-4.5% too little
in big balls (R >= 10) at the weakest absorption (kappa = 0.001): the
same short-path-length bias as at v1 (s z down to -7 in gate2).  At v2
the gap to the other families is smaller than at v1 because the exact
walk to the second collision takes the hardest part (the single-scatter
walks) off every model.

## Limit

Only the fairness-rule budget (4 trials, 8,000 steps at v1) was used.
Larger batches or many more steps might help the discriminator, but they
would break the same-data, same-steps rule all families follow.
