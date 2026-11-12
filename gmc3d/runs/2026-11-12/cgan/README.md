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

## Limit

Only the fairness-rule budget (4 trials, 8,000 steps at v1) was used.
Larger batches or many more steps might help the discriminator, but they
would break the same-data, same-steps rule all families follow.
