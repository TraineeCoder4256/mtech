# First model comparison, m1.0 (5 Nov 2026)

Two families, two training seeds each, under the shared fairness rules
(`ball/models/fairness.py`): the same 4,000,000 + 500,000 exact walks
(data hash 472ebbedcf7d4093), the same loop and 8,000 steps at batch 4,096,
weights within +-20% of the flow's. Code at commit 96deaa0 (interface v1).
Trained on this container's 4 CPU cores. Weights are in
`/mnt/project-files/models/<family>_seed<k>/` with a `manifest.json` each.

Command: `python run.py compare --backends nflow nflow_seed1 cfm cfm_seed1
oracle table`. Full output in `compare.log`, numbers in `compare.json`,
training logs in `train_*.log`.

## Training

| model | weights | network passes per draw | final val loss | train time |
|---|---|---|---|---|
| nflow seed 0 | 146,554 | 1 | NLL 2.1831 | 385 s |
| nflow seed 1 | 146,554 | 1 | NLL 2.1902 | 374 s |
| cfm seed 0 | 149,090 | 10 | CFM 1.2242 | 189 s |
| cfm seed 1 | 149,090 | 10 | CFM 1.2206 | 181 s |

The two losses measure different things, so they do not compare across
families; only the scores below do.

## Against the exact walk, ball by ball

200,000 exact walks against 200,000 draws at R = 1.5, 3, 5, 10, 20 and
29 mfp (`checks/ball_metrics.py`). "worst |z|" is the largest gap, in
standard errors, over the mean exit cosine, the mean path and the
surviving weight at three absorption levels; under about 3 means "cannot
be told apart at this sample size". "KS excess" is the largest
distribution gap minus its 95% noise level (0.0043).

| backend | worst abs z | KS excess | us per draw at 1k / 10k / 100k |
|---|---|---|---|
| nflow seed 0 | 3.3 | +0.0022 | 6.06 / 2.72 / 3.61 |
| nflow seed 1 | 4.2 | +0.0038 | 5.97 / 2.22 / 3.93 |
| cfm seed 0 | 7.4 | +0.0079 | 21.2 / 13.0 / 25.1 |
| cfm seed 1 | 5.7 | +0.0054 | 28.4 / 12.5 / 23.9 |
| oracle (exact walk) | 2.2 | -0.0003 | 0.82 / 0.75 / 0.72 |
| table | 2.7 | +0.0031 | 0.11 / 0.021 / 0.017 |

Where each family misses:

- **nflow**: path length slightly long in small balls (+0.45% to +0.73% at
  R = 1.5, z up to 4.1 on seed 1), and at R = 29 too much weight survives
  strong absorption (+74% / +69% at kappa 0.1, z about 3): the flow puts a
  little too much probability on short paths in the biggest balls. Same
  sign on both seeds, so it is the model, not the seed.
- **cfm**: path length too long at R = 20 (+1.3% / +0.76%, z 6.6 / 3.8),
  which takes 1.4-1.8% off the surviving weight at kappa 0.01. It does not
  have the flow's R = 29 short-path excess on seed 1 (+0.9%), but seed 0
  shows +28% (z 2.7).

## Verdict

- **Accuracy**: the flow is closer to the exact walk than flow matching at
  the same size and step budget (worst z 3.3-4.2 against 5.7-7.4). Neither
  is exact; both have one region a gate would flag.
- **Cost**: flow matching costs 3.5-7x the flow per draw (10 passes against 1,
  partly offset by lighter passes), and trains in half the time per step.
- On this CPU **both networks cost more per draw than running the exact
  walk itself** (0.7-0.8 us), which is the known result from 5 Oct and the
  2 Nov speed thread; the network only has a chance on a GPU.

Limits: one sample of 200,000 per radius (z values move by about 1 between
runs); no tuning trials yet for cfm (budget 4); CPU timings only, on a
container whose timings spread 8-15%.

## Added later the same day: the conditional VAE (m1.2, commit 3f96133)

Third family, `ball/models/cvae.py`: encoder 3 FiLM blocks, decoder 4,
width 64, latent size 2, 148,776 weights (+1.5% on the flow; a draw uses
the decoder only, 85,412). Same data, loop, 8,000 steps, seeds 0 and 1.
`compare.log` / `compare.json` now hold all six models from one session;
the flow and flow matching rows are identical to the first run above
(their draws are seeded), only the timings moved.

| model | final val loss (nats) | train time |
|---|---|---|
| cvae seed 0 | -ELBO 2.1640 | 358 s |
| cvae seed 1 | -ELBO 2.1626 | 337 s |

The negative ELBO is an upper bound on the cVAE's true NLL, so it can be
read against the flow's exact NLL (2.1831 / 2.1902): the cVAE fits the
held-out walks at least as well as the flow, on average.

One session, all six (timings in this session ran about 1.5x slower than
the first: compare rows within this table only):

| backend | worst abs z | KS excess | us per draw at 1k / 10k / 100k |
|---|---|---|---|
| nflow seed 0 | 3.3 | +0.0022 | 7.8 / 3.3 / 6.2 |
| nflow seed 1 | 4.2 | +0.0038 | 9.6 / 3.6 / 7.0 |
| cfm seed 0 | 7.4 | +0.0079 | 32 / 16 / 30 |
| cfm seed 1 | 5.7 | +0.0054 | 35 / 19 / 37 |
| cvae seed 0 | 3.9 | +0.0043 | 4.0 / 1.15 / 2.7 |
| cvae seed 1 | 4.7 | +0.0045 | 2.4 / 1.17 / 2.4 |
| oracle (exact walk) | 2.2 | -0.0003 | 1.8 / 1.1 / 1.1 |
| table | 2.7 | +0.0031 | 0.13 / 0.03 / 0.04 |

Where the cVAE misses:

- Its ONLY |z| > 3.1 is the R = 29 short-path tail at kappa 0.1 (+38% /
  +48%, z 3.9 / 4.7), the same tail as the flow but smaller (flow +74% /
  +69%). At kappa 0.01 it is inside the noise (+0.16% / +0.87%) where the
  flow is not (+1.4% / +1.7%, z 3.3 / 4.1).
- No small-ball path bias (s within 0.1-0.5%, |z| <= 2.6; the flow's was
  z 2.7-4.1).
- Its distribution SHAPE is slightly blurrier than the flow's: KS of mu
  0.0055-0.0073 at most radii against the 0.0043 noise level (flow
  0.0016-0.0065), while the means stay right. That is the known VAE habit
  of smoothing.

Verdict, updated: the cVAE is the best of the three on the numbers the
transport cares about (means, surviving weight) and 2-3x cheaper per draw
than the flow, about as cheap as running the exact walk on this CPU. The
flow keeps the closest distribution shape. My prediction before the run
(that the cVAE would be worse in the tail) was wrong.
