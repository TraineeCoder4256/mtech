# Every trained model through every check (12 Nov 2026)

Phani asked for the checks the earlier models had missed.  Split with the
Innovation thread: Development ran gate2 (this folder, a5eb9ee);
Innovation ran anisocheck (runs/2026-11-12/aniso-catchup/) and the
problem race (runs/2026-11-12/race-catchup/, v2.1.8 785a033).  Earlier
results are quoted with their folder.

## Anisotropic balls (interface v2)

gate2 = 343 ball-level checks against the exact walk, final rule (a weight
check fails only if |z| > 3.29 AND the absorbed share is off by > 2%).
anisocheck = 20 whole problems against OpenMC.

| model | gate2 failing | weight \|z\| > 3.29 alone (of which >1% off) | us per ball, model / exact walk | anisocheck | where |
|---|---|---|---|---|---|
| nflow seed 0 | 0 | 11 (2) | 6.75 / 2.09 | 19/20 (chance cask mesh fail) | here; anisocheck-m2.1 |
| nflow seed 1 | 0 | 18 (5) | 6.57 / 2.07 | 20/20 | here; aniso-catchup |
| cfm seed 0 | 0 | 22 (5) | 26.65 / 2.05 | 20/20 | here; aniso-catchup |
| cfm seed 1 | 0 | 25 (2) | 25.01 / 2.03 | 20/20 | here; aniso-catchup |
| **cvae seed 0 (default)** | 0 | 21 | 1.75 / 1.35 | 20/20 | accept-m2.1 |
| cvae seed 1 | 0 | 27 (5) | 2.09 / 1.93 | 20/20 | here; aniso-catchup |
| cgan seed 0 | **10** | 75 (34) | 1.82 / 1.81 | **18/20** | cgan/gate; aniso-catchup |
| cgan seed 1 | **4** | 45 (16) | 1.97 / 1.95 | 20/20 | cgan/gate; cgan-innovation |

cgan seed 0's two OpenMC failures are real: sphere_p3_balls leakage z
+6.0 and absorption z -4.0, curved_backward_balls total flux z -3.4, the
same too-little-absorbed bias its gate shows.

## Isotropic balls (interface v1): race against OpenMC

One seed (11), 100k x 40 batches, R* = 3 (exact walk R* = 2), as on
5 Nov.  Worst |z| against OpenMC analog over all problems; seconds on the
four problems that use balls.  Timings are about 2x the 5 Nov race (this
machine and code since v1.6.5), so compare within this table only.

| contender | worst \|z\| (where) | sphere | cask | curved | tilted |
|---|---|---|---|---|---|
| OpenMC analog | 0 | 18.7 | 37.4 | 95.3 | 47.6 |
| ours, plain MC | 1.4 (slab leakage) | 72.4 | 129.5 | 393.7 | 165.0 |
| exact walk | 2.0 (cask total flux) | 35.9 | 115.7 | 352.6 | 176.9 |
| nflow seed 0 (default) | 1.5 (cask leakage) | 54.9 | 145.7 | 398.2 | 211.0 |
| nflow seed 1 | 3.5 (curved total flux) | 54.6 | 135.1 | 387.7 | 190.4 |
| cvae seed 0 | 2.5 (cask leakage) | 42.5 | 115.8 | 375.1 | 186.3 |
| cfm seed 1 | 2.6 (sphere leakage) | 159.0 | 321.8 | 558.8 | 288.2 |
| cgan seed 0 | **11.0** (sphere leakage 4.2% low) | 40.4 | 130.3 | 361.5 | 186.6 |
| cgan seed 1 | **12.1** (sphere leakage 4.5% low) | 41.0 | 126.7 | 361.5 | 200.6 |

From the 5 Nov race (3 seeds, runs/2026-11-05/race-models/): cfm seed 0
3.3 and 3.9 on two of three seeds (sphere leakage, a real bias), cvae seed 1 at most 2.3.  nflow seed 1's 3.5
is one seed on one tally, borderline rather than a clear fail.

## Reading

* Every flow, flow-matching and cVAE model passes every check; the
  accepted defaults stand.
* Both cGANs fail at both interfaces, in the same direction (too little
  absorbed, too much leakage at v2; leakage 4% low at v1, where short
  paths matter differently).
* No model beats the exact walk on time; cvae and cgan come closest.
