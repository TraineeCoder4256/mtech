# Every model family on the reference problems, 5 Nov 2026 (Innovation)

**What was run**
- `race_models.py SEED` for seeds 11, 12 and 13 on 3d-gmc at m1.2 (4548005). It is checks/race with every family as a contender.
- All 9 problems, 4,000,000 histories per contender, 4 CPU cores, one session.
- Medians come from `medians.py` and are in `medians.{log,json}`.
- `superseded/` holds a first start that was stopped after the sphere, because cfm and cVAE seed 1 sat at R* = 10 and got no balls.

**Accuracy** is the worst |z| against OpenMC analog over leakage, total flux and (on the cask) detector flux, for each seed.
- cVAE seeds 0 and 1: at most 2.5, at most 2.3.
- flow: at most 1.9.
- exact walk: at most 2.8. table: at most 3.2 (tilted, one seed).
- cfm: sphere 3.3 and 3.9 on two of three seeds. That is a real bias in sphere leakage.

**Speed:** median seconds where the balls are used. On the other 5 problems the models equal plain MC.

| problem (balls/h) | exact walk | flow | cVAE | cVAE s1 | cfm | plain MC | OpenMC analog |
|---|---|---|---|---|---|---|---|
| sphere (1.00) | 20.3 | 36.9 | 25.6 | 24.3 | 118.7 | 37.9 | 13.4 |
| curved (2.16) | 167.6 | 213.4 | 193.2 | 192.7 | 358.2 | 206.9 | 60.2 |
| cask (1.80) | 53.0 | 81.1 | 64.1 | 63.6 | 208.6 | 65.3 | 28.8 |
| tilted (1.10) | 85.8 | 107.4 | 98.4 | 98.9 | 181.8 | 90.8 | 34.8 |

**Reading the results**
- The cVAE costs 8-31% less time than the flow, and 18-26% more than the exact walk.
- It beats plain MC on the sphere, curved and cask, and loses on tilted.
- Seed-to-seed spread in seconds is 1-11%.
