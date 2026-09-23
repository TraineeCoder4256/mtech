# The model pipeline: built, checked, and its first results (23 Sept 2026)

Branch `claude/project-thread-uy5x0a`, commits `343763b` to `93c437b`, pushed.
No PR. Everything ran on this container's machine: 4 CPU cores, no GPU.

## 1. What now exists

| command | what it does |
|---|---|
| `python run.py <model>` | trains the model if it has no checkpoint, then scores it end to end. Writes `results/<model>/<timestamp>/`: report.txt, metrics.json, provenance.json, fields.npz and three figures |
| `python compare.py` | puts the latest run of every model in one table, `results/comparison.txt`, plus an accuracy-against-cost plot |
| `python check_benchmark.py` | shows that OpenMC and `mc.py` define the same problem, that the frozen reference is intact, and that the two codes agree |
| `python make_reference.py` | rebuilds `reference/` (about 20 min on an idle machine) |

Three models are registered. **cfm** is the flow-matching model. **oracle** runs
real Monte Carlo walks behind the model interface, which makes it the best
accuracy any model could reach. **free** returns random noise at no cost, so
its time is the pipeline's own overhead.

## 2. The checks: did the new code change anything?

| check | result |
|---|---|
| Refactor, compared against the code at `81d8340` (`check_refactor.py`) | **bit-identical**: samples, the lattice solve, and 60 training steps. Re-run after all the changes: still identical |
| `run.py cfm` against `evaluate.py`, same checkpoint | cell table identical to the last digit. GMC counters identical: 166,480 crossings, 400 calls, 83,520 to the network, 835,200 network evaluations, 5.34% clamped. GMC-vs-GMC 1.508% in both, so the two lattice fields are the same |
| Definition check against injected errors | caught all three: a changed absorber σa, a moved absorber, a boundary changed to reflective |
| mc.py against OpenMC, from the stored arrays | scale 1: χ² per cell 1.16, absorption z +0.83. Scale 10: χ² 1.25, max \|z\| 3.67, absorption z −0.73. Flipped controls: 67,833 and 22,131 |

The OpenMC collision estimator (the fix for the non-independent cross-check)
now differs from the flux map by 0.025% at scale 1 (z = −0.38) and by 0.095% at
scale 10 (z = −1.39). Both are within statistics, and the check is now a real
one.

Scale 10 shows one cell at |z| = 3.67 among 49. That cell's OpenMC run had
only 4 million histories and 20 batches, so its z follows a t-distribution with
19 degrees of freedom. The chance of a cell that extreme among 49 is about 8%.
I am treating that as noise and not as a disagreement. The limit: scales 4 and
20 have no OpenMC run yet.

## 3. Results

Accuracy is the lattice error divided by the Monte Carlo noise floor, at
20,000 particles. A value of 1.00 means as good as Monte Carlo.

| model | scale 1 | scale 4 | scale 10 | scale 20 | total flux / reference (1, 4, 10, 20) | µs per crossing | speed vs MC, scale 1 |
|---|---|---|---|---|---|---|---|
| oracle | 1.52× | 1.16× | 0.74× | 1.03× | 1.005, 1.005, 1.003, 1.009 | 1.24 | 0.15× |
| cfm | 1.05× | 1.36× | 1.47× | **3.54×** | 1.001, **0.994, 0.983, 0.955** | 61.3 | 0.0034× (296× slower) |
| free | 33× | 24× | 18× | 40× | meaningless by design | 1.90 | 0.10× |

**cfm's cell physics.** The marginal distances match `evaluate.py`. The new
joint metrics are:
- Sliced W1 is **2.2× the MC floor**; the oracle scores 1.00.
- A classifier tells cfm's samples from MC's with 51.3% accuracy, with z from
  +2.5 to +5.0 across the five cell shapes. Between two MC runs it scores
  50.0%, with |z| ≤ 1.8.

So the model is close to MC, but it can be told apart from it.

## 4. The new finding: cfm under-transmits through absorbers, and this grows with thickness

`evaluate.py` only ever scored accuracy at scale 1, where cfm looks as good as
Monte Carlo. The pipeline scores every scale, and cfm's total flux falls as the
lattice thickens: 0.994, then 0.983, then 0.955. The oracle stays at 1.00 on
the same loop, so the fault is in the model and not in `solve.py`.

**Throwaway test to locate it.** One absorber cell, with an entry straight in
and an entry at an angle, 40,000 particles each. It compares the weight a
particle keeps crossing the cell, `mean(exp(−19·s))`, where 19 is σa/σs.

| scale | W = H | MC | cfm | cfm / MC | z |
|---|---|---|---|---|---|
| 1 | 0.5 | 7.84e-3 | 7.31e-3 | 0.93 | −1.3 |
| 4 | 2 | 7.17e-3 | 4.55e-3 | **0.63** | −7.5 |
| 10 | 5 | 7.58e-3 | 2.38e-3 | **0.31** | −15.7 |
| 20 | 10 | 8.19e-3 | 1.36e-3 | **0.17** | −20.5 |

(The angled entry gives 0.96, 0.65, 0.32 and 0.18.)

**Diagnosis.** In an absorber, almost all the surviving weight comes from the
**shortest paths**: particles that scatter near the face and come straight back
out. `exp(−19·s)` makes that tail of the path-length distribution decide
everything. cfm reproduces the bulk of the distribution well, but not that
tail, and the tail matters more the thicker the cell.

Two things support this reading:
- **log s** is already the worst column in the marginal table, at 0.05–0.06
  against a floor of 0.01–0.02.
- 5.34% of samples are **clamped** to the straight-line minimum, and
  clamping happens exactly in that tail.

This is a diagnosis, not a fix. The training range is not the cause: the cell
sizes the sampler sees are 0.5–20, and the model was trained on 0.05–20.

**Why it matters for the thesis.** GMC can only win at thick scales, and
thick scales are exactly where cfm's accuracy fails. The speed problem and this
accuracy problem pull in opposite directions.

## 5. Two weaknesses in my own metrics, found by the oracle

This is what the oracle is for: it is exact, so anything it fails is the
metric's fault.

1. **Two GMC solves is too few.** The oracle scores 1.52×, 1.16×, 0.74× and
   1.03× when it should score about 1.0 everywhere.
   - A separate test with 8 oracle solves at scale 1 gave individual errors
     from 0.77% to 1.66%, with a mean of 1.08% against the MC floor of 0.98%.
     That is 1.10×, and the mean field was consistent with pure noise.
   - With 2 solves, the ratio is uncertain by about ±30%. `evaluate.py` had the
     same weakness.
2. **The per-cell χ² is not calibrated at thick scales.** The oracle scores
   101 at scale 10, which should be about 1.
   - Dark cells have near-zero spread across the 6 floor runs, which makes
     their z values explode.
   - Until this is fixed, read the **total flux** and **bias against noise
     only** lines for systematic error, not χ².

## 6. Decisions for you

1. **More GMC solves per scale: 2 → 6** (recommended). Seeds 8 and 78 stay
   first, so parity with evaluate.py still holds. It adds about 6 minutes to a
   cfm run.
2. **Fix the χ² metric** (recommended): score only cells that the floor runs
   resolve to better than 20%, and use the GMC runs' own spread.
3. **Retire evaluate.py?** Parity is shown above. I recommend keeping it until
   you have looked at a `run.py` report yourself.
4. **The absorber finding.** Which direction to take is a discussion for us,
   not something I will start alone. Options include weighting training
   towards short paths, or treating absorption separately from the sampled
   walk.

Not done yet: OpenMC at scales 4 and 20. That is about 2.5 and 15 minutes of
OpenMC and needs nothing new, only `python openmc_lattice.py 4`, then
`python make_reference.py`.
