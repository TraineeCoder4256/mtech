# Run of 28 Sept 2026: the three reference runs under the new scoring

These are the same three models as `runs/2026-09-23/`, with the same cfm
checkpoint, scored again after the groundwork for more model families:

- 6 model solves per scale instead of 2;
- a calibrated per-cell χ²;
- the absorber-cell test as a standard metric.

Everything ran on one machine: 4 CPU cores, no GPU, with the three runs one
after another. The code is commit `c2c02a3`. The runs started before that
commit was made, so `provenance.json` says `git_dirty: true` against
`274dc58`. The code that ran is the committed code.

```
comparison.txt      python compare.py over the three runs (+ comparison.pdf)
cfm/, oracle/, free/  python run.py <name>; report.txt first, console.log is the screen output
checks/
  chi2_calibration.py / .txt  exact Monte Carlo scored as a model: the new χ² averages
                              ~1, the old one reached 10^5
  check_refactor.txt          cfm still trains and samples bit-identically after the
                              training loop moved to generators/training.py
```

## What the new scoring shows

| | scale 1 | scale 4 | scale 10 | scale 20 |
|---|---|---|---|---|
| oracle, error / floor | 1.22× | 0.83× | 1.01× | 0.97× |
| oracle, cell χ² | 0.82 | 1.79 | 0.79 | 0.39 |
| oracle, absorber model/MC | 0.97 | 0.99 | 1.00 | 0.98 |
| cfm, error / floor | 1.16× | 1.34× | 1.32× | 3.74× |
| cfm, cell χ² | 1.55 | **6.03** | **7.74** | **56.2** |
| cfm, absorber model/MC | 0.90 | 0.56 | 0.32 | 0.17 |
| cfm, total flux / reference | 0.996 | 0.991 | 0.984 | 0.955 |

- **The oracle now passes every test at every scale.** The oracle is exact,
  so it has to pass, and on 23 Sept the old χ² failed it (101 at scale 10).
  Its scale-1 error of 1.22× is within the noise of 6 solves: the 8-solve
  test on 23 Sept gave 1.10×.
- **cfm fails the χ² test from scale 4 upward.** The old test could not show
  this. It agrees with cfm's falling total flux and with the absorber test,
  so three independent measures now point at the same fault.
- **The absorber numbers reproduce the 23 Sept throwaway test** (0.93, 0.63,
  0.31, 0.17) at 2.5× the sample size.
- **cfm speed:** 73.0 µs per crossing, against 61.3 µs on 23 Sept, on the
  same machine type with the same model. That makes it 370× slower than MC,
  not 296×. Only the timing changed; the model did not. Speed numbers on this
  shared machine vary by about 20% from run to run.
