# Run of 29 Sept 2026: the normalizing flow, first training seed

The second model family, `generators/nflow.py` (a neural spline flow), trained
once at seed 0 and scored by `run.py` exactly as cfm, oracle and free were on
28 Sept. Everything ran on one machine: 4 CPU cores, no GPU. The code is
commit `505a974`.

```
comparison.txt      python compare.py: nflow beside the 28 Sept cfm, oracle, free (+ .pdf)
nflow/              python run.py nflow; report.txt first, console.log is the screen
                    output, train_log.txt and loss.png are the training curve
checks/
  test_nflow.py     correctness of the flow before training: identity at start,
                    exact inverse, log-density against autograd, 1D density sums to 1
  trials/           the three tuning trials (trial.py and their logs); see the
                    docstring of generators/nflow.py for the table
```

Training: 20,000 steps in 122.6 min, inside the 3 h cap. The validation NLL
was still falling at the last step (0.84 at step 19,500, 0.76 at 20,000).

## nflow against cfm

| | scale 1 | scale 4 | scale 10 | scale 20 |
|---|---|---|---|---|
| cfm, error / floor | 1.16× | 1.34× | 1.32× | 3.74× |
| nflow, error / floor | 1.55× | 1.24× | 3.31× | 3.78× |
| cfm, cell χ² | 1.55 | 6.03 | 7.74 | 56.2 |
| nflow, cell χ² | 2.62 | 4.16 | 22.3 | 41.5 |
| cfm, absorber model/MC | 0.90 | 0.56 | 0.32 | 0.17 |
| nflow, absorber model/MC | 1.08 | 0.98 | 0.90 | 0.78 |
| cfm, total flux / reference | 0.996 | 0.991 | 0.984 | 0.955 |
| nflow, total flux / reference | 1.012 | 0.992 | 0.961 | 0.955 |

| cost at scale 1 | cfm | nflow |
|---|---|---|
| network evaluations per sample | 10 | 1 |
| time per macro-cell crossing | 73.0 µs | 21.0 µs |
| speed against MC | 370× slower | 79× slower |
| path lengths clamped | 5.38% | 0.62% |

- **The flow is 3.5 times faster than cfm** at the same size, because one
  draw is one pass through the network instead of ten. It is still 79 times
  slower than Monte Carlo on this lattice.
- **It keeps the absorber tail far better**: 0.78 of the right transmission
  at scale 20, against cfm's 0.17. It is trained to give every real exit a
  high probability, and the short paths are real exits, so it cannot ignore
  them the way a sample-quality objective can.
- **But its lattice accuracy is not better.** It is worse at scale 1 (bias
  1.04% against cfm's 0.39%, and 1.2% too much flux) and at scale 10, and
  equal at scales 4 and 20. Both families lose 4.5% of the flux at scale 20,
  even though the flow's absorber number is much better. So the scale-20 loss
  is not only the absorber tail. This is an observation, not yet a diagnosis.
- One training seed is not a ranking. Seed 1 is training now (fairness rule,
  `generators/base.py`).
