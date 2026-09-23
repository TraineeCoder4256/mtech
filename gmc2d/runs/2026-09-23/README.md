# Run of 23 Sept 2026: the first pipeline results

The pipeline's first scored runs, kept in the repository so the numbers in
`FINDINGS.md` can be checked without re-running anything. Everything ran on
one machine, 4 CPU cores and no GPU, at commit `93c437b`.

```
FINDINGS.md       what was found, with the raw numbers and the decisions still open
comparison.txt    python compare.py over the three runs below (+ comparison.pdf)
cfm/              python run.py cfm     (+ train_info.json, train_log.txt, loss.png:
                  trained with the unmodified train.py of 81d8340)
oracle/           python run.py oracle
free/             python run.py free
checks/
  evaluate_py_same_checkpoint.txt   evaluate.py on the same checkpoint: the
                                    parity check against cfm/report.txt
  check_benchmark.txt               python check_benchmark.py
  make_reference.log                how reference/ was built
  openmc_results_x1.txt, _x10.txt   the OpenMC runs in the reference
  oracle_noise.py / .txt            8 oracle solves: why 2 GMC solves per scale
                                    is too few to rank models
  absorber_transmission.py / .txt   cfm vs MC surviving weight through one
                                    absorber cell, by scale
```

Each run directory holds `report.txt` (read this), `metrics.json` (every
number), `provenance.json` (commit, hardware, versions, hashes), `fields.npz`
(the GMC flux fields) and three figures.

The two check scripts run from `gmc2d/`, like this:
`PYTHONPATH=. python runs/2026-09-23/checks/oracle_noise.py`.
Re-running them reproduced these outputs exactly.
