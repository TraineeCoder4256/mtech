# Generative Monte Carlo for particle transport

M.Tech project: a comparative study of generative surrogate models for
accelerating Monte Carlo particle transport, reproducing and extending
**arXiv:2512.13965v1** (Farmer, Murray, Krotz & McClarren) on their
steady-state variant of the Lattice geometry of **arXiv:2505.17284**
(Schotthöfer & Hauck, ORNL). The variant differs from ORNL's benchmark: it is
steady-state rather than time-dependent, and its absorbers scatter a little
(σs = 0.5, σa = 9.5) instead of not at all.

## Everything lives in [`gmc2d/`](gmc2d/)

```bash
cd gmc2d
python make_data.py         # single-cell training set
python run.py cfm           # train (if needed) and score one model
python compare.py           # every scored model side by side
```

See [`gmc2d/README.md`](gmc2d/README.md) for what each file does and how the
pieces fit together.

Three checks sit outside that pipeline:

```bash
python validate.py          # the MC baseline against three exact identities
python openmc_lattice.py    # the same lattice solved by OpenMC
                            # (needs conda install -c conda-forge openmc)
python check_benchmark.py   # both lattice definitions agree, and the frozen
                            # reference agrees with OpenMC
```

## History

An earlier version of this work lived in `mc2d/`, `gmc/`, `scripts/` and
`docs/` at the repository root. It was superseded by `gmc2d/`, which is a
smaller, flat rewrite with the same physics, and has been removed. The old
tree remains in the git history if it is ever needed.
