# Generative Monte Carlo for particle transport

M.Tech project: a comparative study of generative surrogate models for
accelerating Monte Carlo particle transport, reproducing and extending
**arXiv:2512.13965v1** (Farmer, Murray, Krotz & McClarren) on the Lattice
benchmark of **arXiv:2505.17284** (Schotthöfer & Hauck, ORNL).

## Everything lives in [`gmc2d/`](gmc2d/)

```bash
cd gmc2d
python make_data.py         # single-cell training set
python train.py             # conditional flow-matching model
python evaluate.py          # accuracy and cost against the Monte Carlo baseline
```

See [`gmc2d/README.md`](gmc2d/README.md) for what each file does and how the
pieces fit together.

Two checks sit outside that three-command pipeline:

```bash
python validate.py          # the MC baseline against three exact identities
python openmc_lattice.py    # the same lattice solved by OpenMC, differenced
                            # cell by cell (needs conda install -c conda-forge openmc)
```

## History

An earlier version of this work lived in `mc2d/`, `gmc/`, `scripts/` and
`docs/` at the repository root. It was superseded by `gmc2d/`, which is a
smaller, flat rewrite with the same physics, and has been removed. The old
tree remains in the git history if it is ever needed.
