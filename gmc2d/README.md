# Generative Monte Carlo for 2-D particle transport

Reproduction of **arXiv:2512.13965v1**, *Generative Monte Carlo Sampling for
Constant-Cost Particle Transport* (Farmer, Murray, Krotz, McClarren).

**The idea.** Inside a scattering cell, replace the whole random walk with one
draw from a conditional generative model. Monte Carlo cost per cell grows with
optical thickness — thicker cell, more scattering events. The sampler costs a
fixed number of network evaluations whatever the thickness. Above some
crossover, the sampler should win.

**Whether it wins here is what `run.py` measures, for any model, including
when the answer is no.**

**The benchmark** is Farmer et al.'s steady-state variant of the ORNL lattice
geometry (arXiv:2505.17284), not the ORNL benchmark itself: ORNL's is
time-dependent (final time 3.2) with purely absorbing blocks (σa = 10,
σs = 0); this one is steady-state with σs = 0.5, σa = 9.5. ORNL's published
solutions therefore cannot validate this code; OpenMC does instead.

## Run it

```bash
python validate.py       # ~3 min   -> checks the MC baseline against exact results
python make_data.py      # ~2 s     -> data/cells.npz (65 MiB, 2.88M rows)
python run.py cfm        # ~50 min training the first time, then ~10 min
                         #          -> models/cfm/ + results/cfm/<timestamp>/
python run.py oracle     # the accuracy goalpost (exact MC walks)
python run.py free       # the cost goalpost (a draw that costs nothing)
python compare.py        # -> results/comparison.txt, comparison.pdf
```

Settings are named constants at the top of each file; the only argument is
the model's name (and `train` to force retraining). A GPU is used
automatically if `torch.cuda.is_available()`.

Needs `numpy numba matplotlib torch`; `openmc_lattice.py` also needs OpenMC
(`conda install -c conda-forge openmc`, it is not on PyPI).

The older `train.py` + `evaluate.py` pair still works and gives the same
cell-physics numbers, but scores against a fresh, noisy Monte Carlo run each
time. `run.py` supersedes it.

## What it does

**`make_data.py`** — a particle enters a rectangular cell of optical size
`W × H` through the left face at height `xi` in direction `Omega_in`; Monte
Carlo walks it to its exit; record where it left, which way it was going, how
far it went.

No geometry is involved. A cell is fully described by its optical size, so `W`
and `H` are drawn directly — **both, independently**, matching the paper's
conditioning on `(W, H, xi, Omega_in)`. Square cells are just `W = H`, and the
lattice used at the end is one instance of that. Sizes are drawn *continuously*
log-uniform rather than from a grid, so evaluation conditions are genuinely
unseen.

**`train.py`** — conditional flow matching. Two preprocessing rules do real
work: uncollided particles are dropped (their exit is an exact function of
their entry — a Dirac delta a smooth flow cannot represent, handled
analytically at sampling time instead), and the train/validation split is by
entry condition rather than by row, so validation cannot be satisfied by
memorising a sibling sample.

**`evaluate.py`** — three questions:

1. **Does it learn the cell physics?** Exit distributions vs MC for several
   shapes *including rectangles*, at unseen entry conditions. A table, because
   the content is a handful of numbers.
2. **Does it solve a real problem?** Chain the sampler across the 7×7 lattice.
   Two MC runs with different seeds give the noise floor — the best score
   anything could get. → `figures/accuracy.pdf`
3. **Is it faster?** The same lattice with cross sections scaled up. →
   `figures/speed.pdf`

Everything numeric goes to `results/evaluation.txt`: time per scattering event,
time per network evaluation, time per cell crossing each way, how GMC's wall
time splits between the network and the host loop, and the break-even
arithmetic — how many scattering events one GMC crossing costs, versus how many
the geometry actually has. **That subtraction is the answer to "why is it not
faster", and it says which of two problems you have:** if the network dominates
the time, the model is too expensive (fewer steps, cheaper solver,
distillation); if the host loop dominates, the model is not the bottleneck and
no model work will fix it.

## Plugging in a model

A model is a class in `generators/` with four methods and one number:

```
fit(data, out_dir, time_cap)   train on the shared split; return facts
sample(c, generator, cond)     standardised conditions (n, 5) -> standardised
                               exit states (n, 6)
save(out_dir) / load(out_dir)  checkpoint in models/<name>/
nfe                            network evaluations per sample (its cost class)
```

Register it in `generators/__init__.py` and run `python run.py <name>`.
Everything physical stays outside the model, in `sampler.CellSampler`: the
analytic uncollided branch, decoding to a position and direction, the
clamp to the straight-line path, and the cost counters. So every model is
judged on exactly its job, the collided exit distribution, and none can win
or lose on how it handles the parts that are the same for all.

Two reference "models" bracket every real one:

- **oracle** runs real Monte Carlo walks behind the same interface. It is the
  best accuracy any model could reach through this pipeline.
- **free** returns its input noise. It costs nothing, so its time is the
  pipeline's own overhead, the floor no model can go below.

## The frozen reference

`reference/lattice_x{1,4,10,20}.npz` hold the answer every model is scored
against: `mc.py` at 40 million histories per scale, OpenMC's solution where
it exists, and six independent 20,000-particle Monte Carlo solves that give
the noise floor. `reference/manifest.json` records how they were made. Each
file stores a fingerprint of `mc.lattice()` and refuses to load if the
problem has changed. Rebuild with `python make_reference.py` (~25 min).

Before this, `evaluate.py` re-ran Monte Carlo for each evaluation, so each
model got its own noisy reference. The floor is also defined differently:
`evaluate.py` compared two noisy MC runs with each other; `run.py` compares
one MC run with a near-exact reference, which is smaller by about √2. Model
errors are measured the same way, so the ratio to the floor is what carries
over between the two.

## Files

```
mc.py              Monte Carlo: the single-cell walk, the full-domain solver,
                   and the lattice test geometry
validate.py        checks mc.py against three exact analytic identities
data.py            encodings, normalisation, leak-free dataset split
model.py           the velocity field, the CFM loss, weight averaging
generators/        one file per model: base.py (the interface), cfm.py,
                   oracle.py, free.py
sampler.py         CellSampler: the physics around any model's draw
solve.py           chain the sampler across a mesh
make_data.py       the training set
benchmark.py       the protocol (scales, particle counts, seeds) and the
                   reference loader
make_reference.py  builds reference/
metrics.py         cell-level (marginal and joint) and lattice-level scores
run.py             train if needed, then score one model
compare.py         all scored models in one table
openmc_lattice.py  the lattice in OpenMC, as an independent check
check_benchmark.py the two lattice definitions and the reference agree
check_refactor.py  the Generator split changed no number (bit-identical)
train.py, evaluate.py   the original pipeline, kept for comparison
```

Nothing in `mc.py` imports the model, and `mc.py` and `solve.py` are
unchanged by the pipeline.

## Validating the baseline

Everything is measured against the Monte Carlo solver, so it is checked
against results that are known **exactly** — not against another code. A
cross-code comparison shows two codes agree; an analytic identity shows the
code is right.

| test | exact result | what it exercises |
|---|---|---|
| Dirac mean-chord invariance | mean total path = `2WH/(W+H)`, independent of scattering | free-path sampling, isotropic scattering, boundary detection, path accumulation |
| uncollided fraction | `P(k=0) = exp(-chord)` | the exponential free-path law and chord geometry |
| particle balance | absorbed / source = 1 in a thick pure absorber | mesh traversal, track-length estimator, implicit capture, tally normalisation |

`python validate.py` reports each as `z = (measured - exact) / standard error`;
`|z| < 3` passes. All twelve cases currently pass, with relative errors below
0.5%, down to an uncollided probability of 8.6e-4 (about 7 mean free paths).

One subtlety worth knowing: in test 1 the standard error is computed over
independent **entry conditions**, not over particles. Particles sharing a
condition are correlated, and treating them as independent understates the
error by roughly 3x — enough to turn a passing case into a false failure.

What these tests do **not** cover: they confirm the solver conserves
particles and gets single-cell statistics exactly right, but conservation
does not prove the flux is in the right *place*. That is closed by OpenMC:
`openmc_lattice.py` solves the same lattice independently, and at 40 million
histories each the two agree cell by cell to within statistics (χ² per cell
0.80, median difference 0.10%, absorption 0.966594 ± 0.000027 against
0.966569 ± 0.000171). Flipping OpenMC's field upside down gives χ² ≈ 10⁵,
so the test can see an error. `check_benchmark.py` repeats the comparison
from the stored arrays.

## Honest notes

- **The MC baseline is numba-compiled and multi-threaded; the GMC host loop is
  single-threaded NumPy.** A GPU number for GMC compared against CPU MC measures
  hardware, not algorithm. Report the hardware on both sides.
- **The lattice cells are optically thin** (0.5–1.0 mfp at the published
  scale). Most particles cross without scattering at all, so there is nothing
  for a sampler to save. That is why `SCALES` exists, and why the honest
  headline is a crossover, not a speedup.
- **Only the boundary model.** A particle born inside a cell cannot be handled
  by a sampler conditioned on entry through a face, so the source cell falls
  back to analog MC — one cell out of 49, timed separately in the report.
- **No distillation.** The paper's 1–2 evaluation regime is not reproduced.
