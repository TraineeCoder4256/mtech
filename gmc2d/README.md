# Generative Monte Carlo for 2-D particle transport

Reproduction of **arXiv:2512.13965v1**, *Generative Monte Carlo Sampling for
Constant-Cost Particle Transport* (Farmer, Murray, Krotz, McClarren).

**The idea.** Inside a scattering cell, replace the whole random walk with one
draw from a conditional generative model. Monte Carlo cost per cell grows with
optical thickness — thicker cell, more scattering events. The sampler costs a
fixed number of network evaluations whatever the thickness. Above some
crossover, the sampler should win.

**Whether it wins here is what `evaluate.py` measures, including when the
answer is no.**

## Run it

```bash
python validate.py       # ~3 min   -> checks the MC baseline against exact results
python make_data.py      # ~2 s     -> data/cells.npz (65 MiB, 2.88M rows)
python train.py          # ~40 min  -> models/
python evaluate.py       # ~10 min  -> figures/ + results/
```

Two more, once there is a trained model, for the speed question specifically:

```bash
python speed_profile.py    # ~8 min    -> where GMC's time goes, and the crossover
python speed_prototype.py  # ~20 min   -> whether the candidate fixes recover it
```

No flags. Settings are named constants at the top of each file. A GPU is used
automatically if `torch.cuda.is_available()`.

Needs `numpy numba matplotlib torch`.

**[`SPEED.md`](SPEED.md) reads those two reports and says what to do.** One
full run of everything above is committed under
[`runs/2026-09-21/`](runs/2026-09-21/), so the numbers can be checked without
re-running the pipeline.

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

## Files

Eight files, flat, no packages.

```
mc.py           Monte Carlo: the single-cell walk, the full-domain solver,
                and the lattice test geometry
validate.py     checks mc.py against three exact analytic identities
data.py         encodings, normalisation, leak-free dataset split
model.py        the velocity field, the CFM loss, weight averaging
sampler.py      cell geometry, ODE solve, exit-state decoding
solve.py        chain the sampler across a mesh
make_data.py    step 1
train.py        step 2
evaluate.py     step 3
```

The dependency graph is a line: `make_data` → `mc`; `train` → `data`, `model`;
`evaluate` → everything. Nothing in `mc.py` imports the model.

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
does not prove the flux is in the right *place*. Cross-validating the flux
field against an independent solution — such as the ORNL reference — remains
open.

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
