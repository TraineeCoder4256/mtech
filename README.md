# Generative Monte Carlo for 2-D particle transport

A reproduction of **arXiv:2512.13965v1**, *Generative Monte Carlo Sampling for
Constant-Cost Particle Transport* (Farmer, Murray, Krotz, McClarren).

**The idea.** Inside an optically thick, materially uniform cell, replace the
entire in-cell scattering random walk with one draw from a conditional
generative model. Monte Carlo cost per cell grows with optical thickness —
thicker cell, more scattering events to simulate. The generative sampler costs
a fixed number of network evaluations no matter how thick the cell is. Above
some crossover thickness, the sampler should win.

**Whether it actually wins here is the question `scripts/evaluate.py` answers,
in numbers, including when the answer is no.**

---

## The whole pipeline is three commands

```bash
python scripts/make_data.py      # ~11 s    -> data/singlecell.npz
python scripts/train.py          # ~40 min  -> models/boundary/
python scripts/evaluate.py       # ~10 min  -> figures/ + results/
```

Add `--device cuda` to the last two for a GPU. Timings are for 4 CPU cores.

That is the entire main path. Everything else in the repo is either the Monte
Carlo baseline those three depend on, or the paper's own baseline figures kept
aside in `scripts/baseline/`.

---

## What each step does

### 1. `scripts/make_data.py` — the training data

One physical experiment, repeated: a particle enters a square purely-scattering
cell of optical width `W` through the left face at height `xi` in direction
`Omega_in`; Monte Carlo walks it until it leaves; record where it left (`p`),
which way it was going (`Omega_out`) and how far it travelled (`s`).

No larger geometry is involved. A cell is completely described by its optical
width `W = pitch × sigma_s`, so `W` is sampled directly on a log grid from 0.025
to 20 mfp rather than derived from any particular problem's materials.

Conditions are sampled **uniformly**, not physically: the model must be accurate
everywhere it will be asked, and at solve time it gets asked wherever the
geometry happens to send particles.

```bash
python scripts/make_data.py                                   # 3.1 M rows
python scripts/make_data.py --n-cond 512 --per-cond 32        # smaller
```

### 2. `scripts/train.py` — the model

Conditional flow matching. Take a training exit state `y` and Gaussian noise
`z`, pick a time `t`, place a point on the straight line between them
(`x_t = (1-t)z + ty`), and ask the network what velocity carries a particle
along that line. The answer is `y - z`. Regress on it. To sample, integrate the
learned field from noise at `t=0` to `t=1`. `gmc/cfm.py` is nine lines.

Two preprocessing rules carry real weight:

- **Uncollided particles (`k == 0`) are dropped.** Their exit is an exact
  function of their entry — a Dirac delta, which a smooth flow cannot
  represent. They are sampled analytically at solve time instead.
- **The train/validation split is by entry condition, never by row.** Each
  condition has ~48 sampled exits; splitting by row would put siblings of a
  training row into validation and report memorisation as generalisation.

```bash
python scripts/train.py --device cuda --batch 16384 --lr 3e-3
```

### 3. `scripts/evaluate.py` — does it work, and is it faster

One geometry throughout: the 7×7 cm lattice from `mc2d.problems`, a checkerboard
of absorbing blocks in a scattering background with an isotropic source in the
middle cell.

**Accuracy** — solve it twice with Monte Carlo (different seeds) and once with
the sampler. The two MC runs differ only by seed, so the gap between them is
pure statistical noise: that is the floor, and the only honest yardstick.
Beating it is impossible; approaching it is the goal. Everything is compared on
the 7×7 macro-cell grid, because the sampler returns total path length in a cell
but not where inside it went — its flux is inherently cell-averaged, and
comparing against the fine 112×112 mesh would compare two different quantities.

**Speed** — the same geometry with every cross section multiplied by a scale
factor, which makes cells optically thicker without changing the layout.

```bash
python scripts/evaluate.py --device cuda --n 50000 --scales 1 4 10 20
```

Three outputs:

| file | what it holds |
|---|---|
| `figures/geometry.pdf` | the problem itself: materials, source, the optical width of every cell, and the flux on both the fine mesh and the macro cells |
| `figures/accuracy.pdf` | MC and GMC flux fields, the error map, and a lineout with the noise floor drawn in |
| `figures/speed.pdf` | wall time vs optical thickness, the speedup curve, and where GMC's time actually goes |
| `results/evaluation.txt` | every raw number behind both figures |

**Look at `figures/geometry.pdf` first** — it shows the optical width of every
cell, and that one number decides whether the method can possibly pay off. In
this lattice at its published scale the cells are 0.5–1.0 mfp: a particle
crosses most of them without scattering even once. There is nothing there for a
generative sampler to save. That is why `--scales` exists.

**`results/evaluation.txt` is the one to read.** It has the numbers a plot
cannot show: time per scattering event, time per network evaluation, time per
cell crossing each way, how GMC's wall time splits between the network and the
NumPy host loop, and the break-even arithmetic — how many scattering events one
GMC cell crossing costs, versus how many the geometry actually has. That
subtraction is what tells you *why* the speedup is what it is, and which of the
two possible problems you have:

- the **network** dominates GMC's time → the model is too expensive (fewer ODE
  steps, a cheaper solver, distillation);
- the **host loop** dominates → the model is not the bottleneck at all, and no
  amount of model work will fix it.

---

## Layout

```
mc2d/            the Monte Carlo baseline (numba, multi-threaded)
  transport.py     full-domain solver, track-length estimator
  problems.py      the lattice and hohlraum benchmarks
  singlecell.py    the in-cell walk that generates training data
gmc/             the learned sampler
  data.py          encodings, normalisation, leak-free split
  model.py         the velocity field v(x, t, c)
  cfm.py           the flow-matching loss + EMA
  geometry.py      cell geometry, shared by encoder and decoder
  sampler.py       ODE integration, decoding, uncollided branch
  transport.py     chains the sampler across a mesh
  device.py        cpu / cuda / mps
scripts/
  make_data.py     step 1
  train.py         step 2
  evaluate.py      step 3
  baseline/        the paper's own MC figures (Fig 2b, 3, 4a, 4b)
```

---

## Honest notes

- **The MC baseline is numba-compiled and multi-threaded; the GMC host loop is
  single-threaded NumPy.** A GPU number for GMC compared against CPU MC measures
  hardware, not algorithm. If you report one, name the hardware on both sides,
  and report the CPU-vs-CPU comparison too.
- **`W` is not held out.** The validation split holds out entry *conditions*, not
  optical sizes, so generalisation to unseen `W` is untested.
- **Cells are square** (`H = W`). The conditioning vector has a separate `H` slot
  and the code paths exist, but no rectangular data does.
- **Only the boundary model is implemented.** A particle born inside a cell
  cannot be handled by a sampler conditioned on entry through a face, so the
  source cell falls back to analog MC. It is one cell out of 49 and is timed
  separately in the report.
- **No distillation.** The paper's 1–2 NFE regime is not reproduced.
- `docs/` and `scripts/build_paper.py` predate this simplification and describe
  the earlier, more complicated pipeline. Treat them as history.
