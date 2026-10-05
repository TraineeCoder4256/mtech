# gmc3d: ball-step GMC for any 3D CSG geometry

Status: **Stage 1 built** (any CSG geometry, one energy group, isotropic
scattering, fixed source). Results and the open research decisions are in
`runs/2026-10-05/STAGE1.md`.

The method: right after a scatter, a distance field gives the biggest
one-material ball around the particle. If the ball is big enough, one call
to a backend (the exact walk, a lookup table, or a trained network) moves
the particle to the ball's edge, replacing every scatter inside. Otherwise
ordinary Monte Carlo takes one flight. The geometry lives only in the
distance field, so one small model serves any shape.

The requirements, in five stages, are in the SRS document
("3D GMC: Software Requirements Specification"):

1. Any CSG geometry, one energy, isotropic scattering, fixed source.
2. + anisotropic scattering.
3. + multigroup energy.
4. + criticality (k-eff).
5. + continuous energy, inside OpenMC.

## Running it

From this folder, with the `gmc` environment's Python (it has OpenMC 0.16,
Numba and PyTorch):

    python run.py solve slab                    # plain Monte Carlo
    python run.py solve slab --backend table --r-star 3 --openmc
    python run.py validate                      # Stage 1 suite vs OpenMC
    python run.py table                         # build the lookup table
    python run.py train                         # train the network
    python run.py ballcheck                     # backends vs the exact walk
    python run.py measure                       # speed study
    python run.py race                          # OpenMC vs ours, head to head
    python run.py race --device cuda            # ... with the network on a GPU
    python -m checks.gpu_bench --device cuda    # GPU cost per draw only
    python -m pytest -q tests                   # unit tests

## File map

Read top to bottom: each file only uses files above it or beside it.

```
problems/            THE INPUTS: each file builds an openmc.Model
  common.py            one-group materials, sources, tallies (shared)
  sphere.py            homogeneous sphere / reflective cube (exact answers)
  slab.py              thick shield slab (the headline problem)
  lattice3d.py         gmc2d's checkerboard in 3D (balls can't help)
  curved.py            cylinder, sphere, cone and a void gap
  nested.py            3D lattice, rotated/translated universe
  cask.py              storage cask with a duct: a template for your own

core/                THE SOLVER
  surfaces.py          planes, spheres, cylinders, cones: value, ray
                       distance, nearest distance, normal
  geometry.py          which cell am I in; distance to the next surface;
                       the distance field (nearest surface, any level)
  openmc_import.py     openmc.Model -> the arrays above (refuses what
                       Stage 1 can't do, by name)
  rng.py               one random stream per particle (OpenMC's generator)
  tallies.py           flux and absorption scoring, mesh tracking, error bars
  transport.py         plain Monte Carlo + ball steps; the batched loop
  output.py            results to JSON; mesh tallies to VTK (ParaView)

ball/                THE BALL STEP
  walk.py              the exact walk inside a ball (the truth)
  oracle.py            backend 1: run the exact walk live
  table.py             backend 2: draw from pre-computed walks
  data.py              training data + the network's input/output encoding
  network.py           backend 3: a normalizing flow, and its training

checks/              IS IT RIGHT, AND IS IT FAST
  openmc_ref.py        run OpenMC on the same model, read its tallies
  compare.py           ours vs a reference, bin by bin (z-scores)
  analytic.py          problems with exact answers
  validate.py          the Stage 1 suite (all problems, OpenMC, exact balls)
  ball_metrics.py      one radius at a time: backend vs exact walk
  measure.py           cost per flight, big-ball share, cost per draw, R*
  race.py              OpenMC vs our plain MC vs ball runs, timed
  gpu_bench.py         the network's cost per draw on a GPU, and what it
                       would do to the race (needs no OpenMC)

tests/               pytest: surfaces, geometry vs OpenMC, distance-field
                     safety (1.2 x 10^6 balls), solver, backends
run.py               the command line above
runs/<date>/         logs and JSON of each run (kept in git; VTK files are
                     not, rerun `solve` to get them)
data/                generated: libraries, walks, table, networks, OpenMC
                     run folders (ignored by git; rebuilt on demand)
```

How one run flows:

```
problems/slab.py --openmc.Model--> core/openmc_import.load --Problem-->
core/transport.run --(per batch)--> particles fly (geometry, tallies, rng)
                   --(when a ball fits)--> ball backend .sample(R) -> (mu, s)
                   --> tallies.Results --> output.save_json / checks.compare
```

## Rules for this folder

All 3D work lives here, on branch `3d-gmc` (SRS NF-8). `gmc2d/` stays frozen
so the 2D results remain reproducible; code reused from it (the spline flow,
the training loop conventions, the Monte Carlo conventions) is copied and
adapted, not imported. Each file's docstring says what uses it and what it
uses, and explains the design so it can be presented.
