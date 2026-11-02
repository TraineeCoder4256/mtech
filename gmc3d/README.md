# gmc3d: ball-step GMC for any 3D CSG geometry

Version: see `VERSION`; what each version changed is in `CHANGELOG.md`;
what is still left out (and whether it is refused or silently ignored) is
in `GAPS.md`, updated with every version; the remaining version 1 work is
in `V1_PLAN.md`.

Status: **Stage 1 built** (any CSG geometry, one energy group, isotropic
scattering, fixed source). Results and the open research decisions are in
`runs/2026-10-05/STAGE1.md`. Since v1.1, mesh tallies (flux maps) are right
inside balls too, and surface current tallies exist; see
`runs/2026-10-27/MESH.md`.

The method: right after a scatter, a distance field gives the biggest
one-material ball around the particle. If the ball is big enough, one call
to a backend (the exact walk, a lookup table, or a trained network) moves
the particle to the ball's edge, replacing every scatter inside. Otherwise
ordinary Monte Carlo takes one flight. The geometry lives only in the
distance field, so one small model serves any shape.

The speed study and the ranked list of GPU optimisations are in
`runs/2026-10-06/SPEED.md`.

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
    python run.py meshcheck                     # mesh profile vs exact walks
    python run.py maps                          # flux maps vs OpenMC (PNG)
    python run.py measure                       # speed study
    python run.py race                          # OpenMC vs ours, head to head
    python run.py race --device cuda            # ... with the network on a GPU
    python -m checks.gpu_bench --device cuda    # GPU cost per draw only
    python -m pytest -q tests                   # unit tests
    python run.py check my_model.py             # what gmc3d will use, refuse
                                                # and warn about (no run)
    python run.py robust                        # 30 random models vs OpenMC
    python run.py detcheck                      # point detectors checked
    python run.py gapcheck                      # the v1.6.1 inputs (filters,
                                                # crossings, albedo, MCPL,
                                                # triggers, ...) vs OpenMC
    python run.py score                         # one number out of 10 from
                                                # tests, validate, robust and
                                                # detcheck results

What gmc3d accepts, class by class: `ACCEPTED_INPUTS.md`.  What it leaves
out: `GAPS.md`.

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
  tallies.py           flux, absorption and surface-current scoring, mesh
                       tracking, the mesh rules for balls, error bars
  transport.py         plain Monte Carlo + ball steps; the batched loop
  ext.py               the general scorer: every tally the fast kernels in
                       tallies.py do not take (any OpenMC filter, crossing
                       tallies as OpenMC scores them)
  detectors.py         point detectors: the next-event estimator, and how
                       a ball's unseen scatters are scored
  model_check.py       `run.py check`: every refusal and warning at once
  output.py            results to JSON; mesh tallies to VTK (ParaView)

ball/                THE BALL STEP
  walk.py              the exact walk inside a ball (the truth)
  oracle.py            backend 1: run the exact walk live
  table.py             backend 2: draw from pre-computed walks
  data.py              training data + the network's input/output encoding
  network.py           backend 3: a normalizing flow, and its training
  profile.py           where in a ball its flux lies (the 'profile' mesh
                       rule): a table of walk positions by path length

checks/              IS IT RIGHT, AND IS IT FAST
  openmc_ref.py        run OpenMC on the same model, read its tallies
  compare.py           ours vs a reference, bin by bin (z-scores)
  analytic.py          problems with exact answers
  validate.py          the Stage 1 suite (all problems, OpenMC, exact balls)
  robust.py            randomly built models vs OpenMC (pass or refuse)
  detcheck.py          point detectors: exact, balls vs plain, vs OpenMC
  gapcheck.py          every v1.6.1 input against OpenMC (20 cases)
  score.py             the score out of 10 and what it leaves out
  ball_metrics.py      one radius at a time: backend vs exact walk
  mesh_profile.py      the mesh rule vs exact walks, one ball at a time
  flux_maps.py         mesh tallies vs OpenMC as pictures (PNG per problem)
  measure.py           cost per flight, big-ball share, cost per draw, R*
  race.py              OpenMC vs our plain MC vs ball runs, timed
  gpu_bench.py         the network's cost per draw on a GPU, and what it
                       would do to the race (needs no OpenMC)
  net_profile.py       what one draw is MADE of: weights, flops, torch
                       operations, bytes, and the call stage by stage
  net_variants.py      candidate rewrites of the sampling path (columns,
                       compiled, CUDA graphs, float16), each checked
                       against the original before it is timed
  precision_check.py   what float16 and bfloat16 cost in accuracy,
                       ball by ball
  loop_profile.py      how big the backend's batches are and what sets
                       them; what a free backend would cost
  gpu_project.py       what a measured cost per draw does to a whole run

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

All 3D work lives here, on branch `3d-gmc` (SRS NF-8). Every push carries a
version: `VERSION` here and a git tag `v<version>` on the pushed commit.
v1.0 is Stage 1 as of 14 Oct 2026; v1.1.x is the mesh-tally work (bug fixes
and small features); v2.0 starts anisotropic scattering (Stage 2). `gmc2d/` stays frozen
so the 2D results remain reproducible; code reused from it (the spline flow,
the training loop conventions, the Monte Carlo conventions) is copied and
adapted, not imported. Each file's docstring says what uses it and what it
uses, and explains the design so it can be presented.
