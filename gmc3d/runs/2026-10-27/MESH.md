# v1.1: flux maps right inside balls (27 Oct 2026)

Before v1.1, totals, cells, materials and leakage were right, but mesh
tallies (flux maps) were wrong wherever balls ran: the old `centre` rule
put a whole ball's flux in the voxel at its centre. v1.1 replaces it with
the `profile` rule, which is exact on average. Every number below comes
from the logs in this folder.

## The idea, in plain words

A ball step tells us how far the particle walked inside the ball (s) and
where it came out, but not where it went inside. The flux it laid down is
spread through the ball, and the ball can cover hundreds of voxels.

Imagine rain over a field. You can't see where every drop falls, but you
know how rain spreads on average. So instead of tipping the whole bucket
into the middle square (`centre`), you throw it at random squares, picked
so that over many buckets each square gets exactly its fair share. Any one
bucket lands in the "wrong" place, but the totals per square come out
right. That is "exact on average", the same promise every Monte Carlo
tally makes. OpenMC's collision estimator works the same way.

How a point is picked (`ball/profile.py`):

1. **When along the walk.** Pick a path length T between 0 and s. Later
   parts of the walk carry less weight, because absorption makes the weight
   decay as exp(-kappa T), with kappa = sigma_a / sigma_s. So T is drawn
   with exactly that weighting, from a formula, using the ball's own s.
2. **Where walks are at that path length.** A pre-computed table records,
   for 32 ball sizes and 96 path lengths, how far from the centre walks
   that are still inside the ball sit (one million exact walks per size).
   The draw takes a radius from it and a random direction.
3. **Not yet scattered.** A walk that hasn't scattered by T is at radius T
   exactly, so those points are placed exactly.

Absorption never enters the table: it enters only through step 1, exactly.
That was the second design. The first one tabulated the flux profile on a
grid of kappa and blended neighbouring grid points, and it failed in
absorbers (chi2/dof up to 900 against exact walks at kappa = 19).

**Splitting.** One point per ball is exact on average but very noisy: a
10 cm ball in 1 cm voxels dumps its whole flux into one of about a
thousand voxels. So the flux is split over K points, with
K = min(16, ceil(2 (d/h)^2)), where d is the ball radius and h the voxel
width. K changes only the noise and the cost, never the mean. Exact
balls, 2M histories, mesh chi2/dof against OpenMC:

| points per ball (most) | sphere | time (s) | slab | time (s) |
|---|---|---|---|---|
| 1  | 1.945 | 1.9 | 1.346 | 6.8 |
| 4  | 1.192 | 2.5 | 1.116 | 8.5 |
| 16 | 1.074 | 5.2 | 0.870 | 12.7 |
| 64 | 1.079 | 15.0 | 0.847 | 22.7 |

Past 16 points nothing improves, so the cap is 16. With one point the
sphere's chi2 of 1.9 came from voxels at the edge, which hold a thin
sliver of material: they were hit rarely but hard. The totals there were
right (sum over the sliver voxels: balls 5.18e-4, plain MC 4.85e-4, OpenMC
4.68e-4), but the error bars weren't trustworthy for such rare, large
scores.

## Check 1: one ball at a time (`meshcheck.log`)

The rule against fresh exact walks, with no geometry and no OpenMC: the
share of flux in each of 32 radial shells, 400,000 walks a side. The test
points sit between table grid points and run from pure scatterers to the
lattice3d absorber (kappa = 19).

| R (mfp) | kappa | shells | chi2/dof | max abs z |
|---|---|---|---|---|
| 3.0 | 0 | 32 | 1.016 | 2.86 |
| 3.0 | 0.1 | 32 | 1.032 | 2.53 |
| 5.5 | 0 | 32 | 1.115 | 2.29 |
| 5.5 | 0.01 | 32 | 0.729 | 2.32 |
| 5.5 | 0.25 | 32 | 0.829 | 2.31 |
| 12 | 0 | 32 | 0.846 | 1.72 |
| 12 | 0.0333 | 32 | 1.194 | 2.19 |
| 12 | 2.0 | 9 | 0.722 | 1.13 |
| 25 | 0 | 32 | 0.753 | 1.93 |
| 25 | 0.0083 | 32 | 0.641 | 2.08 |
| 25 | 0.1 | 25 | 1.113 | 2.31 |
| 1.5 | 19 | 10 | 0.814 | 1.77 |
| 8 | 19 | 2 | 0.026 | 0.18 |

All 13 agree within noise. Shells holding under 1e-4 of the flux are left
out: in strong absorbers that is everything beyond the centre.

## Check 2: validate, all six problems vs OpenMC (`validate.log`)

2M histories per run. Exact balls with R* = 1, the hardest test of the
driver. OpenMC now uses 400 batches so its error bars can be trusted (see
the note at the end). Pass rule: no tally whose chi2 has p < 0.001, and no
single number with |z| > 3.29.

| problem | mesh bins | plain MC chi2/dof | exact balls chi2/dof | verdict |
|---|---|---|---|---|
| sphere | 5,078 | 1.002 | 1.043 | PASS / PASS |
| slab | 60 | 0.895 | 1.365 | PASS / PASS |
| curved | 2,744 | 0.950 | 0.927 | PASS / PASS |
| nested | 1,000 | 0.911 | 1.020 | PASS / PASS |
| lattice3d | 49 | 0.894 | 0.894 (no balls fit) | PASS / PASS |
| cask | 1,236 | 1.005 | 0.998 | PASS / PASS |

**Overall: PASS.** Before v1.1 the exact-ball mesh numbers were 73,816
(sphere), 538 (slab) and 2,200 (curved).

The slab's 1.365 sits high, but its 60 depth bins are strongly correlated,
so their chi2 swings a lot. The first validate run gave 1.499, an earlier
test 0.870. To check for a hidden bias, balls were run against plain MC at
10M histories each: chi2/dof 0.642, max |z| 2.04. A real bias would grow
with more histories, and this shrank. So it is noise.

## Check 3: the network, all six problems (`maps.log`, `maps/`)

The trained network, R* = 3, `profile` rule, 2M histories, against OpenMC.
One PNG per problem in `maps/`. Each shows our map, OpenMC's map and the z
map, three mid-planes for the 3D meshes and a curve for the slab.

| problem | mesh chi2/dof | max abs z | leakage z |
|---|---|---|---|
| sphere | 1.055 | 4.20 | +0.70 |
| slab | 0.568 | 2.61 | -0.51 |
| curved | 0.985 | 3.92 | +1.08 |
| nested | 1.063 | 3.21 | +0.04 |
| lattice3d | 1.053 | 2.53 | +1.25 |
| cask | 1.113 | 4.59 | +0.55 |

**Before (v1.0, network, `centre` rule, same settings):**
`maps/before_v1.0_*.png`. The sphere is at chi2/dof 70,593, the slab at
2,038 and the cask at 6.75. The slab picture shows the old rule's damage:
a saw-tooth near the source and the far face, and a curve that sags by 30%
deep in the slab.

## Per-surface current (the slab's `sides` tally)

Current through x = 0 (reflection, negative) and x = 60 (transmission,
positive), matching OpenMC's sign. Against OpenMC: plain MC max |z| 0.62,
exact balls 1.45, network 0.48. A unit test checks that transmission minus
reflection equals the total leakage to 1e-12.

## What it costs (`mesh_noise.log`)

2M histories, seed 5, network on THIS container's CPU. FOM is for the mesh
flux, taken as the median relative error over voxels:

| problem | run | time (s) | median rel. error | FOM vs plain MC |
|---|---|---|---|---|
| sphere | plain MC | 9.7 | 3.844% | 1.00 |
| sphere | network, profile | 18.1 | 3.700% | 0.58 |
| slab | plain MC | 12.2 | 0.513% | 1.00 |
| slab | network, profile | 19.1 | 0.467% | 0.77 |
| cask | plain MC | 22.2 | 16.234% | 1.00 |
| cask | network, profile | 33.9 | 16.108% | 0.67 |
| curved | plain MC | 50.8 | 2.037% | 1.00 |
| curved | network, profile | 67.7 | 1.994% | 0.78 |

- **Noise is not the problem.** Per history, the `profile` maps are
  slightly LESS noisy than plain MC's (3.70% vs 3.84% on the sphere).
- **The CPU network's time is the problem.** That is the same finding as
  STAGE1 and SPEED, not something v1.1 caused. On the sphere 16.0 s of the
  18.1 s is the network (`maps.log`), and transport alone is 5.7 s.
- **GPU context, a projection and not a run.** At the 0.30 us per draw
  measured on Phani's T4, the sphere's 2M draws would take about 0.6 s.
  That puts the run near 6.3 s and the mesh FOM near 1.7x plain MC.
- **The mesh points themselves cost about 0.1 us each**, up to 16 per ball:
  about 3.3 s on the sphere's 2M balls (the K table above).
- `cap` matches plain MC here because with 1 cm voxels and R* = 3 almost no
  balls fit. So `cap` is exact and costs nothing on fine meshes, but it
  gives up the balls.

## Things found along the way

- **OpenMC's error bars with 20 batches were not good enough for a
  pass/fail on big meshes.** Each one is uncertain by about 16%, so z
  follows Student's t and z^2 averages 1.12. The first validate run used 20
  batches and failed several tallies, plain MC included (lattice3d
  absorption 1.635). Same histories in 400 batches gave the same answers
  and honest error bars, and everything passed.
- **Nested leakage z is -3.20 for plain MC and -3.25 for exact balls** in
  validate (OpenMC seed 101). Plain MC has no balls, so this is not v1.1.
  With seed 105 (maps run) the network gets +0.04. It is the same pattern
  as 13 Oct (-2.3 at 4M, -0.3 at 10M). Worth a 10M recheck when convenient.
- **Numba cache trap.** A `@njit(cache=True)` function that calls a
  function in ANOTHER module keeps the old compiled callee after that
  module is edited. This produced a fake failure (chi2 in the thousands)
  until the caches were deleted. After editing an njit function, delete
  `*.nbi` / `*.nbc`.

## Commands

    python run.py meshcheck      # Check 1
    python run.py validate       # Check 2
    python run.py maps           # Check 3 and the PNGs (--backend oracle|table|network)
