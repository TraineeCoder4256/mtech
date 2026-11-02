# Does the ball step still buy a speed-up at v1.6.1?

2 November 2026, speed track, branch `gmc-speed`.  Asked by Phani after
the plain-Monte-Carlo findings in `VERSIONS.md` and `TALLIES.md`: those
numbers are for plain MC, so if the surrogate is used instead, is there
still a speed-up?

Nothing in `core/` or `ball/` was changed.  Every number below comes from
a worktree of a released commit, timed in one session on the same four
cores.

## How it was measured

500,000 histories (100k x 5 batches), `r_star = 2`, the fastest of three
repetitions inside a round, two interleaved rounds, so v1.0 and v1.6.1
meet the same machine.  Run-to-run spread on this container is 8-15%.

The backend is **the exact walk inside the ball** (`ball/oracle.py`), not
the trained flow: this container has no trained model (`models/` is
gitignored).  The exact walk is the *free-backend ceiling* — it is what a
perfect, zero-cost surrogate would score, and its own time is reported
separately (0.14-0.44 s per run, 4-12% of a run).  What the real network
would add on top is in the last section.

OpenMC runs with `survival_biasing = True` and
`cutoff = {"weight": 0.25, "weight_avg": 1.0}`, the same weight game as
ours, 500 x 1000 batches so its error bar is as finely split as ours,
fastest of three seeds.  Each problem's **own tally set is on both
sides**, so OpenMC pays for the same tallies we do.

v1.0 ball runs use `mesh_rule = "centre"` because `"profile"` did not
exist yet; v1.6.1 uses `"profile"`.  That is not a like-for-like
correctness comparison and it matters — see cause 2.

## The short answer

**Yes, but only on the cheap tallies, and the margin is a third of what
it was.**  Ball step against our own plain Monte Carlo, wall time:

| problem | v1.0, no tallies | v1.0, own tallies | v1.6.1, no tallies | v1.6.1, own tallies |
|---|---|---|---|---|
| sphere | 1.17 / 0.63 = **1.84x** | 1.55 / 0.66 = **2.36x** | 2.73 / 1.61 = **1.70x** | 6.22 / 3.52 = **1.77x** |
| slab   | 1.66 / 1.12 = **1.48x** | 2.08 / 1.17 = **1.77x** | 3.59 / 2.72 = **1.32x** | 7.09 / 9.00 = **0.79x** |
| cask   | 2.90 / 1.74 = **1.67x** | 3.28 / 1.95 = **1.68x** | 5.39 / 3.43 = **1.57x** | 10.70 / 8.89 = **1.20x** |

(plain MC seconds / ball seconds = speed-up.)

And against OpenMC, which is the number that decides whether the method
is worth anything, the ball step no longer wins:

| problem | contender | seconds | vs OpenMC | rel. err. | FOM x |
|---|---|---|---|---|---|
| sphere | OpenMC implicit | 3.84 | - | 0.561% | 1.00x |
|        | ours, plain MC | 6.22 | 0.62x | 0.579% | 0.58x |
|        | ours + exact-walk balls | 3.52 | 1.09x | 0.834% | **0.49x** |
| slab   | OpenMC implicit | 3.92 | - | 0.0435% | 1.00x |
|        | ours, plain MC | 7.09 | 0.55x | 0.0424% | 0.58x |
|        | ours + exact-walk balls | 9.00 | 0.44x | 0.0408% | **0.50x** |
| cask   | OpenMC implicit | 8.26 | - | 2.044% | 1.00x |
|        | ours, plain MC | 10.70 | 0.77x | 2.069% | 0.75x |
|        | ours + exact-walk balls | 8.89 | 0.93x | 2.070% | **0.91x** |

At v1.0 the same backend ran the sphere's own tallies in 0.66 s against
OpenMC's 2.43 s (`vsopenmc_v100.json`), about 3.7x on wall time.  Today
it is 1.09x on wall time and 0.49x on FOM.  **The overhead has eaten the
ball step's entire advantage over OpenMC.**

## The ball run lost more than plain MC did

| | v1.0 | v1.6.1 | |
|---|---|---|---|
| sphere, no tallies | balls 0.63 | 1.61 | **2.53x** slower (plain MC 2.34x) |
| sphere, own tallies | balls 0.66 | 3.52 | **5.36x** slower (plain MC 4.02x) |
| slab, no tallies | balls 1.12 | 2.72 | **2.42x** (plain MC 2.16x) |
| slab, own tallies | balls 1.17 | 9.00 | **7.67x** (plain MC 3.41x) |
| cask, no tallies | balls 1.74 | 3.43 | **1.97x** (plain MC 1.86x) |
| cask, own tallies | balls 1.95 | 8.89 | **4.55x** (plain MC 3.26x) |

This is the important result.  A ball run on the sphere does 0.04 flights
per history against plain MC's 18.57 — 460 times fewer — yet it slowed
down by as much as plain MC did.  So almost none of the overhead we added
since v1.0 is per flight.  It sits exactly where the ball step cannot
skip it.

### Cause 1: the source sampler, which is per history

Re-measured in this session: **156.8 ms (sphere), 149.6 (slab), 144.1
(cask) per 100k batch**, i.e. 0.72-0.78 s of a 500k run, independent of
the problem.  A ball run cannot amortise it over flights, because it is
charged once per particle born.

Share of today's ball run with no tallies: **48% (sphere)**, 28% (slab),
21% (cask).  Take it off and the ball run is 0.83 / 1.97 / 2.71 s against
v1.0's 0.63 / 1.12 / 1.74 — so on the sphere the source sampler is most
of the ball run's whole regression.

Same cause as in `VERSIONS.md`, but it hurts the ball step twice as much,
because the ball step shrinks everything the sampler is competing with.

### Cause 2: the `profile` mesh rule, which is per ball

Scoring cost a ball run almost nothing at v1.0 and is now the larger half
of it: sphere 0.66 - 0.63 = **+0.03 s** at v1.0, 3.52 - 1.61 = **+1.91 s**
at v1.6.1.  v1.0's ball runs used the `centre` rule, which dropped a
ball's whole path into the voxel holding its centre: one scoring call per
ball, and wrong on average.  v1.1.0 replaced it with `profile`, which
spreads a ball's path over a radial profile — right on average, and many
scoring calls per ball.  The correctness was worth having; this is its
price, and it lands entirely on ball runs.

### Cause 3: some tallies cap the ball step, or switch it off

Each tally timed alone, 500k histories, flights and balls per history
next to it:

| problem | tally | plain MC | balls | speed-up | flights mc / balls | balls |
|---|---|---|---|---|---|---|
| sphere | (none) | 2.60 | 1.56 | 1.66x | 18.6 / 0.0 | 1.00 |
| sphere | total | 2.64 | 1.43 | 1.85x | 18.6 / 0.0 | 1.00 |
| sphere | cell | 2.61 | 1.47 | 1.77x | 18.6 / 0.0 | 1.00 |
| sphere | mesh | 3.66 | 2.97 | **1.23x** | 18.6 / 0.0 | 1.00 |
| sphere | sphmesh | 5.03 | 3.04 | 1.66x | 18.6 / 0.0 | 1.00 |
| slab | (none) | 3.64 | 2.58 | 1.41x | 22.7 / 7.4 | 1.07 |
| slab | total | 3.77 | 2.53 | 1.49x | 22.6 / 7.4 | 1.07 |
| slab | depth1d | 4.48 | 4.04 | 1.11x | 22.6 / 7.4 | 1.06 |
| slab | depth | 4.49 | 4.45 | **1.01x** | 22.6 / 7.4 | 1.07 |
| slab | sides | 3.73 | 2.45 | 1.52x | 22.6 / 7.4 | 1.07 |
| slab | sides_from | 3.46 | 2.60 | 1.33x | 22.6 / 7.4 | 1.06 |
| slab | sides_cell | 3.53 | 2.41 | 1.46x | 22.6 / 7.4 | 1.07 |
| slab | faces1d | 4.51 | 4.81 | **0.94x** | 22.6 / **17.2** | 1.14 |
| cask | (none) | 5.21 | 3.36 | 1.55x | 40.0 / 10.1 | 2.18 |
| cask | total | 5.42 | 3.36 | 1.61x | 40.0 / 10.2 | 2.18 |
| cask | cell | 5.14 | 3.33 | 1.54x | 40.1 / 10.1 | 2.18 |
| cask | material | 5.22 | 3.46 | 1.51x | 40.1 / 10.1 | 2.18 |
| cask | mesh | 7.00 | 4.16 | 1.68x | 40.0 / 10.1 | 2.18 |
| cask | plan2d | 6.89 | 4.31 | 1.60x | 40.1 / 10.2 | 2.18 |
| cask | duct_grid | 7.48 | 6.73 | **1.11x** | 40.0 / 10.2 | 2.18 |

Three things to read off it:

1. **Plain tallies keep the full speed-up**: `total`, `cell`, `material`
   and the surface tallies all stay at 1.3-1.85x.  A ball run scores them
   from the ball's own profile and never loses a flight.
2. **`faces1d` caps the ball step.**  It is a mesh-surface current tally,
   and `tallies.mesh_room(..., faces_only=True)` shrinks every ball so it
   cannot carry a particle across a mesh face unseen.  Flights per history
   go 7.4 -> 17.2, and the speed-up goes to 0.94x: the ball step is a net
   loss.  This is why the whole slab set is 0.79x — `faces1d` removes the
   flight saving while every other tally still charges profile scoring.
3. **Volume mesh tallies eat it from the other side**: slab `depth` 1.01x,
   cask `duct_grid` 1.11x, sphere `mesh` 1.23x.  Balls are kept, but the
   profile scoring costs about what the flights would have.

There is a fourth, not in the table because it is categorical:
`ext.no_balls` switches the backend off completely for any tally filtering
on direction (polar, azimuthal, spherical harmonics, `mu`) or on collision
count, because a ball hides both.  `openmc_import` warns and the run
silently becomes plain MC.  So part of what v1.6.1 bought — "handles any
OpenMC input" — is coverage the surrogate cannot serve at all.

## What the real network would add on top

The exact walk's own time here is 0.14-0.44 s per 500k run.  A network
replaces that with one draw per ball (1.0-2.2 balls per history, so
0.5-1.1M draws):

- On a T4, compiled, measured 6 Oct: 0.30 us per draw at a large batch ->
  **0.15-0.33 s**, the same order as the exact walk.  The model is not the
  problem.
- On this CPU, 4-7 us per draw -> 2-7 s, which would wipe out the whole
  remaining margin.  CPU-side inference is still a non-starter.

So the ceiling in the tables above is the honest one for a GPU network,
give or take a tenth of a second.  What limits the method today is the
driver, not the surrogate.

## What this does not say

- No trained flow in this container, so no accuracy claim here, and the
  network's own cost is projected from the 6 October T4 bench, not run.
- `r_star = 2` only.  The network's best measured R* was 10; a larger R*
  replaces more flights per draw and would move the ball rows, in both
  time and variance.
- v1.0's with-tallies ball runs used the `centre` mesh rule, which was
  wrong on average.  Part of the 5-8x in that column is the v1.1.0
  correctness fix, not waste.
- Two rounds, three repetitions each; 8-15% spread, so nothing under about
  1.2x is a real difference.
- FOM uses leakage only.  On the sphere the ball step's own error bar is
  48% wider than plain MC's (0.834% against 0.579%), which is most of why
  its FOM is lower than its wall-time ratio suggests.

## What I would look at, in this order

1. **The source sampler.**  It is now 48% of the sphere's ball run and
   nothing in the ball step can hide it.  Same fix as in `VERSIONS.md`:
   parallel, no 1.2x over-draw, ask only "is there material here?", cache
   a point source.
2. **Profile scoring per ball.**  +1.91 s on the sphere where `centre`
   cost +0.03 s.  Worth asking how many profile points a ball really
   needs, and whether the points can share the mesh walk instead of each
   re-searching it (the `_next_cut` and `_point_bin` rescan in
   `TALLIES.md`, which a ball run pays once per profile point).
3. **`faces1d`-style capping.**  A mesh-surface current tally currently
   costs the ball step everything.  Whether a ball can report its face
   crossings analytically, from the profile it already has, is a research
   question, not an engineering one.

Nothing here is approved or landed.
