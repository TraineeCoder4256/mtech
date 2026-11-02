# The tally path: ours or OpenMC's advantage?

Speed thread, 2 Nov 2026, same session for every row, 500,000 histories,
plain MC (no ball backend).  OpenMC runs with survival biasing and the
same weight cutoff, so both codes play implicit capture.  Scripts and raw
logs beside this file.  Nothing in `core/` or `ball/` changed; the two
ablations are throwaway copies.

## The answer, up front

**We are not doing tallies wrong in general, and OpenMC is not
especially good at them.**  Scoring the same tallies costs ours 1.28x
OpenMC's on the sphere, 1.35x on the slab, and **0.76x on the cask**,
where we are the cheaper of the two.

**The whole of today's gap to OpenMC is the bare transport.**  Taking
each code's own no-tally run as its transport:

| | sphere | slab | cask |
|---|---|---|---|
| ours, no tallies | 3.08 s | 3.57 s | 5.53 s |
| OpenMC, no tallies | 0.71 s | 0.86 s | 1.21 s |
| **transport ratio** | **4.3x** | **4.2x** | **4.6x** |
| ours, tally cost | 3.51 s | 3.71 s | 4.82 s |
| OpenMC, tally cost | 2.75 s | 2.75 s | 6.33 s |
| tally ratio | 1.28x | 1.35x | 0.76x |
| whole-run gap | 3.13 s | 3.67 s | 2.81 s |
| of which transport | 2.37 s (76%) | 2.71 s (74%) | 4.32 s (154%) |

So the headline ratios from this morning (2.29x / 1.81x / 1.31x) UNDERSTATE
the transport problem: our tally scoring is about as expensive as
OpenMC's, and that shared cost dilutes the ratio.  On the cask the
tallies are actually carrying us.

**This corrects what I said earlier today.**  I wrote that 72% of the
sphere's increase since v1.0 is tally scoring and left the impression
that the tally path was the bigger half of the gap to OpenMC.  The first
half is right about our own growth; the second is wrong.  Measured, the
tally path is 24% of the sphere's gap, 26% of the slab's, and on the cask
it is in our favour.

## Each tally on its own

Seconds a single tally adds to that problem's no-tally run, both codes
seeing the same tally object.  `bins` is OpenMC's own `num_bins`.

### sphere (ours 3.08 s bare, OpenMC 0.71 s)

| tally | bins | ours | OpenMC |
|---|---|---|---|
| total (no filter) | 2 | 0.00 | +0.50 |
| cell | 2 | 0.00 | +0.65 |
| mesh, 20^3 Cartesian | 16000 | +0.68 | +0.86 |
| **sphmesh, 5 x 3 x 4 spherical** | **120** | **+2.22** | +1.35 |
| all four | 16124 | +3.51 | +2.75 |

### cask (ours 5.53 s bare, OpenMC 1.21 s)

| tally | bins | ours | OpenMC |
|---|---|---|---|
| total | 2 | 0.00 | +2.09 |
| cell, 7 cells x 3 scores | 21 | 0.00 | +1.89 |
| material | 12 | +0.34 | +2.52 |
| mesh, 12^3 | 3456 | +1.75 | +1.22 |
| plan2d, 24^2 | 1152 | +1.90 | +1.30 |
| duct_grid, rectilinear | 825 | +1.69 | +1.52 |
| all six | 5468 | +4.82 | +6.33 |

### slab (ours 3.57 s bare, OpenMC 0.86 s)

| tally | bins | ours | OpenMC |
|---|---|---|---|
| total, 4 scores | 4 | +0.11 | +0.88 |
| depth1d, 30 bins | 120 | +1.09 | +1.07 |
| depth, 60 x 1 x 1 | 120 | +1.48 | +0.91 |
| sides, surface current | 2 | +0.22 | +0.14 |
| **sides_from, with a cell-from filter** | 2 | **+0.69** | -0.07 |
| sides_cell, with a cell filter | 2 | +0.55 | +0.13 |
| **faces1d, mesh surface currents** | 24 | **+1.45** | +0.31 |
| all seven | 274 | +3.71 | +2.75 |

Two readings jump out.  An unfiltered or cell-filtered tally is free for
us and costs OpenMC 0.5 to 2.1 s: OpenMC pays a large fixed price just
for having tallies switched on, where our `_fast` path is a single
lookup.  And our Cartesian meshes are in OpenMC's league (0.68 s for
8,000 voxels).

The rows in bold are where we are genuinely worse, and all three are the
same mistake wearing different clothes: **recomputing from scratch
something the step before already established.**  Details and the proof
for each are below.  (The sphmesh row reads +2.22 s here and +2.03 s in
the scaling table further down; same tally, two passes, 8-15% apart,
which is this container's spread.)

## The three places we really are slower, and why

### 1. Curved meshes: we rescan, OpenMC steps

A Cartesian mesh is walked by a DDA: keep the distance to the next wall
on each of the three axes, step the nearest one, update that one axis.
O(1) a step, which is why 8,000 Cartesian voxels cost 0.68 s.

`_next_cut` (`core/tallies.py`) does something else for a spherical or
cylindrical mesh.  At every cut it re-solves the intersection with
**every** r sphere, **every** phi half-plane and **every** theta cone of
the mesh, then takes the nearest.  It never uses the fact that the ray
just crossed a known surface and can only reach its neighbours next.
`_point_bin` then searches the bins again for the midpoint of each piece.

That predicts something testable: refine the mesh and our cost should
grow with the number of edge surfaces, while a code that steps
incrementally should not care.  Same sphere problem, only the spherical
mesh's `r_grid` changed, everything else fixed:

| r bins | tally bins | ours | OpenMC |
|---|---|---|---|
| 1 | 24 | +1.70 s | +1.14 s |
| 2 | 48 | +1.94 s | +1.08 s |
| 5 | 120 | +2.03 s | +1.08 s |
| 10 | 240 | +2.71 s | +1.46 s |
| 20 | 480 | +2.98 s | +1.32 s |
| 40 | 960 | +4.44 s | +1.29 s |

**OpenMC is flat across a 40-fold refinement.  We grow 2.6x.**  That is
the rescan, measured.  Note also that even at one r bin we cost 1.70 s,
so most of it is the per-cut scan over the phi and theta surfaces, not
the r ones.

### 2. Mesh surface currents: we walk the mesh twice

`score_mesh_faces` is a second, complete DDA over the same mesh, run
after `_mesh_segment` has already walked it, with an `np.exp` at every
face crossed.  OpenMC gets its face crossings out of the single walk it
already does.  Slab `faces1d`: ours +1.45 s, OpenMC +0.31 s, **4.7x**.

### 3. A surface tally with a cell filter costs us a second geometry walk

When a surface tally carries a cell or cell-from filter, every crossing
calls `G._path` again, a full descent from the root, to find out which
cell the particle entered.  The transport knew that one flight later
anyway.  Slab: `sides` (no cell filter) +0.22 s, `sides_cell` +0.55 s,
`sides_from` +0.69 s, against OpenMC's +0.14 / +0.13 / -0.07.

## Two generic costs that turned out not to matter

Both tested by patching a throwaway copy, physics deliberately wrong,
only the clock read.

* **Four scores per bin.**  Every bin holds flux, absorption, scatter and
  total, and `_add` writes all four even for a flux-only tally.  Making
  it write one: tally cost 3.77 / 3.11 / 5.03 s against the baseline's
  3.53 / 3.41 / 5.49.  No saving beyond noise.  **Limit:** this removed
  the arithmetic but kept the array layout, so the same cache lines are
  touched either way.  It does not rule out the 4x memory footprint;
  testing that needs NS = 1 throughout, which breaks the results reader,
  so it is a code change rather than an experiment.
* **The exponential per voxel.**  `_decayed` takes an `exp` and an
  `expm1` for every voxel, because our weight decays continuously along a
  flight while OpenMC's survival biasing keeps it constant between
  collisions.  Replacing it with `w * length`: 3.54 / 3.18 / 4.53 s
  against 3.53 / 3.41 / 5.49.  Worth 0% on the sphere, 7% on the slab,
  **17% on the cask**, which has three meshes and the most voxel
  crossings.  Real but second order, and it could be halved exactly
  rather than removed: score each voxel from the weight at its two walls,
  `(E_i - E_i+1) / sigma_a`, which needs one exponential per wall instead
  of two per voxel.  The reason the current form exists is accuracy: the
  difference of two exponentials cancels badly when a voxel is thin, which
  is what `expm1` avoids, so it would have to be a hybrid.

## What I would look at, in order (nothing landed, nothing approved)

1. **The transport, not the tallies.**  This is the whole gap.  The two
   causes from this morning's note stand: the source sampler and the
   `Geometry` tuple.
2. **Stop rescanning in the curved-mesh stepper** (item 1 above).  It is
   the most expensive single tally in the suite and the only one whose
   cost grows where OpenMC's does not.
3. **Fold the mesh-face crossings into the volume walk** (item 2), and
   **carry the cell across a crossing instead of re-descending** (item 3).
   Item 3 is the same "cache the cell" idea that the 30 Oct note raised
   for the transport, which would pay twice.

## What this does not say

* Nothing here involves the ball step or the network: plain Monte Carlo
  throughout, and the tally sets are the problems' own.
* OpenMC's large fixed cost for simple tallies (up to +2.09 s on the
  cask for one unfiltered tally) is its own business; it is what makes
  the whole-run ratios look kinder to us than the transport deserves.
* Single runs on this container spread 8-15%, so only differences well
  outside that are called out, and ablations compare rows from the same
  pass.
