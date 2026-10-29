# gmc3d changelog

Every push to `3d-gmc` gets a version, written in `VERSION` and tagged
`v<version>` in git. The first number is the stage (2.0 = anisotropic
scattering). Within version 1 the middle number is the area (1.1 mesh
tallies and the easy tally items, 1.2 sources, 1.3 multi-filter tallies,
1.4 geometry, 1.5 the medium tally items; renumbered 29 Oct 2026); the
last counts pushes within it.

## v1.4.1 (29 Oct 2026): hex lattices, tori, quadrics, rotation, distributed materials

- **`HexLattice`**, both orientations ('y' flat tops, 'x' flat sides), 2D
  and 3D. The element is found as OpenMC finds it (nearest of four
  candidate centres), and each element's universe comes from OpenMC's own
  index mapping. The six walls (and the axial ones in 3D) bound flights
  and balls.
- **Bug fixed (since v1.0): a lattice's outer universe.** OpenMC fills
  every would-be element outside the array with the outer universe in
  that element's own coordinates; we used one frame for all of them and
  stopped flights only at the array's edge. Right for a uniform outer
  universe (every test problem so far), wrong for any other. Now the
  would-be elements' walls count like real ones. Nested's results change
  within their error bars (no longer bit-identical to v1.4.0).
- **`XTorus`, `YTorus`, `ZTorus`** (circular or elliptical tube). Flights:
  the first root of a quartic, bracketed between its turning points (no
  general polynomial solver; 0.4 us a flight). Nearest distance: exact,
  the distance to the tube's ellipse in the plane through the axis.
  Refused: a tube that reaches the axis (c >= a).
- **`Quadric`** (any second-order surface). Flights: a quadratic. Nearest
  distance has no formula, so a guaranteed lower bound from the value,
  gradient and largest curvature: balls are never too big, only smaller
  than they could be. Exact for a plane.
- **Rotational periodic walls**: two planes through the z axis; a particle
  leaving through one is turned about z onto the other, by OpenMC's own
  angle rule (which takes the problem to lie on the + side of both).
- **Distributed materials** (a cell with one material per instance): every
  instance is numbered in OpenMC's order and the descent keeps the number
  as a running sum, so the cost is two additions per level. Refused: a
  wrong-length fill list, a distributed cell in an outer universe.
- New problems `hexcore` (both hex orientations, a non-uniform outer
  universe) and `sector` (a rotational quarter with an elliptical ZTorus,
  an XTorus, a tilted ellipsoid quadric and a distributed pin lattice).
- Measured in OpenMC 0.16: a distributed fill with a void entry stops
  OpenMC in multigroup mode, so that case is checked by unit test only.

## v1.4.0 (29 Oct 2026): tilted shapes, white and periodic walls

- **`Cylinder` and `Cone` on any axis** (OpenMC's general forms, which it
  writes as quadrics). Exact flight crossing, exact nearest distance (so
  balls stay as large as they can be against tilted walls), and the
  normal, all in `core/surfaces.py`. Checked against OpenMC's own surface
  function, by brute force on a fine grid of surface points (nearest is
  never too large and at most 0.008 cm smaller than the grid minimum), and
  by locating 3,000 random points in the new problem as OpenMC does.
- **White boundary**: a particle at the wall comes back in a new
  direction drawn from the cosine law about the inward normal, OpenMC's
  definition. Unit test: mu^2 and the azimuth are uniform (50,000 draws);
  a closed white sphere absorbs every particle.
- **Periodic boundary, translational**: a particle leaving one plane
  comes back through its partner plane with the same direction. Partners
  come from `periodic_surface`, or pair automatically for X/Y/Z planes as
  in OpenMC. Unit test: in the uniform slab, periodic side walls give the
  same leakage as mirror walls.
- Refused, with a message: rotational periodic (non-parallel partners,
  v1.4.1), periodic on a non-plane, `albedo` other than 1, a current tally
  on a white or periodic surface.
- New problem `tilted`: a block with a pipe along (1, 1, 1) and a cone
  along (1, -1, 2), periodic x walls, white y walls, vacuum z faces.
- Validate on all 7 problems and the results in `runs/2026-10-29/v1.4/`.

## v1.3.0 (29 Oct 2026): tallies with several filters

- **Several filters per tally**, combined as OpenMC does (the last
  filter's bin varies fastest), so results line up with OpenMC's
  `tally.mean` row for row. Track tallies: Cell, Material, Mesh, Energy in
  any mix. Current tallies: Surface with Cell, CellFrom (partial
  currents), Material, Energy.
- **Cells filled with a universe or lattice can be tallied.** The descent
  now records the cell at every level, and a cell filter matches at any
  level, as in OpenMC (a filled cell and a cell inside it both score).
  Ball steps keep the path of the ball's centre.
- **EnergyFilter**: one group, so every particle sits at the group's
  midpoint, as OpenMC does (measured).
- **Bug fixed (since v1.1): current through an inner surface** counted
  flights crossing the surface's plane outside the cell's real boundary.
  Nested core face: 0.0821 before, 0.0415 now, OpenMC 0.0408. Only vacuum
  faces had been validated, so no earlier result changes.
- Conventions measured in OpenMC 0.16: at a crossing a cell / material
  filter sees the cell entered, a cell-from filter the cell left; on a
  vacuum face both see the cell left.
- Refused, with a message: other filter types (Universe, CellInstance,
  ...), two filters of one kind, CellFrom on a track tally, Surface with
  Mesh.
- **Speed: plain MC 5-16% slower than v1.2** (sphere, cask). A 10x
  slowdown in the first build (a constant negative array index in Numba)
  was found and fixed, and one-filter tallies take a shortcut; the rest
  is an open check in `GAPS.md`.
- New tallies on nested (filled cells, mesh x material, cell x energy,
  partial currents) and slab (face currents by cell and cell-from).
  Validate: every new tally passes; all other results bit-identical to
  v1.2 at the same seed (`runs/2026-10-29/v1.3/`).
- 10 new unit tests (92 in total).

## v1.2.0 (29 Oct 2026): every OpenMC source

- **New `core/sources.py`.** Before each batch the source sites (position,
  direction, weight) are drawn in Python from the model's own OpenMC
  objects, and transport starts from that list. Spaces: `Point`, `Box`,
  `CartesianIndependent`, `CylindricalIndependent`,
  `SphericalIndependent`, `PointCloud`, `MeshSpatial` (regular 3D,
  rectilinear, cylindrical and spherical meshes). Angles: `Isotropic`,
  `Monodirectional`, `PolarAzimuthal`. Sources: `IndependentSource`,
  `MeshSource`, `FileSource` (.h5), and lists of several sources.
  Biased distributions give each site the matching weight.
- **Constraints** on cells, materials or universes at any depth, by
  drawing again, as OpenMC does. `energy_bounds`, `time_bounds` and
  `rejection_strategy = "kill"` are warned (they do nothing here, or in
  OpenMC 0.16 either).
- **Source strengths now count (silent bug before v1.2).** Each site weighs
  the sum of all strengths and leakage is per unit of source weight, as
  measured in OpenMC. Before, a source with strength 3 gave tallies a third
  of OpenMC's.
- **Russian roulette is relative to the birth weight**, so a site born
  with weight 2 is treated like one born with weight 1.
- Refused, with a message: `CompiledSource`, mcpl files, non-neutron
  particles, fissionable constraints, a bias on `Isotropic`.
- `python run.py sourcecheck`: 11 source cases against OpenMC, all PASS
  (one borderline mesh chi2 on the first seed, gone on two more).
  Validate: all six problems PASS after rechecks; the nested open check
  from v1.1 is closed at 10M histories (`runs/2026-10-29/v1.2/`).
- 20 new unit tests (82 in total).

## v1.1.3 (29 Oct 2026): no silent settings, more scores, more mesh shapes

- **Nothing in the input is ignored silently any more (settings and
  tallies).** Settings that change only noise or run length (survival
  biasing, weight windows, cutoffs, temperatures, `max_lost_particles`, and
  anything else not read) are named in a warning (`IgnoredSetting`, also
  in `Problem.warnings`, printed by `run.py solve`). Eigenvalue runs and
  photon transport are refused. Tally triggers are warned about.
- **Refused now, with a message:** estimators other than track length,
  tally nuclides (other than `total`), tally derivatives,
  `MeshSurfaceFilter` and `MeshBornFilter`. The last two used to pass as
  ordinary mesh filters and were scored wrongly without a word.
- **Scores `scatter` and `total`.** Each bin now holds flux, absorption,
  scatter and total, accumulated separately (a bin spanning two materials
  has no single cross section). `Results` tallies are (bins, 4).
- **Mesh shapes:** 1D and 2D `RegularMesh` (unbounded on missing axes, as
  in OpenMC) and `RectilinearMesh` (binary search per axis). All meshes
  share one grid walk built on bin edges; the v1.1 ball rule works
  unchanged.
- New tallies on the test problems: slab `depth1d` (1D, all four scores),
  scatter and total on the slab's total tally; cask `plan2d` (2D),
  `duct_grid` (rectilinear), `material` (all four scores), total on `cell`.
- **OpenMC 0.16 crashes** (segmentation fault) on a multigroup
  track-length `scatter` tally scored in a void cell, so the cask scores
  scatter by material only. Not our bug; noted in the cask problem.
- 18 new unit tests (62 in total). Validate against OpenMC: all six
  problems PASS (`runs/2026-10-29/`).

## v1.1.2 (27 Oct 2026): model-change column and the version 1 plan

- `GAPS.md` gains a "model change?" column: whether closing each gap needs
  the trained network to change. Only the physics stages, balls over 30 mfp
  and the network's short-path tail do; those are on hold.
- `V1_PLAN.md`: the remaining version 1 work (source, geometry, tallies;
  the small and medium gaps), as pushes v1.1.3 to v1.1.8. No code changes.

## v1.1.1 (27 Oct 2026): the gaps list

- `GAPS.md`: everything gmc3d leaves out so far (geometry, physics, source,
  tallies, ball-method limits, open checks), with whether each is refused
  or silently ignored, and a rough size. Updated with every version from
  now on. No code changes.

## v1.1.0 (27 Oct 2026): flux maps right inside balls, surface currents

- **New mesh rule `profile`, now the default everywhere.** A ball's flux is
  split over up to 16 points of its walk, each drawn from where real walks
  lay their flux (`ball/profile.py`). Exact on average for every voxel;
  balls stay full-size. Absorption is handled exactly, with no grid in
  sigma_a / sigma_s.
- **The `centre` rule is gone.** It put a whole ball in one voxel and
  smeared every map (chi2/dof against OpenMC: sphere 73,816, slab 538).
  `cap` (shrink the ball to one voxel) stays as an option.
- **Surface current tallies.** An OpenMC `SurfaceFilter` with score
  `current` on root-universe vacuum or transmission surfaces: the signed
  weight crossing, + towards the surface's positive side (OpenMC's sign,
  checked on the slab). The slab problem now reports transmission and
  reflection separately (tally `sides`).
- **Validate now passes or fails on every tally**, mesh tallies included,
  and covers all six problems (the cask was added).
- New commands: `run.py meshcheck` (the profile against exact walks) and
  `run.py maps` (flux maps against OpenMC as PNG files).
- Results: `runs/2026-10-27/MESH.md`.

## v1.0 (commit 177c5db, 14 Oct 2026): Stage 1

Any CSG geometry, one group, isotropic scattering, fixed source; exact
walk, lookup table and network backends; validation against OpenMC; the
speed study, GPU measurements and transport-loop breakdown
(`runs/2026-10-05/STAGE1.md`, `runs/2026-10-06/SPEED.md`).
