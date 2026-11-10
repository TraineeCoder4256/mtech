# gmc3d changelog

Every push to `3d-gmc` gets a version, written in `VERSION` and tagged
`v<version>` in git. The first number is the stage (2.0 = anisotropic
scattering). Within version 1 the middle number is the area (1.1 mesh
tallies and the easy tally items, 1.2 sources, 1.3 multi-filter tallies,
1.4 geometry, 1.5 the medium tally items; renumbered 29 Oct 2026); the
last counts pushes within it.

## v2.1.1 (10 Nov 2026): interface v2, the anisotropic ball model spec

Agreed with Development on 10 Nov (Phani: "discuss with development for
network related stuff"). It adds a second spec and leaves v1 as it is.
Balls in isotropic materials keep the v1 call and model; balls in
anisotropic materials will call a v2 model.

- **`ball/interface.py`**, `ANISO_*`, `ANISO_INTERFACE_VERSION = 2`:
  - in: R in [1, 30], and the law's Legendre moments g1..g12 of the
    density OpenMC samples;
  - out: cos_n, mu, phi, s (as walk_aniso);
  - `check_aniso_outputs`.
  - `INTERFACE_VERSION` stays 1, so v1 models load as before.
- **`ball/laws.py`** (new): the training family, with mean cosine -0.5 to
  0.95. It mixes Henyey-Greenstein, two-lobe HG, Legendre P1-P5 through
  OpenMC's conversion, and histograms of 4-16 bins. The held-out set is
  the test problems' laws.
- **`ball/data.py`**: `make_aniso` and `dataset_aniso` (one law per row
  from a pool of 20,000) and `data_hash`. The 1M-row reference set has
  sha256 e9969835...82dcbbc31.
- **`checks/aniso_truncation.py`**: how many moments the exit depends on
  (`runs/2026-11-10/interface-v2/README.md`). It takes 3-5 for moderate
  laws. Peaked ones (HG 0.9) in balls of 1-3 mfp still differ at 8 (KS
  0.048) and mostly not at 12 (0.018), hence 12 moments, not the 8 first
  proposed.
- Tests: `tests/test_interface_v2.py` (4).

## v2.1.0 (10 Nov 2026): balls in anisotropic materials, exact walk

Phani's go (10 Nov): v2.1 before the benchmarks; network matters with
Development. The ball step now works in anisotropic materials with the
exact walk (the Oracle). Trained models stay isotropic-only until
interface v2 (agreed with Development on 10 Nov, its own push next).

- **`ball/walk.py` walk_aniso** (new): the walk starts at the centre along
  the direction u0 just sampled from the law (or the source's), and every
  scatter turns by the law. Its exit, by symmetry, is four numbers:
  cos_n (exit normal . u0), mu, phi (exit azimuth about the normal, from
  the plane of n and u0; the driver draws its sign), s. With isotropic
  scattering it reproduces v1's (mu, s) (KS tests). The exit point leans
  towards u0 even then, because the first flight runs along it.
- **Driver**: anisotropic materials get balls when the backend has
  `aniso = True` (the Oracle; `sample_aniso`) and no point detector or
  point-scored ext tally needs a point inside a ball. With a mesh tally,
  an anisotropic ball is held inside one voxel, because ball/profile.py's
  radial table is for isotropic walks only. Isotropic problems are still
  bit-identical (5 problems, plain and balls).
- **anisocheck**: 20 of 20 (`runs/2026-11-10/v2.1/anisocheck.log`). Every
  case now also runs with exact balls, and 7 problems are repeated without
  mesh tallies so the anisotropic balls keep their full size: sphere P3
  (1.0M balls), nested clipped, cask P1 (3.8M), hexcore histogram,
  tilted tabular (5.2M), curved backward (7.1M), and the cask with only
  the concrete anisotropic. lattice3d is left out because its cells hold
  no ball even when isotropic.
- validate, detcheck (11/11), robust (27 pass + 3 refused of 30) as
  before; 227 tests pass (7 new in `tests/test_aniso_balls_v21.py`); score
  9.9. Plain MC speed unchanged within this machine's noise (`speed.log`).

## v2.0.0 (10 Nov 2026): anisotropic scattering, plain Monte Carlo

Phani's go (10 Nov) for v2 after the plan doc. v2.0 is Stage 2 without
balls: every scattering law OpenMC 0.16 multigroup accepts is read and
sampled exactly as OpenMC samples it (read from its scattdata.cpp,
mgxs.cpp, settings.cpp). Balls stay on in isotropic materials and are
switched off in anisotropic ones; v2.1 brings balls (and interface v2).

- **`core/scattering.py`** (new): Legendre data through OpenMC's tabular
  conversion (33 points by default, `settings.tabular_legendre`
  num_points, negative density set to 0, trapezoid norm, piecewise
  linear sampling), or by rejection when that conversion is disabled;
  histogram and tabular formats; `settings.max_order` truncation. One
  row per material in `Physics.ang`. Also the law's pdf and its Legendre
  moments.
- **Import**: the old "anisotropic is stage 2" refusal is gone; refused
  now are `representation='angle'` and histogram bins below 0.
  `run.py check` prints each material's law and mean cosine.
- **Transport**: an isotropic material samples exactly the same random
  numbers as before, so isotropic problems are bit-identical to v1.6.5
  (5 problems, plain and balls, every tally compared bit by bit).
- **Detectors**: an anisotropic scatter weights the next-event kernel by
  2 pdf(mu). **Filters**: LegendreFilter and scatter-cosine
  SphericalHarmonicsFilter score the law's moments (with a law the
  latter makes the run plain Monte Carlo; all-isotropic stays as before).
- **`checks/aniso.py`, `run.py anisocheck`** (new): 13 of 13 pass
  (`runs/2026-11-10/v2.0/anisocheck.log`). Exact: a beam's mean
  absorption depth 1 / (sigma_t - sigma_s mu_bar) for 7 laws (|z| <= 1.43;
  P3 2.007 +- 0.006 cm against 2.001). Against OpenMC (1M histories
  each): 10 problem/law cases (P1, P3, clipped, backward, histogram,
  tabular, rejection, 9 points, max_order, one anisotropic material in
  the cask with balls in the others: 2.79M balls), the scatter-cosine
  filters, and void point detectors (z +1.34, -0.96).
- **OpenMC bug found**: with histogram data its `scatter` score is
  sigma_s times order/2 (4 bins 2.0x, 12 bins 6.0x; flux and absorption
  right). Proved from its source and measured
  (`runs/2026-11-10/v2.0/openmc_histogram_bug.log`, GAPS.md 2b).
- validate: as v1.6.5 (all pass but the known slab plain-MC depth item,
  chi2/dof 2.156). Tests: 220 pass (23 new in
  `tests/test_scattering_v20.py`).
- `run.py score` 9.9 / 10 as v1.6.5 (detcheck 11 / 11, robust 27 pass
  + 3 refused of 30; `runs/2026-11-10/v2.0/SCORE.md`). Speed: plain MC
  as v1.6.5 (sphere 4.92-5.17 s vs 5.30-5.31, slab 5.47-5.52 vs
  5.84-5.85, cask 8.69-9.21 vs 8.92-8.95; `speed.log`); an anisotropic
  flight costs 3.4% more (552 vs 534 ns).
- `problems/common.scattering_law` gives any problem a law (one for all
  materials, or per material) without touching its isotropic library.

## v1.6.5 (5 Nov 2026): model race seeds 2 and 3, medians

`runs/2026-11-05/race-models/`: seeds 12 and 13 plus `medians.py` and its
log, and a README. The medians confirm seed 1. The cVAE agrees with OpenMC
on all 9 problems (worst |z| 2.5). Where balls are used it is 8-31% faster
than the flow and 18-26% slower than the exact walk. cfm biases sphere
leakage (|z| 3.3 and 3.9).

## v1.6.4 (5 Nov 2026): every model family raced on the reference problems

Phani's ask: a full run of the reference problems with the new model
(cVAE, m1.2) for accuracy and speed. `runs/2026-11-05/race-models/`:
`race_models.py` (checks/race with every family as a contender), seed 1
of 3 (`race_seed11.{log,json}`); seeds 2 and 3 follow. All 9 problems
agree with OpenMC (|z| <= 2.5; cfm sphere leakage z -3.3). cVAE is the
fastest model (6-32% under the flow) but 16-26% slower than the exact
walk wherever balls are used. `superseded/` holds a first start stopped
after the sphere (cfm and cVAE seed 1 sat at R* = 10, no balls).

## v1.6.3 (5 Nov 2026): checks follow the model store

Development's m1.0 (96deaa0, `ball/models/CHANGELOG.md`) moved the flow
behind a model registry; trained flows now live in
`data/models/nflow_seed<k>/`, or in the shared home it copies from.
m1.0 is in the tree, not yet accepted as a model (no gate run yet).

- `checks/gpu_bench.py`, `net_profile.py`, `net_variants.py` find the
  trained flow with `ball.models.store.locate("nflow", 0)` instead of the
  old `flow_seed0/` folder, which no longer exists (they would have
  fallen back to an untrained network). No other change.

## v1.6.2 (5 Nov 2026): interface v1, the model interface spec

Phani's ask (5 Nov): bring the pluggable model pipeline into gmc3d, with
Innovation and Development sharing `3d-gmc` without clashes. Agreed split:
Innovation owns what a learned backend takes in and gives out
(`ball/interface.py`), the exact walk, the data generator, the profile,
`checks/` (incl. the acceptance gate) and the top-level files;
Development owns `ball/models/` (families, encodings, training, fairness,
comparison) and numbers its pushes m1.0, m1.1 ... in
`ball/models/CHANGELOG.md`. An interface change is its own commit tagged
"interface vN"; Innovation then messages Development.

- **`ball/interface.py`** (new), interface v1: input R in [1, 30] mfp;
  outputs mu in (0, 1) and s > R, physical units, no encoding.
  `INTERFACE_VERSION = 1`, `N_IN`, `N_OUT`, `R_MIN`, `R_MAX`, and
  `check_outputs` (counts draws outside the ranges). On 200,000 exact
  walks every draw is inside (mu 0.0075 to 0.99999999940, min s/R
  1.00000013).
- No other code changes; `ball/data.py` is unchanged and the results of
  every check are the same as v1.6.1.

## v1.6.1 (30 Oct 2026): every OpenMC input that is not a physics stage

Phani's ask: implement the 18 plain items of the v1.6.0 gaps list, then
recount the gaps. Every item is checked against OpenMC by the new
`python run.py gapcheck` (20 cases, plain MC and exact balls; all pass,
`runs/2026-10-30/v1.6.1/gapcheck.log`).

- **General scorer** (`core/ext.py`): any tally the fast kernels do not
  take goes here, with every OpenMC filter: Universe, CellInstance,
  Distribcell, MaterialFrom, CellBorn, MeshBorn, MeshMaterial, Particle,
  Polar, Azimuthal, SphericalHarmonics, Mu, MuSurface, Collision, Time,
  SpatialLegendre, Zernike, ZernikeRadial, Legendre, Energyout, Weight;
  two filters of one kind; CellFrom on a track tally; Surface with Mesh;
  MeshSurface on curved meshes and with other filters. Scores nu-scatter,
  inverse-velocity, and the fission and decay scores (0 here, as in
  OpenMC). Collision and analog estimators are accepted (scored by track
  length, warned).
- **Crossing tallies as OpenMC scores them**: a Surface, CellFrom or
  MaterialFrom filter with flux or current is scored at crossings, flux
  by the surface-crossing estimator w/|mu| with OpenMC's grazing rule.
  Lattice walls count as crossings (OpenMC scores every one). Reflective
  walls are scored twice, periodic ones once at the partner plane with
  the weight that arrived.
- **Geometry**: boundary albedo; general `Plane` periodic pairs; horn and
  spindle tori. Rotational periodic walls with the problem on the - side
  of a plane were silently wrong in both codes: now turned the right way.
- **Sources**: MCPL files; 1D and 2D `MeshSource`; `Isotropic` with a
  bias; source times (for Time filters and inverse-velocity).
- **Tally triggers** are honoured: batches run until every trigger is met
  or `max_batches`.
- **Point detectors** with reflective, white and periodic walls (wall
  images) and with any source (its angular density towards the point).
  Checked with the trained network backend too (`detnet.log`).
- **Found and fixed while checking against OpenMC**:
  - MaterialFrom at a vacuum leak or before a reflection is OpenMC's
    material_last (the material before the last cell search, none after a
    collision), not the cell's own.
  - Curved-mesh phi faces: with two phi bins every crossing was filed
    as going up; the way is now read from phi itself.
  - Mu filter: OpenMC never resets the scatter cosine between histories,
    so it bins an unscattered flight by the previous history's last one.
    gmc3d now carries it the same way, from the analog history its
    weighted walk stands for (roulette waits until that history is
    absorbed). Before: up to 9.5 sigma per bin.
  - A store into an array inside `_advance`'s prange body did not reach
    `_track` (numba parfor lowering); moved into a helper call.
  - OpenMC 0.16's `Tabular.evaluate` takes one value at a time.
- **OpenMC errors found** (GAPS.md part 2b): inward crossings of a curved
  mesh face are filed by the next face along the ray; a CylindricalMesh
  with phi range pi/3 in a wedge sums 6% below OpenMC's own cell tally.
- **GAPS.md rewritten**: real gaps only, then what OpenMC refuses too or
  gets wrong, then open checks.
- `checks/robust.py`: the four former refusals (albedo, collision
  estimator, fission score, horn torus) now run and must PASS; the
  refusal test uses a tally derivative and the events score.
- New tests: `tests/test_general_v161.py`; tests that expected refusals
  of now-accepted inputs updated (183 pass).
- The robustness builder cut a horn torus out of its object without
  giving the hole to the background (a region in no cell; OpenMC lost
  particles there, model 13). Fixed in `checks/robust.py`.
- Speed: plain MC 6-7% slower than v1.6.0 with answers bit-identical;
  cause not found (not the general scorer's branches). Cold-cache compile
  54 s -> 115 s. `runs/2026-10-30/v1.6.1/speed.log`, GAPS.md open checks.

## v1.6.0 (30 Oct 2026): hardening, point detectors, one score out of 10

- **Point detectors** (`problems/common.point_detector`, `core/detectors.py`):
  the flux at a point by the next-event estimator, scored from every
  isotropic emission (source sites and scatters). Inside a ball, the
  unseen scatters are scored as sigma_s x the ball's flux at 4 profile
  points, plus a term for the straight flights that a ball's conditioned
  walk leaves out. Without that term the detector check measured balls
  2-6% low; with it, all four detectors agree with plain MC (z -1.30 to
  +0.11). An optional averaging radius r0 gives finite variance inside a
  scattering material. Refused with reflective, white or periodic walls,
  and with non-isotropic sources. `run.py detcheck` checks detectors
  three ways: exactly (rel err 5e-10), balls against plain MC, and
  against OpenMC's small void spheres.
- **Slab reference at 20M histories**, as Phani asked. It does not
  settle the slab depth item, but it found its cause.
  - The 60 depth bins are strongly correlated: neighbours +0.88, about 6
    independent bins in all.
  - So their chi2/dof spreads 3.5x wider than the pass rule assumes: of
    40 runs of a correct code, 7 fail it.
  - The run that fails (plain MC, seed 1) has a total flux 2.15 sigma low,
    and every bin moves with it. It fails against the 20M reference too
    (chi2/dof 2.16).
  - A rule that counts effective bins passes it (p 0.04). That rule is
    left for Phani to decide.
  - validate now uses the 10x reference for the slab (`REF_SCALE`).
- **Robustness suite** (`run.py robust`, V1-ROB-4): 30 models built at
  random within what gmc3d accepts, each against OpenMC (plain MC and
  exact balls). Each must pass or be refused cleanly; about one in seven
  is given an unsupported feature to test the refusal.
- **Edge cases** (`tests/test_edge_cases_v160.py`, V1-ROB-5). Found and
  fixed: a source site exactly on a surface lay in no cell and was redrawn
  for ever (a hang). It is now moved 1e-8 cm along its direction, which is
  how OpenMC decides the side. Also new: a source no site of which can be
  accepted now stops after about 20 rounds instead of 1000, and a model in
  which nothing absorbs and nothing leaks is refused.
- **`run.py check model.py`** (V1-ROB-6): every refusal and warning of a
  model at once, without running it. It takes a problem name, a Python
  file, model.xml or a folder. It works around OpenMC 0.16 dropping
  `<macroscopic>` when it reads materials from XML.
- **`ACCEPTED_INPUTS.md`** (V1-ROB-7): every OpenMC class and setting,
  marked used, refused or warned.
- **`run.py score`**: one number out of 10, with these weights:
  - unit tests 2;
  - tallies against OpenMC 4;
  - exact answers 1;
  - random models 2;
  - point detectors 1.

  It is printed with the failing checks and with what it does not
  measure: the trained network, speed, and physics beyond Stage 1.

## v1.5.0 (30 Oct 2026): curved meshes, currents on any surface, mesh surface currents

- **`CylindricalMesh` and `SphericalMesh`** in a `MeshFilter`, with any
  origin, partial angle ranges and other filters alongside. A flight is
  cut wherever it meets an edge surface (cylinder or sphere for r,
  half-plane for phi, cone or the plane z = 0 for theta, plane for z) and
  each piece goes to the bin of its midpoint. Checked against brute force
  along random flights (to the sampling step) and against OpenMC. Balls
  work unchanged ('profile' rule: bin of a point; 'cap' rule: distance to
  the nearest edge surface). Curved meshes are not written to `.vtk` or
  drawn as flux maps.
- **Currents on every surface**: reflective and white walls (exactly 0, as
  OpenMC scores them), periodic walls (scored once, on the wall the
  particle comes back through), and surfaces inside universes and
  lattices (signed in the surface's own frame; counted only if the cell at
  that level changes). The geometry descent now also returns the level
  and the sign of the surface a flight stops at.
- **`MeshSurfaceFilter`** on Regular (1D, 2D, 3D) and Rectilinear meshes,
  alone: the weight through each element face, in and out, in OpenMC's bin
  order. Faces lying on surfaces and on vacuum boundaries are counted as
  OpenMC counts them. While one is present, balls are shrunk to stay
  inside one voxel of its mesh, whatever the mesh rule.
- **Three OpenMC 0.16 errors found** (`runs/2026-10-30/v1.5/openmc_bugs.log`):
  theta bins at pi/2, currents inside moved universe frames, and the sign
  on rotational walls. In each case gmc3d agrees with an exact or
  conservation answer, and those tallies are checked by unit tests rather
  than against OpenMC.
- **Speed: plain MC as fast as v1.4.1** (200k x 4 histories, best of 3:
  cask 19.73 s against 20.0-20.3 s, hexcore 28.32 s against 27.2-28.9 s).
  Found on the way: every array added to the tally `Layout` or to
  `Geometry` is paid for on every call that receives it. The first build
  of v1.5 put seven new arrays in `Layout` and made cask 43% slower with
  no curved mesh in it. The new mesh data now lives in its own
  `tallies.MeshExtra`, passed only to the new kernels; the periodic
  partner sits in `surf_shift`'s fourth column; and the descent works out
  a crossing's sign only when a current tally exists.
- New test tallies: sphere (spherical mesh), curved (cylindrical mesh,
  alone and x material; spherical mesh), lattice3d (reflective walls),
  tilted (periodic and white walls, partial currents), slab (1D mesh
  surface), nested (3D rectilinear mesh surface). `tests/test_tallies_v150.py`.

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
