# 3D GMC Version 1: Software Requirements Specification

Oct 29, 2026 · @Phani

## 1. Purpose and scope

Version 1 is the complete transport shell around the ball model: when it is done, any one-group, isotropic, fixed-source OpenMC model that a shielding benchmark is likely to use either runs and matches OpenMC, or is refused with a message naming the feature. After v1, work moves to models and physics only (v2.0 starts with anisotropic scattering).

This SRS replaces nothing. The Stage 1 SRS still holds for the core ball method; this document covers version 1 as a whole: what v1.0 to v1.1.2 already deliver, what v1.1.3 to v1.1.8 must add, and what stays outside v1.

**Version 1 is done when all of these hold:**

1. Every requirement in sections 3 to 6 is met and its acceptance test passes.
2. `GAPS.md` has no row marked S or M under source, geometry, tallies or run settings, except rows listed in section 7.
3. No input setting is ignored silently: each one is used, refused, or named in a warning.
4. `python run.py validate` passes on every test problem, including the new ones added for v1.1.3 to v1.1.8.

**Versioning rule (Phani, 27 Oct 2026).** Every push carries a version number in `VERSION` and `CHANGELOG.md`, never in a branch name. v1.0 is the Stage 1 base (commit 177c5db). Renumbered 29 Oct 2026: v1.1.x is the mesh work plus the easy tally items, then each area gets its own minor version (v1.2 sources, v1.3 multi-filter tallies, v1.4 geometry, v1.5 medium tally items), with fixes inside an area as v1.2.1 and so on. v2.0 begins with anisotropic scattering.

**Where the work lives.** Branch `3d-gmc` of TraineeCoder4256/mtech, folder `gmc3d/`. No pull request, and main is never touched.

## 2. System overview

The user hands gmc3d an ordinary OpenMC Python model. gmc3d reads it, checks every feature, and runs its own transport, in which most of each particle's path is replaced by ball steps.

**The ball step.** After each scatter, the distance field gives the largest ball around the particle that holds one material. If its radius R is at least R\* mfp, a backend returns where the particle leaves the ball (the exit direction mu) and how far it walked inside (s). Absorption is applied exactly as a weight factor exp(-sigma\_a s). Smaller balls, and every flight that crosses a surface, are tracked as plain Monte Carlo.

**Backends.** `oracle` (an exact walk inside the ball, the reference), `network` (the trained conditional neural spline flow, 146,554 weights, R from 1 to 30 mfp) and `table` (kept only as a reference; never used in comparisons).

**What the user gets.** The same tallies the OpenMC model asked for (cell, material and mesh flux and absorption, surface currents, leakage), each with a standard error, plus timing and counters. `checks/` compares any result with an OpenMC run of the same model.

**The boundary that v1 fixes.** Everything outside the ball step (reading the input, geometry, sources, boundaries, tallies) is the transport shell. The network only ever sees R and returns (mu, s), so the shell can grow without retraining. Version 1 finishes the shell; version 2 onwards changes the model.

&#91;embedded content: gmc3d data flow · shell vs model\]

Everything in the shaded shell is version 1 work; only the highlighted ball model changes in version 2 and later.

## 3. Built so far (v1.0 to v1.1.2)

These requirements are met and validated against OpenMC 0.16. Evidence is in `gmc3d/runs/2026-10-27/` (MESH.md, validate.log, maps.log) and in the Stage 1 report.

| ID | requirement | version | evidence |
| --- | --- | --- | --- |
| V1-IN-1 | Read an OpenMC Python model directly; stop with an `Unsupported` error naming any feature not handled | v1.0 | `core/openmc_import.py` |
| V1-IN-2 | Physics: multi-group mode, 1 group, one macroscopic per material, isotropic scattering, no fission, total = absorption + scatter | v1.0 | refused otherwise |
| V1-GEO-1 | Surfaces: X/Y/Z planes, general plane, sphere, X/Y/Z cylinders, X/Y/Z cones; regions with intersection, union, complement | v1.0 | locate matches OpenMC on random points |
| V1-GEO-2 | Universes, cells filled with universes, RectLattice; vacuum, transmission and reflective boundaries | v1.0 | nested and lattice3d problems |
| V1-SRC-1 | IndependentSource with Point or Box space, Isotropic or Monodirectional angle | v1.0 | all 6 problems |
| V1-TAL-1 | Flux and absorption with no filter, CellFilter (material and void cells), MaterialFilter, 3D RegularMesh | v1.0 | validate |
| V1-TAL-2 | Mesh flux inside balls exact on average ('profile' rule: path length drawn exactly, radius from the Q(r given l, R) table, split over up to 16 points) | v1.1.0 | network mesh chi2/dof 0.57 to 1.11 on 6 problems (v1.0 rule: up to 70,593) |
| V1-TAL-3 | Surface current through root-universe vacuum and transmission surfaces, OpenMC's sign convention | v1.1.0 | slab `sides`: network max abs z 0.48 |
| V1-VAL-1 | `run.py validate` gives PASS or FAIL: fail if any tally chi2 has p below 0.001 or any single value has abs z above 3.29; OpenMC reference in 400 batches | v1.1.0 | all 6 PASS |
| V1-VAL-2 | `run.py maps` draws our flux map, OpenMC's and the z map for every problem | v1.1.0 | 6 PNGs |
| V1-DOC-1 | `VERSION`, `CHANGELOG.md` and `GAPS.md` (with a 'model change?' column) updated on every push | v1.1.1, v1.1.2 | in the repo |

The 'centre' mesh rule was removed in v1.1.0. The 'cap' rule (shrink balls to fit one voxel) stays as an option.

## 4. Requirements still to build (v1.1.3 to v1.5)

These come from `V1_PLAN.md`. Phani gave the go on 29 Oct 2026. Done so far: the four v1.1.3 rows (commit 0764f63), the three v1.2 source rows (v1.2.0, commit 0013942), the four v1.3 tally rows (V1-TAL-7 to V1-TAL-10, v1.3.0, commit 052abb5), the two v1.4.0 geometry rows (V1-GEO-3 and V1-GEO-4, commit 3db14a2), the three v1.4.1 geometry rows (V1-GEO-5 to V1-GEO-7, commit 7b5ed2f; results in `gmc3d/runs/2026-10-29/v1.4.1/V1.4.1.md`) and the three v1.5 tally rows (V1-TAL-11 to V1-TAL-13, v1.5.0, commit ef479a3; results in `gmc3d/runs/2026-10-30/v1.5/V1.5.md`). All planned version 1 rows are now built. v1.6.0 (commit 641c6c4) added point detectors, the slab reference rerun and the hardening rows of section 5; results in gmc3d/runs/2026-10-30/v1.6/V1.6.md. None needs the network to change. Each row is checked against OpenMC on a problem that uses the feature.

| ID | requirement | push | acceptance test |
| --- | --- | --- | --- |
| V1-RUN-1 | Warn, naming each one, about OpenMC settings that are not used: survival biasing, weight windows, cutoffs, temperatures, `max_lost_particles`, anything else not read. Refuse eigenvalue runs and photon transport | v1.1.3 | unit test per setting |
| V1-TAL-4 | Refuse tally estimators other than track length, nuclides and derivatives; warn about triggers; refuse MeshSurfaceFilter and MeshBornFilter (until v1.1.3 these passed as ordinary meshes) | v1.1.3 | unit test |
| V1-TAL-5 | Scores `total` and `scatter` | v1.1.3 | vs OpenMC, slab and cask |
| V1-TAL-6 | 1D and 2D RegularMesh, RectilinearMesh | v1.1.3 | vs OpenMC, new meshes on slab and cask |
| V1-SRC-2 | Sample sources from the model's own OpenMC distribution objects: CartesianIndependent, SphericalIndependent, CylindricalIndependent, PointCloud, MeshSpatial, PolarAzimuthal | v1.2.0 (done) | sampled sites match the object's own samples; one solve per type vs OpenMC: all PASS |
| V1-SRC-3 | FileSource and MeshSource | v1.2.0 (done) | vs OpenMC: PASS |
| V1-SRC-4 | Source `constraints` (born only in given cells, materials, universes), by rejection | v1.2.0 (done) | vs OpenMC, cells three levels down: PASS (closes a silent gap) |
| V1-TAL-7 | Several filters per tally (mesh x cell, mesh x material, cell x surface, and so on) | v1.3.0 (done) | vs OpenMC, nested (mesh x material, cell x energy): PASS |
| V1-TAL-8 | EnergyFilter accepted (one bin at one group) | v1.3.0 (done) | unit test, and vs OpenMC on nested: PASS |
| V1-TAL-9 | Partial currents (CellFromFilter x SurfaceFilter) | v1.3.0 (done) | vs OpenMC, slab and nested: PASS |
| V1-TAL-10 | Cell tallies on cells filled with a universe or lattice | v1.3.0 (done) | vs OpenMC, nested (3 levels on one path): PASS |
| V1-GEO-3 | Cylinders and cones on any axis | v1.4.0 (done) | locate on 3,000 random points matches OpenMC; tilted problem vs OpenMC: PASS |
| V1-GEO-4 | White boundary and translational periodic boundary | v1.4.0 (done) | vs OpenMC, tilted problem (periodic x, white y): PASS |
| V1-GEO-5 | HexLattice (both orientations, 2D and 3D) | v1.4.1 (done) | vs OpenMC, hexcore (both orientations, tiled outer universe): PASS |
| V1-GEO-6 | Quadric and torus surfaces, with a guaranteed lower bound on distance where no formula exists | v1.4.1 (done) | vs OpenMC, sector (elliptical ZTorus, XTorus, tilted ellipsoid quadric): PASS |
| V1-GEO-7 | Rotational periodic boundary and distributed materials | v1.4.1 (done) | vs OpenMC, sector (90 degree rotational quarter, nine distributed pins): PASS |
| V1-TAL-11 | CylindricalMesh and SphericalMesh | v1.5.0 (done) | vs OpenMC |
| V1-TAL-12 | Surface currents on reflective surfaces and on surfaces inside a filled universe | v1.5.0 (done) | vs OpenMC |
| V1-TAL-13 | MeshSurfaceFilter, with balls shrunk to one voxel while it is present | v1.5.0 (done) | vs OpenMC |

Order: silent gaps first (safety), then sources and multi-filter tallies (the benchmarks need them), then geometry.

## 5. Robustness: no surprises in a demonstration

The promise for the end demonstration: any input either runs correctly or stops at once with a message saying which feature is not handled. It never crashes mid-run and never gives a quiet wrong answer.

| ID | requirement | status |
| --- | --- | --- |
| V1-ROB-1 | Every OpenMC feature is used, refused with `Unsupported` naming it, or warned about. Nothing is ignored silently | met for refusals; the silent rows close in v1.1.3 and v1.1.4 |
| V1-ROB-2 | Refusals happen while reading the model, before any transport starts | met |
| V1-ROB-3 | Lost particles (a point no cell claims) are counted and reported, never a crash | met |
| V1-ROB-4 | Robustness suite: randomly built models (shapes, lattices, boundaries, sources, tallies within the accepted set) run against OpenMC; each must PASS or be refused cleanly | v1.6.0 (done): 30 random models, 25 PASS, 5 refused cleanly |
| V1-ROB-5 | Edge-case tests: empty tallies, a source on a surface, a ball touching a boundary, zero-volume cells, very thin and very thick materials | v1.6.0 (done): 10 tests; fixed a source on a surface that hung |
| V1-ROB-6 | `python run.py check model.py`: reads a user's model and lists what will be used, refused and warned, without running it | v1.6.0 (done) |
| V1-ROB-7 | An 'accepted inputs' page listing every OpenMC class and setting gmc3d handles | v1.6.0 (done): ACCEPTED\_INPUTS.md |

What this cannot promise: inputs outside the v1 physics (more groups, anisotropy, fission) are refused, not run. That is correct behaviour, not a crash.

## 6. Non-functional requirements and acceptance

| ID | requirement |
| --- | --- |
| V1-NF-1 | Correctness first: every tally is unbiased. A feature that cannot be made exact on average inside a ball shrinks the balls there instead (as `cap` does) |
| V1-NF-2 | The network is never changed by v1 work. The trained model and `data/ball/profile.npz` are reused as they are |
| V1-NF-3 | Each push is a v1.1.x version with `VERSION`, `CHANGELOG.md` and `GAPS.md` updated; the gaps it leaves out are named in the report to Phani |
| V1-NF-4 | Code stays in `gmc3d/` with the existing layout (core, ball, checks, problems, tests); gmc2d is copied from, never imported |
| V1-NF-5 | Results are reproducible from a seed. OpenMC references use seed + 100 |
| V1-NF-6 | Speed is reported, not required: CPU network timings always appear with the GPU projection beside them; the lookup table is never used in comparisons |
| V1-NF-7 | Runs on the project's environment: Python with OpenMC 0.16, numba, torch, scipy, matplotlib |

**Acceptance for each push:**

1. All unit tests pass (178 at v1.6.0). From v1.6.0, python run.py score turns the unit tests, validate, the random models and the detector checks into one score out of 10 (9.9 at v1.6.0).
2. A test problem that uses the new feature is compared with OpenMC at 2M histories, OpenMC in 400 batches.
3. Pass rule as in V1-VAL-1: no tally with chi2 p below 0.001, no single value with abs z above 3.29.
4. `run.py validate` still passes on all earlier problems, so nothing already built gets worse.
5. Report to Phani with what was run, the raw numbers, the diagnosis and the limits.

**Acceptance for version 1 as a whole:** the four conditions in section 1, and the robustness suite (V1-ROB-4) passing.

## 7. Out of scope for version 1

These stay refused (or work as described) at the end of v1. The first group needs the model to change and waits for the model work; the second is large engineering that no planned benchmark needs.

| item | why not in v1 | when |
| --- | --- | --- |
| Anisotropic scattering | the scattering law becomes a network input | v2.0 |
| More than one energy group, nuclides inside materials | energy becomes a network input | Stage 3 (MTech goal) |
| Fission and k-eff | fission sites inside balls need the model | Stage 4 (stretch) |
| Continuous energy | model change | Stage 5 (future work) |
| Photons, coupled neutron-gamma | a gamma model | later |
| Balls over 30 mfp | shrunk to 30 mfp today (exact, slower); retrain on a wider R range | hold |
| Network's rare short-path tail (R = 29, kappa = 0.1) | retraining or a new architecture | hold |
| DAGMC / CAD geometry | large engineering (distance field over a mesh) | later |
| UnstructuredMesh (tetrahedral) | large engineering | later |
| CompiledSource | user C++ code | stays refused |
| Honouring weight windows inside balls | splitting needs the walk inside the ball | later (warned in v1.1.3) |
| Scores beyond flux, absorption, total, scatter, current | need data a one-group library lacks | later |
| Network on GPU end to end | transport stays on CPU | later |

The full list, kept current with each push, is `gmc3d/GAPS.md`.

## 8. Open decisions for Phani

- [x] Go for v1.1.3 (given 29 Oct 2026).
- [x] Add the hardening push v1.6 (V1-ROB-4 to V1-ROB-7) to version 1 (given 30 Oct 2026; done in v1.6.0, commit 641c6c4).
- [x] Point detectors: profile points as stand-in collision sites (chosen 30 Oct 2026; built in v1.6.0, matches plain MC and OpenMC).
- [x] Slab depth tallies: adopt a pass rule that counts correlated bins (effective degrees of freedom), since neighbouring depth bins are 0.88 correlated (about 6 independent of 60). Recommended; it would also cover reference bins with almost no counts.
- [x] Nested leakage z of about -3.2: rechecked at 10M histories in v1.2.0, agrees with OpenMC (leakage z +0.79). Closed.
- [x] Push tags v1.0 to v1.2.0 from your machine once all versions are done (the cloud gateway refuses tag pushes; the version to commit list is kept).
