# gmc3d changelog

Every push to `3d-gmc` gets a version, written in `VERSION` and tagged
`v<version>` in git. The first number is the stage (2.0 = anisotropic
scattering). Within version 1 the middle number is the area (1.1 mesh
tallies and the easy tally items, 1.2 sources, 1.3 multi-filter tallies,
1.4 geometry, 1.5 the medium tally items; renumbered 29 Oct 2026); the
last counts pushes within it.

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
