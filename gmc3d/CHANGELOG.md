# gmc3d changelog

Every push to `3d-gmc` gets a version, written in `VERSION` and tagged
`v<version>` in git. The middle number counts updates within a stage
(1.1 = mesh tallies, still Stage 1); the first number is the stage
(2.0 = anisotropic scattering); the last counts pushes within an update.

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
