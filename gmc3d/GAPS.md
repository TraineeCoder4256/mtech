# What gmc3d leaves out (the gaps list)

Every version adds what it left out here and removes what it fixed.

Columns:
- **shows as**: "refused" means `core/openmc_import.py` stops with an
  `Unsupported` error naming the feature. "SILENT" means the model runs
  anyway and the feature is ignored or approximated, which is the more
  dangerous kind.
- **size**: S = a day or less, M = a few days, L = a week or more.
- **model change?**: whether fixing it needs the trained network to change
  (retraining, new inputs or a new architecture). The network sees only the
  ball's radius R and returns (mu, s), so geometry, sources, tallies and
  boundaries normally stay outside it. Gaps marked YES are on hold until
  the model work they belong to.
- **plan**: the push in `V1_PLAN.md` that closes it, or "hold" / "later".

Last updated: v1.1.2 (27 Oct 2026).

## Geometry (left in v1.0)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| `Cylinder` and `Cone` on any axis | refused | S | no: only the distance field changes | v1.1.6 |
| White boundary | refused | S | no: balls never cross a surface | v1.1.6 |
| Periodic boundary, translational | refused | S | no | v1.1.6 |
| Periodic boundary, rotational | refused | M | no | v1.1.7 |
| `HexLattice` | refused | M | no: only the distance field and lookup change | v1.1.7 |
| `Quadric` | refused | M | no: a guaranteed lower bound on the distance just makes balls smaller | v1.1.7 |
| `XTorus` / `YTorus` / `ZTorus` | refused | M | no (same lower-bound idea for elliptical tubes) | v1.1.7 |
| Distributed materials (a material per cell instance) | refused | M | no: absorption is applied outside the network | v1.1.7 |
| Cell tallies on cells filled with a universe or lattice | refused | S | no | v1.1.5 |
| DAGMC / CAD geometry | refused | L | no, but large engineering (BVH distance field) | later |

## Source (left in v1.0)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| `CartesianIndependent`, `SphericalIndependent`, `CylindricalIndependent` | refused | S | no: sources only set where particles start | v1.1.4 |
| `PointCloud`, `MeshSpatial` | refused | S | no | v1.1.4 |
| `PolarAzimuthal` angle | refused | S | no | v1.1.4 |
| `FileSource`, `MeshSource` | refused | M | no | v1.1.4 |
| Source `constraints` (born only in given cells, materials, universes) | **SILENT** | S | no | v1.1.4 |
| `CompiledSource` (user C++ code) | refused | L | no, but it is someone else's compiled code | stays refused |
| Source energy | ignored (one group) | - | YES at Stage 3 (energy becomes a network input) | Stage 3 |

## Tallies (left in v1.0 and v1.1)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| Tally `estimator` other than track length | **SILENT** | S to refuse | no to refuse; a collision estimator inside balls would be a method question | v1.1.3 (refuse) |
| Tally nuclides, triggers, derivatives | **SILENT** | S to refuse | no | v1.1.3 (refuse) |
| Scores `total`, `scatter` | refused | S | no: same track-length estimator | v1.1.3 |
| Other scores (`events`, `heating`, reaction rates by nuclide) | refused | S-M each | no, but most need data a one-group library lacks | later |
| 1D and 2D `RegularMesh`, `RectilinearMesh` | refused | S | no: the v1.1 ball rule only asks which bin holds a point | v1.1.3 |
| More than one filter per tally | refused | M | no | v1.1.5 |
| `EnergyFilter` | refused | S | no now (one bin); real bins at Stage 3 | v1.1.5 |
| Partial currents (`CellFromFilter` x `SurfaceFilter`) | refused | S after multi-filter | no | v1.1.5 |
| `CylindricalMesh`, `SphericalMesh` | refused | M | no | v1.1.8 |
| Surface current on reflective surfaces / inside a filled universe | refused | S/M | no | v1.1.8 |
| `MeshSurfaceFilter` (currents through voxel faces) | refused | M | no IF balls are shrunk to one voxel when it is present (exact, less speed-up); YES if balls stay full-size, since (mu, s) cannot say which faces the walk crossed | v1.1.8 (shrink) |
| Point detectors (flux at a point; Kobayashi reports these) | not available | M | no under the idea of using v1.1 profile points as stand-in collision sites; that idea is a method decision for Phani | Phani decides |
| `UnstructuredMesh` (tetrahedral) | refused | L | no, but large engineering | later |

## Run settings (left in v1.0)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| OpenMC settings we don't read: survival biasing, weight windows, cutoffs, `max_lost_particles`, temperatures | **SILENT** | S to warn | no. They change noise and speed only, not the expected answer, so a warning fits | v1.1.3 (warn) |
| Honouring weight windows | - | M+ | no at ball exits; splitting INSIDE a ball is impossible without seeing the walk | later |

## Physics (by stage)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| Anisotropic scattering | refused | L | **YES**: the scattering law becomes a network input | v2.0 |
| More than one energy group | refused | L | **YES** | Stage 3 |
| Fission / k-eff | refused | L | **YES** (fission sites inside balls; the v1.1 profile covers where, not how many) | Stage 4 |
| Continuous energy | refused | L | **YES** | Stage 5 |
| Nuclides inside materials | refused | M | no at one group (sum to macroscopic), YES once energy enters | Stage 3 |
| Photons, coupled neutron-gamma | refused | L | **YES** (a gamma model, or the scattering law as input) | later |

## Ball method limits

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| Balls bigger than 30 mfp | **SILENT**: shrunk to 30 mfp (exact, slower) | M | **YES**: retrain the network on a wider R range (the profile table just needs rebuilding) | hold |
| Network's rare short-path tail (R = 29, kappa = 0.1: weight +73%, z 3.1, 5 Oct ballcheck) | **SILENT** | M | **YES**: retraining or a different architecture | hold |
| R* below 1 mfp with a mesh tally | refused (v1.1) | S | no: rebuild the profile table from a smaller R | later |
| Mesh flux in balls exact only on average | by design (v1.1) | - | no | - |
| Mesh points cost about 0.1 us each, up to 16 per ball | by design (v1.1) | S | no: adapt the point count to the mesh's real dimensions | later |
| Lookup-table backend's fixed sample (~1.2% leakage bias in strongly absorbing balls) | **SILENT** | - | the table is not the model; kept only as a reference | - |
| Network on GPU only for the draws; transport stays on CPU | by design | L | no | later |

## Open checks (unfinished, not missing features)

| item | since | note |
|---|---|---|
| Nested leakage z about -3.2 for plain MC and exact balls (seed 101); +0.04 with seed 105 | v1.1 validate | same pattern on 13 Oct; recheck at 10M+ histories |
| No race against OpenMC with weight windows | v1.0 | the fair comparison for shielding claims |
