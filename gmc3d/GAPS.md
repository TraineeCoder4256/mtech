# What gmc3d leaves out (the gaps list)

Every version adds what it left out here, and removes what it fixed. Each
gap says which version left it, how it shows up for a user, and a rough
size (S = a day or less, M = a few days, L = a week or more). "Refused"
means `core/openmc_import.py` stops with an `Unsupported` error naming the
feature. "Silent" means the model runs anyway and the feature is ignored or
approximated, so it is the more dangerous kind.

Last updated: v1.1.0 (27 Oct 2026).

## Geometry (left in v1.0)

| gap | how it shows | size | note |
|---|---|---|---|
| Cylinders and cones on a general axis (`Cylinder`, `Cone` with any direction) | refused | S | only X/Y/Z-axis ones today |
| `Quadric` (general second-order surface) | refused | M | needs a general distance-to-surface bound for the balls |
| `XTorus` / `YTorus` / `ZTorus` | refused | M | quartic ray intersection; distance bound harder |
| `HexLattice` | refused | M | hexagonal element lookup and walls |
| White and periodic boundaries | refused | S/M | white = isotropic reflection; periodic needs paired surfaces |
| Distributed materials / cell instances (one cell, many materials) | refused | M | |
| DAGMC / CAD geometry | refused | L | triangle surface meshes; distance field needs a BVH |
| Cell tallies on cells filled with a universe or lattice | refused | S | material and void cells only |

## Physics (left in v1.0, by stage)

| gap | how it shows | size | note |
|---|---|---|---|
| Anisotropic scattering | refused | L | Stage 2 = v2.0 |
| More than one energy group | refused | L | Stage 3 |
| Fission / k-eff | refused | L | Stage 4; needs fission sites born inside balls (the v1.1 profile is half of that) |
| Continuous energy | refused | L | Stage 5 |
| Nuclides inside materials (only one `Macroscopic` per material) | refused | M | |
| Photons, coupled neutron-gamma | refused (multi-group only) | L | gammas born inside balls need the same point draw as v1.1 |
| OpenMC run settings we never read: survival biasing, weight windows, energy/weight cutoffs, `max_lost_particles`, temperatures | **silent** | S to refuse them / M+ to honour them | we always use implicit capture with roulette at 0.25 |

## Source (left in v1.0)

| gap | how it shows | size | note |
|---|---|---|---|
| Spaces other than `Point` and `Box` (`CartesianIndependent`, `SphericalIndependent`, `CylindricalIndependent`, `MeshSpatial`, `PointCloud`) | refused | S each | |
| Angles other than `Isotropic` and `Monodirectional` (`PolarAzimuthal`) | refused | S | |
| `FileSource`, `CompiledSource`, `MeshSource` | refused | M | |
| Source `constraints` / domain rejection (born only in some cells) | **silent** | S | today: born anywhere in the box, rejected only if outside the geometry |
| Source energy | ignored (one group) | - | matters from Stage 3 |

## Tallies (left in v1.0 and v1.1)

| gap | how it shows | size | note |
|---|---|---|---|
| More than one filter per tally (mesh x energy, mesh x cell, ...) | refused | M | needed at Stage 3 |
| Scores other than flux, absorption, current (e.g. `total`, `scatter`, `events`, `heating`, reaction rates) | refused | S each | |
| Tally `estimator` setting (`tracklength` / `collision` / `analog`) | **silent** | S | we always use track length |
| Tally nuclides, triggers, derivatives | **silent** | S to refuse | |
| Point detectors (flux at a point) | not available | M | Kobayashi benchmarks report point fluxes; use small voxels with `cap` for now |
| Leakage is total only in `Results.leakage`; per-surface needs a `SurfaceFilter` tally | - | - | per-surface current added in v1.1 |
| Surface current on reflective surfaces, or on surfaces inside a filled universe | refused (v1.1) | S/M | |
| Partial currents (in / out separately, `CellFromFilter`) | refused (v1.1) | M | needs two-filter tallies |
| 1D and 2D `RegularMesh` | refused (v1.1) | S | |
| `RectilinearMesh` | refused (v1.1) | S | |
| `CylindricalMesh`, `SphericalMesh` | refused (v1.1) | M | curved bin walls |
| `UnstructuredMesh` (tetrahedral) | refused (v1.1) | L | needs a point-location structure and ray-through-tet tracking |
| `MeshSurfaceFilter` (currents through voxel faces) | refused (v1.1) | M | |

## Ball method limits (approximations and ranges)

| gap | how it shows | size | note |
|---|---|---|---|
| Balls bigger than 30 mfp | **silent**: balls shrunk to 30 mfp | S | network and profile table both stop at R = 30 |
| R* below 1 mfp with a mesh tally | refused (v1.1) | S | profile table starts at R = 1 |
| Mesh flux in balls is exact only on average | by design (v1.1) | - | extra noise is measured in `runs/2026-10-27/MESH.md`; profile table error checked by `run.py meshcheck` |
| Mesh points cost about 0.1 us each, up to 16 per ball | by design (v1.1) | S | could adapt K to the mesh's real dimensionality (a 1D slab mesh needs fewer) |
| Network accuracy in its rare short-path tail (R = 29, kappa = 0.1: weight +73%, z 3.1) | **silent** | M | from 5 Oct ballcheck; the six test problems barely use it |
| Lookup-table backend's fixed sample (~1.2% leakage bias in strongly absorbing balls) | **silent** | - | table kept only as a reference, per Phani |
| Network runs on GPU only for the draws; transport stays on CPU | by design | L | the transport loop is now the ceiling (`runs/2026-10-14`) |

## Open checks (not gaps in the code, but unfinished)

| item | since | note |
|---|---|---|
| Nested leakage z about -3.2 for plain MC and exact balls (seed 101); +0.04 with seed 105 | v1.1 validate | same pattern on 13 Oct; recheck at 10M+ histories |
| No race against OpenMC with weight windows | v1.0 | the fair comparison for shielding claims |
