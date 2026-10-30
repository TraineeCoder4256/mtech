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

Last updated: v1.6.0 (30 Oct 2026).

## Geometry (left in v1.0 to v1.4.1)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| Periodic boundary on a non-plane surface | refused (v1.4.0), as in OpenMC | - | no | stays refused |
| Periodic pairing of general `Plane`s without `periodic_surface` set | refused (v1.4.0); X/Y/Z planes pair automatically, as in OpenMC | S | no | later |
| Boundary `albedo` other than 1 | refused (v1.4.0) | S | no: a weight factor at the wall | later |
| Torus whose tube reaches the axis (c >= a: horn or spindle torus) | refused (v1.4.1) | S | no | later |
| Rotational periodic walls that do not contain the z axis | refused (v1.4.1), as in OpenMC | - | no | stays refused |
| Rotational periodic walls with the problem on the - side of a plane | **SILENT**, as in OpenMC: both codes assume the + side of both planes and turn the wrong way otherwise | S | no | later (a check on import) |
| Distributed materials with a void entry (`None` in the fill list) | built (v1.4.1), checked only by unit test: OpenMC 0.16 stops on one in multigroup mode (measured) | - | no | when OpenMC handles it |
| Distributed materials inside a lattice's outer universe | refused (v1.4.1): OpenMC does not number instances there | - | no | stays refused |
| Distributed temperatures (a temperature per cell instance) | not read (one group, one temperature) | - | YES once energy and temperature enter | Stage 3+ |
| Quadric nearest distance is a lower bound, not exact | by design (v1.4.1): balls near a quadric are smaller than they could be (exact, slower) | M | no | later, if a quadric-heavy benchmark needs the speed |
| Torus flights cost about 0.4 us (a sphere 0.03 us) | by design (v1.4.1): a bracketed quartic root | S | no | later |
| DAGMC / CAD geometry | refused | L | no, but large engineering (BVH distance field) | later |

## Source (what v1.2 leaves)

Since v1.2 every OpenMC source kind is read except the rows below
(`core/sources.py`).

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| `CompiledSource` (user C++ code) | refused | L | no, but it is someone else's compiled code | stays refused |
| `FileSource` in MCPL format (`.mcpl`) | refused (v1.2) | S | no | later |
| `MeshSpatial` / `MeshSource` on an `UnstructuredMesh`, or a 1D/2D `RegularMesh` | refused (v1.2) | M (unstructured) / S | no | later |
| `Isotropic` angle with a `bias` | refused (v1.2) | S | no | later |
| Source `energy_bounds`, `time_bounds` constraints | warned, not applied (v1.2); OpenMC 0.16 in multigroup mode does not apply them either (measured) | - | no; energy bounds matter at Stage 3 | Stage 3 |
| `rejection_strategy: 'kill'` | warned, acts as 'resample' (v1.2), as OpenMC 0.16 does for domains (measured) | S | no | later, if OpenMC changes |
| `PointCloud` / `MeshSpatial` `bias` | built (v1.2), checked only by unit test: OpenMC 0.16 refuses the XML its own Python writes for it, so no OpenMC comparison | - | no | when OpenMC fixes it |
| Source energy | ignored (one group) | - | YES at Stage 3 (energy becomes a network input) | Stage 3 |

## Tallies (left in v1.0 to v1.6)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| Tally `estimator` other than track length | refused (v1.1.3) | M | no; a collision estimator inside balls would be a method question | later |
| Tally triggers honoured (stop when precise enough) | warned (v1.1.3) | S | no | later |
| `MeshBornFilter` (bins by birth place) | refused (v1.1.3) | S | no | later |
| Tally derivatives (differential tallies) | refused (v1.1.3) | M | no at one group; the sensitivity of a ball's (mu, s) to a cross section is a research question | later |
| Other scores (`events`, `heating`, reaction rates by nuclide) | refused | S-M each | no, but most need data a one-group library lacks | later |
| Filters other than Cell, CellFrom, Material, Mesh, MeshSurface, Surface, Energy (`UniverseFilter`, `CellInstanceFilter`, `DistribcellFilter`, `MaterialFromFilter`, `CellBornFilter`, angle and time filters, ...) | refused (v1.3) | S each (Universe: the path is already there) | no | later |
| Two filters of the same kind in one tally (OpenMC allows two MeshFilters, for instance) | refused (v1.3) | S | no | later |
| `CellFromFilter` on a track tally (bins a flight by the cell the particle last left) | refused (v1.3) | S | no | later |
| `SurfaceFilter` with a `MeshFilter` | refused (v1.3); `MeshSurfaceFilter` (v1.5) does the usual job | S | no | later |
| `EnergyFilter` bins real energies | one group: every particle sits at the group's midpoint, as in OpenMC (v1.3) | - | **YES** at Stage 3 (energy becomes a network input) | Stage 3 |
| `SphericalMesh` with a theta edge at pi/2 (the plane z = 0) | built (v1.5), checked only against the exact uncollided answer: OpenMC 0.16 scores that edge wrongly (up to 360 sigma off, measured 30 Oct 2026; other cones it gets right) | - | no | when OpenMC fixes it |
| Curved-mesh flights cut at every edge surface, then search again (cost grows with the number of edges) | by design (v1.5): a flight of N pieces tests every edge N times | S | no: walk neighbouring bins only | later, if a fine curved mesh is slow |
| `cap` rule near a curved mesh, outside it: the room is the distance to the nearest edge surface, which may be a plane's or cone's extension that bounds no bin there | by design (v1.5): balls smaller than they need be, never wrong | S | no | later |
| Curved meshes in `.vtk` files and flux-map pictures | skipped (v1.5): both draw rectangular meshes only | S | no | later |
| `MeshSurfaceFilter` on a `CylindricalMesh` or `SphericalMesh` | refused (v1.5) | M | no | later |
| `MeshSurfaceFilter` with any other filter | refused (v1.5) | S | no | later |
| `MeshSurfaceFilter` present: balls are shrunk to stay inside one voxel of its mesh, whatever the mesh rule | by design (v1.5): exact, less speed-up | - | YES to keep full-size balls: (mu, s) cannot say which faces a walk crossed | hold |
| A mesh face lying on a reflective, white or periodic wall | not counted (v1.5), as OpenMC (both stop TINY short of the wall) | - | no | - |
| Surface currents on surfaces inside a universe whose frame is moved (any lattice element but a centred one, a translated cell) | built (v1.5), checked only by conservation (net current into an absorber = - its absorption): OpenMC 0.16 gets these wrong (3 x 3 lattice +0.169 against an absorption of 0.493; `runs/2026-10-30/v1.5/openmc_bugs.log`) | - | no | when OpenMC fixes it |
| Current on a rotational periodic wall: signed against the wall the particle comes in through | built (v1.5), checked by beam unit tests; OpenMC 0.16 signs it against the wall it left, after turning (no current of either wall; same log). Translational walls agree with OpenMC | - | no | when OpenMC fixes it |
| Current on a reflective or white wall | exactly 0 (v1.5), as OpenMC (+w then -w at every reflection); partial currents there are 0 too | - | no | - |
| Point detectors with reflective, white or periodic boundaries | refused (v1.6): the next-event line would also have to follow the wall's images | M | no | later |
| Point detectors with a source that is not isotropic (monodirectional, polar-azimuthal, biased isotropic, `FileSource`) | refused (v1.6): needs the source's angular density towards the point | S | no | later |
| Point detector inside a scattering material without an averaging radius `r0` | by design (v1.6): mean right, variance infinite (the estimator's own property, as in MCNP); with `r0` > 0 the variance is finite and the bias of order r0 | - | no | - |
| Point detectors in balls: scored at 4 profile points (plus the conditioned-walk term), not at the walk's own scatters | by design (v1.6): exact on average, loses the walk's correlation (noise only) | - | no | - |
| Point detectors against OpenMC | only through a small void sphere's average (OpenMC 0.16 has no point detector) | - | no | - |
| Point detectors with the trained network backend | not checked (v1.6 checks use the exact walk) | S | no expected, the scoring is the same | v1.6.x |
| `UnstructuredMesh` (tetrahedral) | refused | L | no, but large engineering | later |
| Flux-map pictures (`run.py maps`) draw a rectilinear mesh's bins as equal widths; `.vtk` output writes only flux and absorption and assumes even spacing | cosmetic (v1.1.3) | S | no | later |

## Run settings (left in v1.0)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| OpenMC settings we don't use: survival biasing, weight windows, cutoffs, `max_lost_particles`, temperatures, anything else not read | warned by name (v1.1.3) | - | no. They change noise, speed or run length only, not the expected answer | honouring them: later |
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
| Plain MC 5-16% slower than v1.2 (sphere 4.10 vs 3.89 s, cask 13.50 vs 11.64 s, `runs/2026-10-29/v1.3/speed.log`) | v1.3 | cause of the last part not found; a fix would be v1.3.1 |
| No race against OpenMC with weight windows | v1.0 | the fair comparison for shielding claims |
| The validation pass rule treats a tally's bins as independent; neighbouring bins of a mesh tally are correlated (slab depth bins: +0.88 next door, +0.37 ten apart), so its chi2/dof spreads 3.5x wider than the rule allows and a correct code fails it by chance (7 of 40 seeds on the slab, `runs/2026-10-30/v1.6/slab_corr.log`) | v1.6 | a 10x OpenMC reference (validate's `REF_SCALE`) does not cure it: v1.6 validate still fails the slab's plain MC depth tallies (chi2/dof 2.16 and 2.24), from one 2.15-sigma shift of the whole profile. A rule that measures the correlation (effective degrees of freedom: about 6 for the slab's 60 bins) passes them (p 0.04; `neff_probe.log`): decision for Phani |
| The pass rule also trusts a reference bin that scored almost nothing: OpenMC's analog run can give a deep bin 0 +- 0 (or a handful of counts with too small an error bar) where our implicit-capture run scores a small, well-measured value, and the z-score then reads 4-6 (robustness model 26: a face current deep behind a shield, 0 +- 0 in OpenMC; `runs/2026-10-30/v1.6/model26.log`) | v1.6 | the suite reruns a FAIL once at 5x before counting it; the same effective-dof decision as above could also drop bins with too few reference counts |
| The robustness suite (v1.6) leaves out the three OpenMC 0.16 errors and draws 30 models | v1.6 | more models or more kinds of object: later |
