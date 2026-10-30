# What gmc3d leaves out (the gaps list)

Every version adds what it left out here and removes what it fixed.

Since v1.6.1 this file has three parts:

1. **Real gaps**: an OpenMC input that OpenMC runs and gmc3d refuses or
   answers differently. Only these are work for us.
2. **Where OpenMC refuses too, or is wrong**: inputs OpenMC itself stops
   on (gmc3d refuses them with the same reason), and places where
   OpenMC 0.16 gives a wrong answer that we proved (gmc3d gives the right
   one, so it cannot be compared with OpenMC there).
3. **Open checks**: unfinished checks, not missing features.

Columns of part 1:
- **shows as**: "refused" means `core/openmc_import.py` stops with an
  `Unsupported` error naming the feature. "SILENT" means the model runs
  anyway and the answer is approximate, the dangerous kind. "slower"
  means the answer is right but balls are smaller or switched off.
- **size**: S = a day or less, M = a few days, L = a week or more.
- **model change?**: whether fixing it needs the trained network to change.
  The network sees only the ball's radius R and returns (mu, s), so
  geometry, sources, tallies and boundaries normally stay outside it.
- **plan**: a version, "hold" (waits for model work) or "later".

Last updated: v1.6.1 (30 Oct 2026). v1.6.1 closed 20 rows of the v1.6.0
list (`CHANGELOG.md`); every one is checked against OpenMC by
`python run.py gapcheck` (20 cases, `runs/2026-10-30/v1.6.1/`).

## 1. Real gaps

### 1a. Physics: the stages (the model work)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| Anisotropic scattering | refused | L | **YES**: the scattering law becomes a network input | v2.0 |
| More than one energy group (and with it real energies in `EnergyFilter`, source energy, `energy_bounds`) | refused | L | **YES** | Stage 3 |
| Nuclides inside materials (and tallies by nuclide) | refused | M | no at one group (sum to macroscopic), YES once energy enters | Stage 3 |
| Scattering multiplicity other than 1 ((n,xn)) | refused | M | **YES** (a ball's walk would gain particles) | Stage 3 |
| Fission, k-eff (`run_mode` eigenvalue, fissile materials) | refused | L | **YES** (fission sites inside balls) | Stage 4 |
| Continuous energy | refused | L | **YES** | Stage 5 |
| Photons, coupled neutron-gamma | refused | L | **YES** | later |
| Distributed temperatures | not read (one group, one temperature) | - | YES once temperature enters | Stage 3+ |

### 1b. Inputs that are not physics

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| DAGMC / CAD geometry | refused | L | no, but large engineering (a distance field over a triangle mesh) | later |
| `UnstructuredMesh` (tetrahedral) in tallies and in `MeshSource` / `MeshSpatial` | refused | L | no, but large engineering | later |
| Tally derivatives (differential tallies) | refused | M | no at one group; the sensitivity of a ball's (mu, s) to a cross section is a research question | later |
| `CompiledSource` (user C++ code) | refused | - | no: it is someone else's compiled program, which a Python code cannot run | stays refused |
| The `events` score | refused, by design: it counts how a code transports, not the problem, so no two codes agree on it | - | no | stays refused |
| OpenMC settings that change noise or run length only (survival biasing, weight windows, cutoffs, `max_lost_particles`, ...) | warned by name; the expected answer is the same | M+ (weight windows) | no at ball exits; splitting inside a ball is impossible without seeing the walk | later |
| Tally `estimator` collision or analog | warned: scored by track length (same expected answer, other noise) | - | no; a collision estimator inside a ball would be a method question | - |

### 1c. Right answer, slower (method limits)

| gap | shows as | size | model change? | plan |
|---|---|---|---|---|
| A tally with a `MuFilter` or `CollisionFilter`, or a `Polar`/`Azimuthal`/`SphericalHarmonics` filter on a track tally | slower: the whole run is plain Monte Carlo (warned); they bin by what happens inside a ball's walk, which the ball hides | - | **YES** to keep balls: the network would have to return the walk's last cosine, its scatter count, or its exit direction distribution | hold |
| `MeshSurfaceFilter` present | slower: balls stay inside one voxel of its mesh | - | YES to keep full-size balls ((mu, s) cannot say which faces a walk crossed) | hold |
| Balls bigger than 30 mfp | **SILENT** shrink to 30 mfp (exact, slower) | M | **YES**: retrain on a wider R range | hold |
| Network's rare short-path tail (R = 29, kappa = 0.1: weight +73%, z 3.1, 5 Oct ballcheck) | **SILENT** | M | **YES**: retraining or another architecture | hold |
| Lookup-table backend's fixed sample (~1.2% leakage bias in strongly absorbing balls) | **SILENT** | - | the table is not the model; kept only as a reference | - |
| R* below 1 mfp with a mesh tally | refused (v1.1) | S | no: rebuild the profile table from a smaller R | later |
| Quadric nearest distance is a lower bound | slower: balls near a quadric are smaller than they could be | M | no | later, if a quadric-heavy benchmark needs it |
| Torus flights about 0.4 us (a sphere 0.03 us) | slower: a bracketed quartic root | S | no | later |
| Curved-mesh flights test every edge surface at every cut | slower with fine curved meshes | S | no: walk neighbouring bins only | later |
| `cap` rule near a curved mesh: room measured to the nearest edge surface, which may be an extension that bounds no bin there | slower: balls smaller than they need be, never wrong | S | no | later |
| Mesh points cost about 0.1 us each, up to 16 per ball | slower | S | no: adapt the point count to the mesh | later |
| Network on GPU only for the draws; transport on CPU | by design | L | no | later |

### 1d. Exact on average, noise differs (by design)

These give OpenMC's expected answer; only the error bar differs.

| item | note |
|---|---|
| Mesh flux and the general scorer's tallies inside balls | scored at profile points (exact on average, v1.1) |
| Point detectors in balls | 4 profile points plus the conditioned-walk term (v1.6) |
| Point detector in a scattering material without `r0` | mean right, variance infinite (the estimator's own property, as in MCNP) |
| Mu filter on a flight before the first scatter | OpenMC never resets the scatter cosine, so it bins that flight by the cosine the previous history on its thread ended with. gmc3d does the same in distribution: each run with a Mu filter follows the analog history its weighted walk stands for, so the carried cosine is the one OpenMC's history would end with (v1.6.1) |
| `WeightFilter` | bins by the weight an analog game carries (the birth weight), which is what OpenMC's analog run sees |

### 1e. Cosmetic

| gap | size | plan |
|---|---|---|
| Curved meshes are skipped in `.vtk` files and flux-map pictures | S | later |
| Flux maps draw a rectilinear mesh's bins as equal widths; `.vtk` writes flux and absorption only and assumes even spacing | S | later |

## 2. Where OpenMC refuses too, or is wrong

### 2a. OpenMC 0.16 refuses these too (gmc3d refuses with the same reason)

| input | OpenMC's message or behaviour |
|---|---|
| Periodic boundary on a non-plane surface | refused |
| Rotational periodic walls that do not contain the z axis | refused |
| `MeshSurfaceFilter` with Cell, CellFrom, Material, MaterialFrom or Surface filters | "Cannot tally mesh surface currents in the same tally as normal surface currents" |
| `MeshSurfaceFilter` with a score other than current | refused |
| A crossing tally (Surface, or CellFrom/MaterialFrom with flux or current) with scores other than current and flux | refused |
| Flux with an `EnergyoutFilter` or a `DelayedGroupFilter` | "Cannot tally flux with an outgoing energy filter" (and delayed group) |
| `EnergyFunctionFilter`, `heating`, `(n,gamma)` in multigroup mode | refused (measured 30 Oct 2026); other reaction-name scores are assumed refused the same way, not each measured |
| Distributed materials, or a `CellInstanceFilter` cell, inside a lattice's outer universe | OpenMC does not number instances there |
| Source `energy_bounds`, `time_bounds` | warned, not applied: OpenMC 0.16 in multigroup mode does not apply them either (measured) |
| `rejection_strategy: 'kill'` | acts as 'resample', as OpenMC 0.16 does for domains (measured) |
| Filters that are always 0 here (Reaction, ParentNuclide, ParticleProduction, DelayedGroup) and the fission and decay scores | accepted and 0, as OpenMC gives (measured) |

### 2b. OpenMC 0.16 is wrong here (proved), so these are checked without it

| item | how we know OpenMC is wrong | how gmc3d is checked |
|---|---|---|
| `SphericalMesh` theta edge at pi/2 | up to 360 sigma off the exact uncollided answer; other cones right (v1.5) | exact uncollided answer |
| Surface currents on surfaces inside a moved universe (any lattice element but a centred one, a translated cell) | net current into an absorber is not minus its absorption (3 x 3 lattice +0.169 against 0.493; `runs/2026-10-30/v1.5/openmc_bugs.log`) | conservation |
| Current on a rotational periodic wall | OpenMC signs it against the wall left, after turning (same log) | beam unit tests |
| Inward crossings of a `CylindricalMesh` / `SphericalMesh` face (`MeshSurfaceFilter`) | OpenMC files an inward crossing by the next face of that axis along the ray, not the face crossed (`mesh.cpp` raytrace_mesh): it reports currents in through the r = 0 'face' of the innermost shell, which does not exist (v1.6.1) | out-crossings match OpenMC bin by bin; in-crossings match it summed over each element's two faces of an axis (`gapcheck` meshsurf) |
| `CylindricalMesh` with a phi range of pi/3 in a rotational wedge | OpenMC's mesh sums to 6% below its own cell tally of the same region (v1.6.1) | gmc3d's mesh sums to its cell tally; `gapcheck` rotation case uses a RegularMesh |
| Distributed materials with a `None` (void) entry | OpenMC 0.16 stops on it in multigroup mode (v1.4.1) | unit test |
| `PointCloud` / `MeshSpatial` source `bias` | OpenMC 0.16 refuses the XML its own Python writes for it (v1.2) | unit test |
| `'scatter'` score in a model with a void cell | OpenMC 0.16 segfaults in multigroup mode (v1.1.3) | unit tests; left out of the robustness suite |
| A lattice inside a rotated cell: the cosine OpenMC uses for a crossing tally's flux at a lattice wall | OpenMC dots the global direction with the lattice-frame normal (`particle.cpp` event_cross_surface); gmc3d uses the true cosine | not compared (an unrotated lattice matches OpenMC, `gapcheck` lattice_crossings); this is a reading of OpenMC's source, not measured |

## 3. Open checks (unfinished, not missing features)

| item | since | note |
|---|---|---|
| Plain MC 5-16% slower than v1.2 (sphere 4.10 vs 3.89 s, cask 13.50 vs 11.64 s, `runs/2026-10-29/v1.3/speed.log`) | v1.3 | cause of the last part not found |
| Plain MC 6-7% slower than v1.6.0 with no new tally in use (cask 24.2 vs 22.6 s, hexcore 33.7 vs 31.8 s, means of four interleaved passes; answers bit-identical; `runs/2026-10-30/v1.6.1/speed.log`) | v1.6.1 | not the general scorer's branches in the flight loop (compiling them out leaves the time unchanged); the extra arguments every history's call carries are the next suspect, unmeasured |
| No race against OpenMC with weight windows | v1.0 | the fair comparison for shielding claims |
| The validation pass rule treats a tally's bins as independent; neighbouring mesh bins are correlated (slab depth bins +0.88 next door), so a correct code fails it by chance (7 of 40 seeds on the slab, `runs/2026-10-30/v1.6/slab_corr.log`) | v1.6 | a rule counting effective bins (about 6 of the slab's 60) passes them (p 0.04): decision for Phani |
| The pass rule trusts a reference bin that scored almost nothing (OpenMC's analog run can give a deep bin 0 +- 0; robustness model 26) | v1.6 | the suite reruns a FAIL once at 5x; the same effective-dof decision could also drop bins with too few reference counts |
| The robustness suite leaves out the OpenMC errors of 2b and draws 30 models | v1.6 | more models or more kinds of object: later |
