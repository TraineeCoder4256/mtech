# Accepted inputs (gmc3d v1.6.1, SRS V1-ROB-7)

These are the OpenMC 0.16 classes and settings that gmc3d reads. Anything
else falls into one of two groups:

- **Refused:** the model is not run, and the error names the item.
- **Warned:** the model runs, but the item is named in a warning. Only
  items that cannot change the expected answer are allowed through this
  way.

Nothing is ignored silently. To see what gmc3d makes of your own model
without running it, use

    python run.py check my_model.py        (or a problem name, model.xml or a folder)

This lists everything used, refused and warned in one go. Larger gaps,
and whether closing each one needs the network to change, are in
`GAPS.md`.

## Settings

| setting | gmc3d |
|---|---|
| `run_mode` 'fixed source' | used |
| `run_mode` 'eigenvalue' | refused (criticality is stage 4) |
| `energy_mode` 'multi-group' | used: one energy group |
| `energy_mode` 'continuous-energy' | refused (stage 3 is multigroup) |
| `particles`, `batches`, `seed` | used |
| `source` | used, see Sources |
| `photon_transport` | refused (neutrons only) |
| tally `triggers` (with `trigger_active`, `trigger_max_batches`, `trigger_batch_interval`) | used: batches run until every trigger is met (v1.6.1) |
| `survival_biasing`, `weight_windows`, `cutoff`, `temperature`, `max_lost_particles`, and every other setting | warned. They change only the noise or the run's length |
| `output`, `verbosity`, `statepoint`, `sourcepoint`, `track`, `max_tracks` | quiet: these only concern OpenMC's own files |

## Materials

| input | gmc3d |
|---|---|
| `Material` holding one `Macroscopic` (with any density) from an `MGXSLibrary` with 1 group | used |
| more than one group | refused (stage 3) |
| nuclides in a material | refused (macroscopic data only) |
| anisotropic scattering (Legendre order > 0, non-zero) | refused (stage 2) |
| fissile data | refused (stage 4) |
| scattering multiplicity other than 1 ((n,xn)) | refused (v1.6.1) |
| `inverse_velocity` | used for time filters and the inverse-velocity score (v1.6.1) |
| total != absorption + scattering | refused |
| nothing absorbs and no surface is vacuum | refused, because no history would end (v1.6) |

## Surfaces and boundaries

| input | gmc3d |
|---|---|
| `XPlane`, `YPlane`, `ZPlane`, `Plane` | used |
| `Sphere` | used |
| `XCylinder`, `YCylinder`, `ZCylinder`, `Cylinder` (any axis) | used |
| `XCone`, `YCone`, `ZCone`, `Cone` (any axis) | used |
| `XTorus`, `YTorus`, `ZTorus` (circular or elliptical tube, horn and spindle tori too) | used |
| `Quadric` | used (the ball distance near it is a lower bound, so slower but exact) |
| boundary `transmission`, `vacuum`, `reflective`, `white` | used |
| boundary `periodic`, translational (parallel planes; general `Plane`s pair by `periodic_surface` or by being the only parallel partner) | used |
| boundary `periodic`, rotational (two planes through the z axis, the problem on either side of each) | used. Refused for planes that do not contain the z axis, as in OpenMC |
| `albedo` | used (v1.6.1): the weight not sent back counts as leakage, as in OpenMC |
| any other surface class | refused |

## Geometry

| input | gmc3d |
|---|---|
| `Cell` filled with a material, void, a `Universe` or a lattice | used |
| a cell with a list of materials (distributed materials) | used. Refused inside a lattice's outer universe (OpenMC does not number instances there) |
| cell `translation`, `rotation` | used |
| region operators `&`, `\|`, `~` | used |
| `RectLattice` (2D, 3D), with an `outer` universe | used |
| `HexLattice` ('x' and 'y' orientation, 2D, 3D) | used |
| DAGMC / CAD universes | refused (large gap) |

## Sources

| input | gmc3d |
|---|---|
| `IndependentSource` (`particle` 'neutron') | used |
| space `Point`, `Box`, `CartesianIndependent`, `CylindricalIndependent`, `SphericalIndependent`, `PointCloud`, `MeshSpatial` | used |
| `Box(only_fissionable=True)` | refused |
| angle `Isotropic` (with or without a `bias`), `Monodirectional`, `PolarAzimuthal` | used |
| energy distribution | read only for its biasing weights (one group) |
| time distribution | used: particles carry time (v1.6.1) |
| `FileSource` (.h5, MCPL `.mcpl` / `.mcpl.gz`) | used |
| `MeshSource`, `MeshSpatial` on Regular (1D, 2D, 3D), Rectilinear, Cylindrical, Spherical meshes | used. Refused on an `UnstructuredMesh` |
| `CompiledSource` | refused (user C++ code) |
| constraints `domains` (cell, material, universe) | used, by redrawing as OpenMC does |
| constraint `fissionable` | refused |
| constraints `energy_bounds`, `time_bounds`, `rejection_strategy` 'kill' | warned (OpenMC 0.16 multigroup does the same) |
| several sources, `strength`, biasing | used |

## Tallies

Since v1.6.1 every filter and score OpenMC 0.16 runs in a one-group,
fixed-source problem is used. Tallies the fast kernels take (cell,
material, energy, mesh and surface filters in the usual combinations) run
there; every other one runs in the general scorer (`core/ext.py`), checked
against OpenMC by `python run.py gapcheck`.

| input | gmc3d |
|---|---|
| scores `flux`, `absorption`, `scatter`, `total`, `nu-scatter`, `inverse-velocity` | used |
| scores `fission`, `nu-fission`, `kappa-fission`, `prompt-nu-fission`, `delayed-nu-fission`, `decay-rate` | used: 0, as OpenMC gives without fissile material |
| score `current` | used with Surface, CellFrom, MaterialFrom or MeshSurface filters |
| score `events` | refused (it counts how a code transports, not the problem) |
| `heating`, `(n,gamma)` and `EnergyFunctionFilter` | refused, as OpenMC 0.16 refuses them in multigroup mode |
| `estimator` 'collision' or 'analog' | warned: scored by track length (the same expected answer) |
| nuclides other than 'total', derivatives | refused |
| `CellFilter`, `CellFromFilter`, `MaterialFilter`, `MaterialFromFilter`, `UniverseFilter`, `CellInstanceFilter`, `DistribcellFilter`, `CellBornFilter`, `EnergyFilter`, `EnergyoutFilter`, `ParticleFilter`, `TimeFilter`, `WeightFilter`, `CollisionFilter` | used, in any combination, a kind twice too |
| `MeshFilter`, `MeshBornFilter`, `MeshMaterialFilter` on `RegularMesh` (1D, 2D, 3D, with `translation`), `RectilinearMesh`, `CylindricalMesh`, `SphericalMesh` | used, combined with any of the above |
| `SurfaceFilter` (any surface, any level, any boundary), `MuSurfaceFilter` | used; a Surface, CellFrom or MaterialFrom tally with flux or current is scored at crossings, lattice walls included, as in OpenMC |
| `MeshSurfaceFilter` on any of those meshes | used, with energy, time and the other non-cell filters. Refused with cell, material or surface filters, as in OpenMC |
| `PolarFilter`, `AzimuthalFilter`, `SphericalHarmonicsFilter`, `MuFilter`, `LegendreFilter`, `SpatialLegendreFilter`, `ZernikeFilter`, `ZernikeRadialFilter` | used. Mu and Collision filters, and Polar, Azimuthal and SphericalHarmonics on a track tally, make the run plain Monte Carlo (warned) |
| `ReactionFilter`, `ParentNuclideFilter`, `ParticleProductionFilter`, `DelayedGroupFilter` | used: 0, as OpenMC gives here |
| unstructured (tetrahedral) meshes | refused (large gap) |

## gmc3d's own additions (no OpenMC equivalent)

| input | gmc3d |
|---|---|
| point detectors, `common.point_detector(model, name, xyz, r0)` (v1.6) | next-event estimator, with balls as described in `GAPS.md`. Any source, any boundary (v1.6.1); refused only with reflective or periodic walls in a problem with no material at all |
