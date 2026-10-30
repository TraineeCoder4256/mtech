# Accepted inputs (gmc3d v1.6, SRS V1-ROB-7)

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
| `survival_biasing`, `weight_windows`, `cutoff`, `temperature`, `max_lost_particles`, tally `triggers`, and every other setting | warned. They change only the noise or the run's length |
| `output`, `verbosity`, `statepoint`, `sourcepoint`, `track`, `max_tracks` | quiet: these only concern OpenMC's own files |

## Materials

| input | gmc3d |
|---|---|
| `Material` holding one `Macroscopic` (with any density) from an `MGXSLibrary` with 1 group | used |
| more than one group | refused (stage 3) |
| nuclides in a material | refused (macroscopic data only) |
| anisotropic scattering (Legendre order > 0, non-zero) | refused (stage 2) |
| fissile data | refused (stage 4) |
| total != absorption + scattering | refused |
| nothing absorbs and no surface is vacuum | refused, because no history would end (v1.6) |

## Surfaces and boundaries

| input | gmc3d |
|---|---|
| `XPlane`, `YPlane`, `ZPlane`, `Plane` | used |
| `Sphere` | used |
| `XCylinder`, `YCylinder`, `ZCylinder`, `Cylinder` (any axis) | used |
| `XCone`, `YCone`, `ZCone`, `Cone` (any axis) | used |
| `XTorus`, `YTorus`, `ZTorus` (circular or elliptical tube) | used. Refused if the tube meets the axis (c >= a) |
| `Quadric` | used (the ball distance near it is a lower bound, so slower but exact) |
| boundary `transmission`, `vacuum`, `reflective`, `white` | used |
| boundary `periodic`, translational (parallel planes) | used |
| boundary `periodic`, rotational (two planes through the z axis) | used. Refused for planes that do not contain the z axis |
| `albedo` other than 1 | refused |
| any other surface class | refused |

## Geometry

| input | gmc3d |
|---|---|
| `Cell` filled with a material, void, a `Universe` or a lattice | used |
| a cell with a list of materials (distributed materials) | used. Refused inside a lattice's outer universe |
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
| angle `Isotropic`, `Monodirectional`, `PolarAzimuthal` | used |
| energy and time distributions | read only for their biasing weights (one group, steady state) |
| `FileSource` (.h5) | used |
| `MeshSource` | used |
| `CompiledSource` | refused (user C++ code) |
| constraints `domains` (cell, material, universe) | used, by redrawing as OpenMC does |
| constraint `fissionable` | refused |
| constraints `energy_bounds`, `time_bounds`, `rejection_strategy` 'kill' | warned (OpenMC 0.16 multigroup does the same) |
| several sources, `strength`, biasing | used |

## Tallies

| input | gmc3d |
|---|---|
| scores `flux`, `absorption`, `scatter`, `total` (track length) | used |
| score `current` with a `SurfaceFilter` or a `MeshSurfaceFilter` | used |
| any other score | refused |
| `estimator` 'collision' or 'analog' | refused (collisions inside a ball are not seen) |
| nuclides other than 'total', derivatives | refused |
| `CellFilter` (any level, filled cells too), `MaterialFilter`, `EnergyFilter` | used, in any combination |
| `MeshFilter` on `RegularMesh` (1D, 2D, 3D), `RectilinearMesh`, `CylindricalMesh`, `SphericalMesh` | used, combined with the filters above |
| `SurfaceFilter` (any surface, any level, any boundary) | used, alone or with cell, cell-from, material and energy filters |
| `CellFromFilter` | used with a `SurfaceFilter` (partial currents). Refused otherwise |
| `MeshSurfaceFilter` on a Regular or Rectilinear mesh | used, alone. Refused on curved meshes or combined with other filters |
| `MeshBornFilter`, `DistribcellFilter`, angle, time, and any other filter | refused |
| unstructured (tetrahedral) meshes | refused (large gap) |
| a filter used twice in one tally | refused |

## gmc3d's own additions (no OpenMC equivalent)

| input | gmc3d |
|---|---|
| point detectors, `common.point_detector(model, name, xyz, r0)` (v1.6) | next-event estimator, with balls as described in `GAPS.md`. Refused unless every source is isotropic, and refused with reflective, white or periodic boundaries |
