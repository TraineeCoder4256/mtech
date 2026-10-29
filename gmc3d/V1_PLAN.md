# Version 1 work plan: source, geometry and tallies

What is left of version 1 before anisotropy (v2.0) starts: closing the
small and medium gaps in source, geometry and tallies listed in `GAPS.md`.
None of them needs the trained network to change (checked item by item;
see the "model change?" column in `GAPS.md`), so none waits on model work.

Status: proposed 27 Oct 2026; Phani gave the go on 29 Oct 2026. v1.1.3
v1.2.0 (sources), v1.3.0 (multi-filter tallies) and v1.4.0 (geometry,
small items) are done; v1.4.1 (geometry, medium items) is next.

Numbering (Phani, 29 Oct 2026): v1.1.x is the mesh work and the easy
tally items; each later area gets its own minor version (v1.2 sources,
v1.3 multi-filter tallies, v1.4 geometry, v1.5 the medium tally items),
and fixes inside an area are v1.2.1 and so on. v2.0 is still anisotropy.

Rules for every push below:
- It carries a version (`VERSION`, `CHANGELOG.md`, git tag).
- It is checked against OpenMC on a test problem that uses the new
  feature, and the unit tests pass.
- It updates `GAPS.md`: removes what it closed, adds anything it leaves out.

Out of scope here: the large gaps (DAGMC/CAD geometry, tetrahedral meshes,
compiled sources) and everything marked YES under "model change?" (balls
over 30 mfp, the network's short-path tail, the physics stages).

## v1.1.2 (done): GAPS.md with the model-change column, and this plan

## v1.1.3 (done 29 Oct 2026): make the silent gaps loud, plus the easy tally items

Built as below, with two changes: tally triggers are warned about, not
refused (they only stop a run early), and `MeshSurfaceFilter` /
`MeshBornFilter`, found passing as plain meshes, are now refused.


| item | how | check |
|---|---|---|
| OpenMC settings we don't read (survival biasing, weight windows, cutoffs, temperatures) | print a clear warning listing each one ignored. These change only noise and speed, never the expected answer, so a warning fits better than a refusal | unit test |
| Tally `estimator` other than track length | refuse (inside a ball there are no visible collisions, so a collision estimator would need its own design) | unit test |
| Tally nuclides, triggers, derivatives | refuse | unit test |
| Scores `total` and `scatter` (sigma_t and sigma_s times flux) | same track-length estimator as absorption | vs OpenMC, slab |
| 1D and 2D `RegularMesh`, `RectilinearMesh` | the same grid walk; missing axes are infinite; rectilinear edges looked up by binary search. The v1.1 ball rule works unchanged | vs OpenMC, new mesh tallies on the slab and cask |

## v1.2 (done, v1.2.0): sources, all of OpenMC's spatial and angle distributions

One mechanism instead of one case per type. Before each batch, Python
samples the source sites (position, direction) from the model's own OpenMC
distribution objects: their 1D pieces (`Uniform`, `Discrete`, `Normal`,
`PowerLaw`, `Tabular`, `Mixture`) already have a `sample` method, checked
in OpenMC 0.16. The transport then starts from that list. This covers:

- spaces: `CartesianIndependent`, `SphericalIndependent`,
  `CylindricalIndependent`, `PointCloud`, `MeshSpatial` (plus `Point`,
  `Box`)
- angles: `PolarAzimuthal` (plus `Isotropic`, `Monodirectional`)
- `FileSource` (read the site file) and `MeshSource`
- source `constraints`: born only in given cells, materials or universes,
  by rejection, as OpenMC does

`CompiledSource` stays refused, because it is user C++ code. Check: each
distribution's sampled sites are compared with the same object's own
samples, then one problem per type is compared with OpenMC.

## v1.3 (done, v1.3.0): tallies with several filters

The standard way, as OpenMC does it: each filter gives a bin index and the
combined bin is their weighted sum. Along a flight the cell, material and
surface bins are fixed, so the mesh walk just adds an offset. This unlocks:

- mesh x cell, mesh x material, cell x surface, and so on
- `EnergyFilter` accepted (one group means one bin; real bins from Stage 3)
- partial currents (in and out separately) with `CellFromFilter` x `SurfaceFilter`
- cell tallies on cells filled with a universe or lattice: the geometry
  descent records every cell it passes through, so a filled cell scores
  whatever is inside it

Check: vs OpenMC on the nested problem (filled cells) and the slab (partial
currents).

## v1.4.0: geometry, the small items (done 29 Oct 2026)

| item | how |
|---|---|
| `Cylinder` and `Cone` on any axis | rotate the point and direction into the axis frame and reuse the X/Y/Z code; surfaces get more than 4 coefficients |
| White boundary | at the wall, a cosine-distributed reflection (OpenMC's definition) |
| Periodic boundary, translational | a particle leaving one plane re-enters at its partner plane with the same direction |

Check: locate vs OpenMC on random points (as for Stage 1), then a solve.

## v1.4.1: geometry, the medium items

| item | how |
|---|---|
| `HexLattice` (both orientations, 2D and 3D) | element lookup in hex coordinates, the six walls as planes for flights and for the ball's distance |
| `Quadric` | ray crossing is a quadratic. Nearest distance has no formula, so use a guaranteed LOWER bound (from the value and gradient at the point and the quadric's curvature). Balls only get smaller, never wrong |
| Torus (`XTorus`, `YTorus`, `ZTorus`) | ray crossing is a quartic, solved robustly as OpenMC does. Nearest distance: exact formula when the tube is circular, a guaranteed lower bound when elliptical |
| Periodic boundary, rotational | rotate position and direction by the sector angle |
| Distributed materials (one cell, a material per instance) | count instances during the geometry descent, as OpenMC does |

Check: locate and solve vs OpenMC on a new problem per item (a hex
assembly, a torus/quadric shape, a rotational sector).

## v1.5: tallies, the medium items

| item | how |
|---|---|
| `CylindricalMesh`, `SphericalMesh` | flights cross curved bin walls: reuse the cylinder/sphere/plane ray code for the walls; the ball rule only needs which bin holds a point, so it works unchanged |
| Surface current on reflective surfaces and on surfaces inside a filled universe | reflective: count both the crossing and the return, as OpenMC does; inner surfaces: use the direction in the universe's own frame |
| `MeshSurfaceFilter` (currents through voxel faces) | flights: count face crossings in the grid walk. Balls: a ball's walk crosses voxel faces in ways (mu, s) cannot tell us, so when this filter is present the balls are shrunk to fit inside one voxel (`cap`). Exact, but fewer ball savings there |

## Needs your decision (research, not engineering)

- **Point detectors** (flux at a point, which Kobayashi reports). The
  standard next-event estimator scores from every collision, and inside a
  ball the collisions are invisible. One idea: the v1.1 profile points are
  distributed like collisions (collision density is proportional to flux),
  so they could serve as pseudo-collision sites and stay unbiased. That is
  a method question, and the detector's infinite-variance problem near the
  point comes with it. Yours to decide; until then, small voxels with
  `cap`.

## Order and why

Make the silent gaps loud first (safety), then sources and multi-filter
tallies (the benchmarks need them), then geometry. Tell me to reorder
anything, or to drop items you don't need for the thesis.
