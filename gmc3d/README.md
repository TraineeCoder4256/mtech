# gmc3d: ball-step GMC for any 3D CSG geometry

Status: requirements stage. No solver code yet.

The method: at each collision, a distance field gives the biggest one-material
ball around the particle. If the ball is big enough, a learned (or exact) ball
step moves the particle to the ball's edge in one call; otherwise ordinary Monte
Carlo takes one flight. The geometry lives only in the distance field, so one
small model serves any shape.

The requirements, split into five stages, are in the project's SRS document
("3D GMC: Software Requirements Specification"):

1. Any CSG geometry, one energy, isotropic scattering, fixed source.
2. + anisotropic scattering.
3. + multigroup energy.
4. + criticality (k-eff).
5. + continuous energy, inside OpenMC.

Rules for this folder (SRS NF-8): all 3D work lives here, on branch `3d-gmc`.
`gmc2d/` stays frozen so the 2D results remain reproducible; code reused from
it is copied and adapted, not imported.
