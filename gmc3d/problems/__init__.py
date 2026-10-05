"""Test problems, each a function returning an openmc.Model.

    sphere      homogeneous sphere; analytic checks; biggest balls
    slab        thick shield slab; the Stage 1 headline problem
    lattice3d   gmc2d's checkerboard in 3D; the negative control
    curved      cylinder, sphere and cone; curved walls
    nested      3D lattice, rotated/translated universe, outer universe

Usage:  from problems import PROBLEMS;  model = PROBLEMS["slab"]()
"""
from . import curved, lattice3d, nested, slab, sphere

PROBLEMS = {
    "sphere": sphere.build,
    "slab": slab.build,
    "lattice3d": lattice3d.build,
    "curved": curved.build,
    "nested": nested.build,
}
