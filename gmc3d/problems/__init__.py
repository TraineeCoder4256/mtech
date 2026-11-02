"""Test problems, each a function returning an openmc.Model.

    sphere      homogeneous sphere; analytic checks; biggest balls
    slab        thick shield slab; the Stage 1 headline problem
    lattice3d   gmc2d's checkerboard in 3D; the negative control
    curved      cylinder, sphere and cone; curved walls
    nested      3D lattice, rotated/translated universe, outer universe
    cask        storage cask with a streaming duct (a user-style example)
    tilted      any-axis cylinder and cone; periodic and white boundaries
    hexcore     hexagonal lattices, both orientations, tiled outer universe
    sector      rotational periodic quarter; tori, quadric, distributed mats

Usage:  from problems import PROBLEMS;  model = PROBLEMS["slab"]()
"""
from . import (cask, curved, hexcore, lattice3d, nested, sector, slab,
               sphere, tilted)

PROBLEMS = {
    "sphere": sphere.build,
    "slab": slab.build,
    "lattice3d": lattice3d.build,
    "curved": curved.build,
    "nested": nested.build,
    "cask": cask.build,
    "tilted": tilted.build,
    "hexcore": hexcore.build,
    "sector": sector.build,
}
