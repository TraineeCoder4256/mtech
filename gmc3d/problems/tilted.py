"""Tilted shapes and the new boundaries (v1.4.0).

A 20 x 20 x 30 cm block of a scatterer with two tilted inserts:

  * a pipe of an absorber: a Cylinder (r = 3 cm) whose axis runs along
    (1, 1, 1) through the centre, so it is tilted to every axis
  * a narrow Cone of a second scatterer, apex at z = -5, axis along
    (1, -1, 2) (35 degrees from z), cut off below its apex

and three kinds of boundary:

  * x faces: periodic (a particle leaving at x = +10 comes back at x = -10)
  * y faces: white (re-enters with a cosine-law direction)
  * z faces: vacuum (the only way out)

so a particle's history mixes all three.  Exercises the any-axis cylinder
and cone kernels (the distance field sees them too, so balls shrink
against tilted walls) and both new boundary conditions.  The pipe and the
cone run into the periodic and white faces, so particles cross those
boundaries inside the inserts as well as in the block.
"""
import openmc

from . import common

ALL_SCORES = ("flux", "absorption", "scatter", "total")


def build():
    materials, m = common.one_group_materials(
        "tilted", {"block": (0.8, 0.01), "pipe": (0.3, 0.3),
                   "cone": (1.5, 0.02)})
    x0 = openmc.XPlane(-10.0, boundary_type="periodic")
    x1 = openmc.XPlane(10.0, boundary_type="periodic")
    y0 = openmc.YPlane(-10.0, boundary_type="white")
    y1 = openmc.YPlane(10.0, boundary_type="white")
    z0 = openmc.ZPlane(-15.0, boundary_type="vacuum")
    z1 = openmc.ZPlane(15.0, boundary_type="vacuum")
    box = +x0 & -x1 & +y0 & -y1 & +z0 & -z1
    pipe = openmc.Cylinder(x0=0.0, y0=0.0, z0=0.0, r=3.0,
                           dx=1.0, dy=1.0, dz=1.0)
    cone = openmc.Cone(x0=0.0, y0=0.0, z0=-5.0, r2=0.04,
                       dx=1.0, dy=-1.0, dz=2.0)
    upper_cone = -cone & +openmc.ZPlane(-5.0) & +pipe
    cells = [
        openmc.Cell(name="pipe", fill=m["pipe"], region=box & -pipe),
        openmc.Cell(name="cone", fill=m["cone"], region=box & upper_cone),
        openmc.Cell(name="block", fill=m["block"],
                    region=box & +pipe & ~upper_cone),
    ]
    geometry = openmc.Geometry(cells)
    tallies = openmc.Tallies([
        common.flux_tally("total", scores=ALL_SCORES),
        common.flux_tally("cell", openmc.CellFilter(cells)),
        common.flux_tally("mesh", common.mesh_filter(
            (-10.0, -10.0, -15.0), (10.0, 10.0, 15.0), (10, 10, 15))),
        common.flux_tally("faces", openmc.SurfaceFilter([z0, z1]),
                          scores=("current",)),
    ])
    src = common.box_source((-2.0, -6.0, -12.0), (2.0, -2.0, -8.0))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
