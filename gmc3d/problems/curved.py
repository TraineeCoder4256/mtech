"""Curved shapes: a cylindrical tank holding a sphere and a cone.

  * tank: z-cylinder, r = 30 cm, 60 cm tall, of a scatterer
  * an absorbing sphere (r = 8 cm) off-centre inside it
  * a cone of a second scatterer (apex up) carved out of the tank's floor
  * a void gap between the tank wall and an outer vacuum box

Exercises cylinders, spheres, cones, unions and complements, void cells,
and balls that shrink as they approach a curved wall.
"""
import openmc

from . import common


def build(sig_s=0.4, sig_a=0.004):
    materials, m = common.one_group_materials(
        "curved", {"water": (sig_s, sig_a),
                   "steel": (0.6, 0.15),
                   "concrete": (0.3, 0.01)})
    tank_wall = openmc.ZCylinder(r=30.0)
    floor, roof = openmc.ZPlane(-30.0), openmc.ZPlane(30.0)
    ball = openmc.Sphere(x0=12.0, y0=0.0, z0=5.0, r=8.0)
    cone = openmc.ZCone(x0=-8.0, y0=4.0, z0=0.0, r2=0.25)   # apex at z = 0
    tank = -tank_wall & +floor & -roof
    lower_cone = -cone & -openmc.ZPlane(0.0)
    outer = openmc.model.RectangularParallelepiped(
        -35.0, 35.0, -35.0, 35.0, -35.0, 35.0, boundary_type="vacuum")
    cells = [
        openmc.Cell(name="sphere", fill=m["steel"], region=-ball),
        openmc.Cell(name="cone", fill=m["concrete"],
                    region=tank & lower_cone),
        openmc.Cell(name="water", fill=m["water"],
                    region=tank & +ball & ~lower_cone),
        openmc.Cell(name="gap", fill=None, region=-outer & ~tank),
    ]
    geometry = openmc.Geometry(cells)
    tallies = openmc.Tallies([
        common.flux_tally("total"),
        common.flux_tally("cell", openmc.CellFilter(cells)),
        common.flux_tally("mesh", common.mesh_filter(
            (-35.0, -35.0, -35.0), (35.0, 35.0, 35.0), (14, 14, 14))),
    ])
    src = common.point_source((-15.0, -10.0, 10.0))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
