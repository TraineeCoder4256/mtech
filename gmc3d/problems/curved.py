"""Curved shapes: a cylindrical tank holding a sphere and a cone.

  * tank: z-cylinder, r = 30 cm, 60 cm tall, of a scatterer
  * an absorbing sphere (r = 8 cm) off-centre inside it
  * a cone of a second scatterer (apex up) carved out of the tank's floor
  * a void gap between the tank wall and an outer vacuum box

Exercises cylinders, spheres, cones, unions and complements, void cells,
and balls that shrink as they approach a curved wall.

Since v1.5 a CylindricalMesh (origin on the tank's floor, 4 rings out to
the outer box's corner region, 6 phi sectors, 4 z layers) checks the
curved-mesh walk, and the same mesh times a material filter checks it
with another filter.  A SphericalMesh about the origin (4 shells, 3 theta
cones, 3 phi sectors) checks theta bins away from the mesh's apex.  (No
theta edge at pi/2: OpenMC 0.16 gets that one wrong, see sphere.py.)
"""
import numpy as np
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
    cyl = openmc.CylindricalMesh(r_grid=[0.0, 10.0, 20.0, 30.0, 40.0],
                                 phi_grid=np.linspace(0.0, 2 * np.pi, 7),
                                 z_grid=[0.0, 15.0, 30.0, 45.0, 60.0],
                                 origin=(0.0, 0.0, -30.0))
    tallies.append(common.flux_tally("cylmesh", openmc.MeshFilter(cyl),
                                     scores=("flux", "absorption")))
    tallies.append(common.flux_tally("cylmesh_mat", openmc.MeshFilter(cyl)))
    tallies[-1].filters = tallies[-1].filters + [openmc.MaterialFilter(
        [m["water"], m["steel"]])]
    tallies.append(common.flux_tally("sphmesh", openmc.MeshFilter(
        openmc.SphericalMesh(r_grid=[0.0, 8.0, 16.0, 26.0, 40.0],
                             theta_grid=[0.0, 0.9, 2.3, np.pi],
                             phi_grid=[0.0, 2.0, 4.2, 2 * np.pi]))))
    src = common.point_source((-15.0, -10.0, 10.0))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
