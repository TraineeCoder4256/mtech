"""A storage cask with a streaming duct: a new geometry written the way a
user would write one, to show the workflow end to end.

  * cavity: z-cylinder, r = 20 cm, 140 cm tall, homogenised fuel; the source
  * steel liner around it (r 20 to 25 cm, with lids)
  * concrete body (r 25 to 58 cm, 176 cm tall): the thick part, 13 mfp
  * outer steel shell (r 58 to 60 cm, 180 cm tall)
  * a void duct, an x-cylinder of r = 2.5 cm, from the liner out through
    the concrete and the shell (radiation streams along it)
  * a void detector sphere, r = 15 cm, 10 cm outside the duct's mouth
  * everything inside a vacuum sphere of r = 120 cm

Uses two cylinder orientations, spheres, planes, unions and complements,
and void cells.  Tallies: total, every cell (the detector is the
interesting one), and a coarse mesh.  Since v1.1.3 also: total on the cell
tally, all four scores by material, a 2D RegularMesh in x-y (unbounded in
z), and a RectilinearMesh whose bins are fine along the duct and coarse
elsewhere.  The `scatter` score sits only on the material tally because
OpenMC 0.16 crashes (segmentation fault, checked 29 Oct 2026) when a
multigroup track-length `scatter` tally is scored in a void cell.
"""
import openmc

from . import common


def build():
    materials, m = common.one_group_materials(
        "cask", {"fuel": (0.6, 0.03),
                 "steel": (0.8, 0.05),
                 "concrete": (0.4, 0.004)})

    def can(r, h):
        """Region inside a z-cylinder of radius r and height 2h."""
        return -openmc.ZCylinder(r=r) & +openmc.ZPlane(-h) & \
            -openmc.ZPlane(h)

    cavity = can(20.0, 70.0)
    liner = can(25.0, 75.0)
    concrete = can(58.0, 88.0)
    body = can(60.0, 90.0)
    duct = -openmc.XCylinder(y0=0.0, z0=0.0, r=2.5) & \
        +openmc.XPlane(25.0) & body
    detector = -openmc.Sphere(x0=85.0, y0=0.0, z0=0.0, r=15.0)
    world = -openmc.Sphere(r=120.0, boundary_type="vacuum")

    cells = [
        openmc.Cell(name="fuel", fill=m["fuel"], region=cavity),
        openmc.Cell(name="liner", fill=m["steel"], region=liner & ~cavity),
        openmc.Cell(name="concrete", fill=m["concrete"],
                    region=concrete & ~liner & ~duct),
        openmc.Cell(name="shell", fill=m["steel"],
                    region=body & ~concrete & ~duct),
        openmc.Cell(name="duct", fill=None, region=duct),
        openmc.Cell(name="detector", fill=None, region=detector),
        openmc.Cell(name="air", fill=None,
                    region=world & ~body & ~detector),
    ]
    geometry = openmc.Geometry(cells)
    duct_grid = openmc.RectilinearMesh()
    duct_grid.x_grid = [-120.0, -60.0, -25.0, 0.0, 25.0, 40.0, 58.0, 60.0,
                        70.0, 85.0, 100.0, 120.0]
    duct_grid.y_grid = [-120.0, -20.0, -2.5, 2.5, 20.0, 120.0]
    duct_grid.z_grid = [-120.0, -90.0, -2.5, 2.5, 90.0, 120.0]
    tallies = openmc.Tallies([
        common.flux_tally("total"),
        common.flux_tally("cell", openmc.CellFilter(cells),
                          scores=("flux", "absorption", "total")),
        common.flux_tally("material", openmc.MaterialFilter(
            [m["fuel"], m["steel"], m["concrete"]]),
            scores=("flux", "absorption", "scatter", "total")),
        common.flux_tally("mesh", common.mesh_filter(
            (-120.0, -120.0, -120.0), (120.0, 120.0, 120.0), (12, 12, 12))),
        common.flux_tally("plan2d", common.mesh_filter(
            (-120.0, -120.0), (120.0, 120.0), (24, 24)),
            scores=("flux", "total")),
        common.flux_tally("duct_grid", openmc.MeshFilter(duct_grid),
                          scores=("flux", "absorption", "total")),
    ])
    src = common.box_source((-14.0, -14.0, -60.0), (14.0, 14.0, 60.0))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
