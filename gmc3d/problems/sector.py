"""A quarter-core sector with tori, a quadric and distributed materials
(v1.4.1).

The quarter x > 0, y > 0 of a cylinder (r = 20 cm, z -10 to 10), with
ROTATIONAL periodic walls: a particle leaving through the x = 0 wall comes
back through the y = 0 wall, turned 90 degrees about z, and the other way
round.  Inside:

  * a 3 x 3 RectLattice of pins (x, y 2 to 8 cm, z -10 to 4) whose pin
    cell has DISTRIBUTED materials: nine instances, nine entries (two
    fuels and moderator, in a pattern), as OpenMC's distribmat.  (No void
    entry: OpenMC 0.16 stops on one in multigroup mode, measured.)
  * a ZTorus ring round the z axis (a = 14.5, tube 1.5 across and 3 tall:
    an elliptical tube), cut by both periodic walls
  * a small XTorus ring (circular tube) beside the lattice
  * a tilted ellipsoid written as a general Quadric above the lattice

Exercises the torus kernels (quartic flights, exact ellipse distance), the
quadric's lower-bound distance, the rotation at the periodic walls, and
the instance count behind distributed materials.
"""
import numpy as np
import openmc

from . import common

ALL_SCORES = ("flux", "absorption", "scatter", "total")


def build():
    materials, m = common.one_group_materials(
        "sector", {"fuela": (0.8, 0.2), "fuelb": (0.6, 0.05),
                   "moderator": (1.0, 0.01), "ring": (0.9, 0.3),
                   "smallring": (0.5, 1.5), "egg": (1.4, 0.1),
                   "shell": (1.2, 0.02)})
    pin_s = openmc.ZCylinder(r=0.6)
    pin = openmc.Cell(name="pin", region=-pin_s)
    pin.fill = [m["fuela"], m["fuelb"], m["moderator"], m["fuelb"],
                m["fuela"], m["fuelb"], m["moderator"], m["fuelb"],
                m["fuela"]]
    pin_u = openmc.Universe(name="pin", cells=[
        pin, openmc.Cell(name="pin-mod", fill=m["moderator"],
                         region=+pin_s)])
    lat = openmc.RectLattice(name="pins")
    lat.lower_left = (2.0, 2.0)
    lat.pitch = (2.0, 2.0)
    lat.universes = [[pin_u] * 3 for _ in range(3)]

    x0 = openmc.XPlane(0.0, boundary_type="periodic")
    y0 = openmc.YPlane(0.0, boundary_type="periodic")
    x0.periodic_surface = y0
    wall = openmc.ZCylinder(r=20.0, boundary_type="vacuum")
    zb = openmc.ZPlane(-10.0, boundary_type="vacuum")
    zt = openmc.ZPlane(10.0, boundary_type="vacuum")
    domain = +x0 & +y0 & -wall & +zb & -zt

    box = openmc.model.RectangularParallelepiped(2.0, 8.0, 2.0, 8.0,
                                                 -10.0, 4.0)
    ring = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=14.5, b=3.0, c=1.5)
    small = openmc.XTorus(x0=12.0, y0=4.0, z0=-6.0, a=2.0, b=0.6, c=0.6)
    # a tilted ellipsoid (p - c)' M (p - c) = 1 round c = (5, 5, 7), semi-
    # axes about 2.6, 1.8 and 1.5 cm, written out as OpenMC's quadric
    M = np.array([[0.16, 0.05, 0.025], [0.05, 0.44, -0.03],
                  [0.025, -0.03, 0.31]])
    c = np.array([5.0, 5.0, 7.0])
    lin = -2.0 * M @ c
    egg = openmc.Quadric(a=M[0, 0], b=M[1, 1], c=M[2, 2], d=2 * M[0, 1],
                         e=2 * M[1, 2], f=2 * M[0, 2], g=lin[0], h=lin[1],
                         j=lin[2], k=c @ M @ c - 1.0)
    cells = [
        openmc.Cell(name="lattice", fill=lat, region=domain & -box),
        openmc.Cell(name="ring", fill=m["ring"], region=domain & -ring),
        openmc.Cell(name="small-ring", fill=m["smallring"],
                    region=domain & -small),
        openmc.Cell(name="egg", fill=m["egg"], region=domain & -egg),
        openmc.Cell(name="shell", fill=m["shell"],
                    region=domain & +box & +ring & +small & +egg),
    ]
    geometry = openmc.Geometry(cells)
    tallies = openmc.Tallies([
        common.flux_tally("total", scores=ALL_SCORES),
        common.flux_tally("cell", openmc.CellFilter(cells + [pin])),
        common.flux_tally("material", openmc.MaterialFilter(
            list(m.values()))),
        common.flux_tally("mesh", common.mesh_filter(
            (0.0, 0.0, -10.0), (20.0, 20.0, 10.0), (20, 20, 10))),
    ])
    src = common.box_source((3.0, 3.0, -4.0), (7.0, 7.0, 2.0))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
