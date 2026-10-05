"""Nesting: a 3D lattice of pin cells, one universe rotated and shifted.

  * pin universe: an absorbing z-cylinder (r = 0.6 cm) in a moderator
  * offset universe: the same pin, placed through a cell with a translation
    and a rotation, so its pin sits off-centre and tilted
  * a 3x3x2 RectLattice (pitch 2 cm) of those universes, with an outer
    universe of moderator around it, inside a reflector shell

Exercises every level of the descent: lattice element lookup, element
walls in the distance field, fill transforms, and the outer universe.
"""
import openmc

from . import common


def build():
    materials, m = common.one_group_materials(
        "nested", {"moderator": (1.2, 0.01), "pin": (0.4, 0.8),
                   "reflector": (0.9, 0.003)})
    pin = openmc.ZCylinder(r=0.6)
    pin_u = openmc.Universe(name="pin", cells=[
        openmc.Cell(name="pin", fill=m["pin"], region=-pin),
        openmc.Cell(name="pin-mod", fill=m["moderator"], region=+pin)])
    tilted_pin = openmc.YCylinder(x0=0.3, z0=0.0, r=0.4)
    inner_u = openmc.Universe(name="tilted", cells=[
        openmc.Cell(name="tilted-pin", fill=m["pin"], region=-tilted_pin),
        openmc.Cell(name="tilted-mod", fill=m["moderator"],
                    region=+tilted_pin)])
    holder = openmc.Cell(name="holder", fill=inner_u)
    holder.translation = (0.2, -0.1, 0.0)
    holder.rotation = (0.0, 0.0, 30.0)
    offset_u = openmc.Universe(name="offset", cells=[holder])
    mod_u = openmc.Universe(name="outer", cells=[
        openmc.Cell(name="outer-mod", fill=m["moderator"])])

    lat = openmc.RectLattice(name="pins")
    lat.lower_left = (-3.0, -3.0, -2.0)
    lat.pitch = (2.0, 2.0, 2.0)
    layer = [[pin_u, pin_u, pin_u], [pin_u, offset_u, pin_u],
             [pin_u, pin_u, pin_u]]
    lat.universes = [layer, [[pin_u] * 3, [pin_u, pin_u, offset_u],
                             [pin_u] * 3]]
    lat.outer = mod_u

    core = openmc.model.RectangularParallelepiped(-4, 4, -4, 4, -3, 3)
    shell = openmc.model.RectangularParallelepiped(
        -10, 10, -10, 10, -10, 10, boundary_type="vacuum")
    cells = [openmc.Cell(name="core", fill=lat, region=-core),
             openmc.Cell(name="reflector", fill=m["reflector"],
                         region=-shell & +core)]
    geometry = openmc.Geometry(cells)
    tallies = openmc.Tallies([
        common.flux_tally("total"),
        common.flux_tally("material", openmc.MaterialFilter(
            list(m.values()))),
        common.flux_tally("mesh", common.mesh_filter(
            (-10, -10, -10), (10, 10, 10), (10, 10, 10))),
    ])
    src = common.point_source((0.5, 0.5, 0.5))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
