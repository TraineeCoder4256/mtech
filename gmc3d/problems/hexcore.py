"""Two hexagonal lattices, one of each orientation (v1.4.1).

A small core in a reflector, 24 x 24 x 30 cm, vacuum outside:

  * lower half (z -10 to 0): a 3D HexLattice with flat tops (OpenMC's 'y'
    orientation), 3 rings, pitch 2.2 cm, two axial layers of 5 cm.  Most
    elements are fuel pins, some are control rods.  Its outer universe is
    NOT uniform: a small rod sphere sits at the centre of every would-be
    element, so it checks that the outer universe is tiled element by
    element, as OpenMC does
  * upper half (z 0 to 10): a 2D HexLattice with flat sides ('x'), 3 rings,
    off-centre by (0.3, -0.2) cm, with a plain moderator outside it
  * both lattices sit in a cylinder of r = 7 cm (so the outer universe
    shows between the hexagons and the cylinder), inside a reflector

Exercises the hexagonal element lookup, the six walls in the distance
field and in flights, and the axial walls of a 3D hex lattice.
"""
import openmc

from . import common

ALL_SCORES = ("flux", "absorption", "scatter", "total")


def _pin(name, mat, mod, r):
    s = openmc.ZCylinder(r=r)
    return openmc.Universe(name=name, cells=[
        openmc.Cell(name=f"{name}-in", fill=mat, region=-s),
        openmc.Cell(name=f"{name}-mod", fill=mod, region=+s)])


def build():
    materials, m = common.one_group_materials(
        "hexcore", {"fuel": (0.6, 0.15), "moderator": (1.2, 0.01),
                    "rod": (0.5, 2.0), "reflector": (1.0, 0.005)})
    pin = _pin("pin", m["fuel"], m["moderator"], 0.6)
    rod = _pin("rod", m["rod"], m["moderator"], 0.8)
    ball = openmc.Sphere(r=0.4)
    dotted = openmc.Universe(name="dotted", cells=[
        openmc.Cell(name="dot", fill=m["rod"], region=-ball),
        openmc.Cell(name="dot-mod", fill=m["moderator"], region=+ball)])
    plain = openmc.Universe(name="plain", cells=[
        openmc.Cell(name="plain-mod", fill=m["moderator"])])

    low = openmc.HexLattice(name="low")
    low.orientation = "y"
    low.center = (0.0, 0.0, -5.0)
    low.pitch = (2.2, 5.0)
    low.universes = [
        [[pin] * 12, [pin, rod] * 3, [rod]],                # bottom layer
        [[rod] + [pin] * 11, [pin] * 6, [pin]],             # top layer
    ]
    low.outer = dotted

    high = openmc.HexLattice(name="high")
    high.orientation = "x"
    high.center = (0.3, -0.2)
    high.pitch = (2.2,)
    high.universes = [[pin, pin, rod] * 4, [rod] + [pin] * 5, [pin]]
    high.outer = plain

    core = openmc.ZCylinder(r=7.0)
    zs = [openmc.ZPlane(z) for z in (-10.0, 0.0, 10.0)]
    shell = openmc.ZCylinder(r=12.0, boundary_type="vacuum")
    zb = openmc.ZPlane(-15.0, boundary_type="vacuum")
    zt = openmc.ZPlane(15.0, boundary_type="vacuum")
    in_core = -core & +zs[0] & -zs[2]
    cells = [
        openmc.Cell(name="low", fill=low, region=-core & +zs[0] & -zs[1]),
        openmc.Cell(name="high", fill=high, region=-core & +zs[1] & -zs[2]),
        openmc.Cell(name="reflector", fill=m["reflector"],
                    region=-shell & +zb & -zt & ~in_core),
    ]
    geometry = openmc.Geometry(cells)
    tallies = openmc.Tallies([
        common.flux_tally("total", scores=ALL_SCORES),
        common.flux_tally("cell", openmc.CellFilter(cells)),
        common.flux_tally("material", openmc.MaterialFilter(
            list(m.values()))),
        common.flux_tally("mesh", common.mesh_filter(
            (-12.0, -12.0, -15.0), (12.0, 12.0, 15.0), (24, 24, 15))),
    ])
    src = common.box_source((-1.0, -1.0, -6.0), (1.0, 1.0, 4.0))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
