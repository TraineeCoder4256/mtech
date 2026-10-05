"""Today's 2D checkerboard lattice, rebuilt in 3D: the negative control.

The 7x7 cm lattice of gmc2d (mc.lattice) as an OpenMC RectLattice of 1 cm
universes, one centimetre tall with REFLECTIVE top and bottom.  Reflecting
in z makes the 3D problem identical to the 2D one, so the answer can be
checked against gmc2d's own Monte Carlo as well as OpenMC.

Every cell is about one mean free path across, so balls are small: the ball
step should gain nothing here.  `scale` multiplies all cross sections, as
in gmc2d.
"""
import openmc

from . import common

ABSORBERS = [(1, 1), (3, 1), (5, 1), (2, 2), (4, 2), (1, 3), (5, 3),
             (2, 4), (4, 4), (1, 5), (5, 5)]       # (ix, iy), as gmc2d


def build(scale=1.0, mesh_per_cm=1):
    materials, m = common.one_group_materials(
        "lattice3d", {"scatterer": (1.0 * scale, 0.0),
                      "absorber": (0.5 * scale, 9.5 * scale)})
    universes = {}
    for name in ("scatterer", "absorber"):
        universes[name] = openmc.Universe(
            name=name, cells=[openmc.Cell(name=name, fill=m[name])])
    lat = openmc.RectLattice(name="checkerboard")
    lat.lower_left = (0.0, 0.0)
    lat.pitch = (1.0, 1.0)
    rows = []
    for iy in reversed(range(7)):                   # OpenMC: top row first
        rows.append([universes["absorber" if (ix, iy) in ABSORBERS
                               else "scatterer"] for ix in range(7)])
    lat.universes = rows
    box = openmc.model.RectangularParallelepiped(
        0.0, 7.0, 0.0, 7.0, -0.5, 0.5, boundary_type="vacuum")
    box.zmin.boundary_type = "reflective"
    box.zmax.boundary_type = "reflective"
    root = openmc.Cell(name="lattice", fill=lat, region=-box)
    geometry = openmc.Geometry([root])
    n = 7 * mesh_per_cm
    tallies = openmc.Tallies([
        common.flux_tally("total"),
        common.flux_tally("material", openmc.MaterialFilter(
            [m["scatterer"], m["absorber"]])),
        common.flux_tally("mesh", common.mesh_filter(
            (0.0, 0.0, -0.5), (7.0, 7.0, 0.5), (n, n, 1))),
    ])
    src = common.box_source((3.0, 3.0, -0.5), (4.0, 4.0, 0.5))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
