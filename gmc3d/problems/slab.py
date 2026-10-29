"""A thick shield slab: the headline problem for Stage 1.

A 60 cm slab in x of a scattering, weakly absorbing material, infinite in y
and z (reflective walls at +-100 cm).  Particles are born isotropically in
the first centimetre and diffuse through.  Deep inside, the nearest
boundary is far away, so balls of tens of mean free paths fit: the regime
where a ball step can pay.

Tallies: a 60-bin depth profile (mesh), total flux/absorption, and (since
v1.1) the current through each face: x = 0 (reflection, negative: leaving
towards -x) and x = 60 (transmission, positive).  Since v1.1.3 the total
tally also scores scatter and total, and a 1D RegularMesh (30 bins in x,
unbounded in y and z, all four scores) checks the 1D mesh path.  Since
v1.3 the face currents are also scored with a cell-from and with a cell
filter (on a vacuum face both are the slab itself).
"""
import openmc

from . import common

ALL_SCORES = ("flux", "absorption", "scatter", "total")


def build(thickness=60.0, sig_s=0.5, sig_a=0.005, half_width=100.0, bins=60):
    materials, m = common.one_group_materials(
        "slab", {"shield": (sig_s, sig_a)})
    x0 = openmc.XPlane(0.0, boundary_type="vacuum")
    x1 = openmc.XPlane(thickness, boundary_type="vacuum")
    walls = [openmc.YPlane(-half_width), openmc.YPlane(half_width),
             openmc.ZPlane(-half_width), openmc.ZPlane(half_width)]
    for p in walls:
        p.boundary_type = "reflective"
    region = +x0 & -x1 & +walls[0] & -walls[1] & +walls[2] & -walls[3]
    cell = openmc.Cell(name="shield", fill=m["shield"], region=region)
    geometry = openmc.Geometry([cell])
    tallies = openmc.Tallies([
        common.flux_tally("total", scores=ALL_SCORES),
        common.flux_tally("depth1d", common.mesh_filter(
            (0.0,), (thickness,), (bins // 2,)), scores=ALL_SCORES),
        common.flux_tally("depth", common.mesh_filter(
            (0.0, -half_width, -half_width),
            (thickness, half_width, half_width), (bins, 1, 1))),
        common.flux_tally("sides", openmc.SurfaceFilter([x0, x1]),
                          scores=("current",)),
        # v1.3: a vacuum face's cell and cell-from are both the slab
        common.flux_tally("sides_from", openmc.CellFromFilter([cell]),
                          scores=("current",)),
        common.flux_tally("sides_cell", openmc.SurfaceFilter([x0, x1]),
                          scores=("current",)),
    ])
    tallies[4].filters = tallies[4].filters + [openmc.SurfaceFilter([x0, x1])]
    tallies[5].filters = tallies[5].filters + [openmc.CellFilter([cell])]
    src = common.box_source((0.0, -half_width, -half_width),
                            (1.0, half_width, half_width))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
