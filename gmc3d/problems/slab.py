"""A thick shield slab: the headline problem for Stage 1.

A 60 cm slab in x of a scattering, weakly absorbing material, infinite in y
and z (reflective walls at +-100 cm).  Particles are born isotropically in
the first centimetre and diffuse through.  Deep inside, the nearest
boundary is far away, so balls of tens of mean free paths fit: the regime
where a ball step can pay.

Tallies: a 60-bin depth profile (mesh), total leakage-relevant flux, and
total flux/absorption.
"""
import openmc

from . import common


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
        common.flux_tally("total"),
        common.flux_tally("depth", common.mesh_filter(
            (0.0, -half_width, -half_width),
            (thickness, half_width, half_width), (bins, 1, 1))),
    ])
    src = common.box_source((0.0, -half_width, -half_width),
                            (1.0, half_width, half_width))
    return openmc.Model(geometry, materials, common.settings(src), tallies)
