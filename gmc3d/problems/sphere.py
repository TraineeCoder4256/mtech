"""Homogeneous sphere with a point source at its centre.

The simplest problem with known answers (checks/analytic.py):
  * pure absorber: leakage = exp(-sigma_a R) exactly;
  * any material: absorption + leakage = 1 (particle conservation);
  * `box=True` turns the sphere into a cube with reflective walls -- an
    infinite medium, where every particle is absorbed, so total flux is
    exactly 1/sigma_a and total absorption exactly 1.

It is also where balls are biggest: away from the edge, a ball can be
almost the whole sphere.
"""
import openmc

from . import common


def build(radius=10.0, sig_s=1.0, sig_a=0.1, box=False, mesh=20):
    materials, m = common.one_group_materials(
        "sphere", {"medium": (sig_s, sig_a)})
    if box:
        planes = [openmc.XPlane(-radius), openmc.XPlane(radius),
                  openmc.YPlane(-radius), openmc.YPlane(radius),
                  openmc.ZPlane(-radius), openmc.ZPlane(radius)]
        for p in planes:
            p.boundary_type = "reflective"
        region = (+planes[0] & -planes[1] & +planes[2] & -planes[3] &
                  +planes[4] & -planes[5])
    else:
        region = -openmc.Sphere(r=radius, boundary_type="vacuum")
    cell = openmc.Cell(name="medium", fill=m["medium"], region=region)
    geometry = openmc.Geometry([cell])
    lo, hi = (-radius,) * 3, (radius,) * 3
    tallies = openmc.Tallies([
        common.flux_tally("total"),
        common.flux_tally("cell", openmc.CellFilter([cell])),
        common.flux_tally("mesh", common.mesh_filter(lo, hi, (mesh,) * 3)),
    ])
    settings = common.settings(common.point_source((0.0, 0.0, 0.0)))
    return openmc.Model(geometry, materials, settings, tallies)
