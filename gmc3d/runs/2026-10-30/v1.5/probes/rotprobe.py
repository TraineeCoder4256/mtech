import sys; sys.path.insert(0, "/home/claude/mtech/gmc3d")
import numpy as np, openmc
from core import openmc_import, transport
from problems import common
from checks import openmc_ref
def build(lattice):
    mats, m = common.one_group_materials("rot", {"mod": (1.0, 0.05), "abs": (0.2, 2.0)})
    x0 = openmc.XPlane(0.0, boundary_type="periodic"); y0 = openmc.YPlane(0.0, boundary_type="periodic")
    x0.periodic_surface = y0
    wall = openmc.ZCylinder(r=10.0, boundary_type="vacuum")
    zb, zt = openmc.ZPlane(-5.0, boundary_type="vacuum"), openmc.ZPlane(5.0, boundary_type="vacuum")
    dom = +x0 & +y0 & -wall & +zb & -zt
    if lattice:
        pin = openmc.ZCylinder(r=0.5)
        u = openmc.Universe(cells=[openmc.Cell(fill=m["abs"], region=-pin), openmc.Cell(fill=m["mod"], region=+pin)])
        lat = openmc.RectLattice(); lat.lower_left = (2, 2); lat.pitch = (2, 2); lat.universes = [[u]*2]*2
        box = openmc.model.RectangularParallelepiped(2, 6, 2, 6, -5, 5)
        cells = [openmc.Cell(fill=lat, region=dom & -box), openmc.Cell(fill=m["mod"], region=dom & +box)]
    else:
        cells = [openmc.Cell(fill=m["mod"], region=dom)]
    t = openmc.Tally(name="w"); t.filters = [openmc.SurfaceFilter([x0, y0])]; t.scores = ["current"]
    src = common.box_source((1.0, 1.0, -1.0), (3.0, 1.5, 1.0))
    return openmc.Model(openmc.Geometry(cells), mats, common.settings(src), [t])
for lattice in (False, True):
    model = build(lattice)
    ref = openmc_ref.run(model, f"{sys.argv[1]}/rot_{int(lattice)}", 20_000, 20, seed=3, reuse=False)
    ours = transport.run(openmc_import.load(build(lattice)), 100_000, 8, seed=2)
    print(f"lattice={lattice}: OpenMC {np.round(ref.tallies['w']['mean'][:,0],4)} +- {np.round(ref.tallies['w']['se'][:,0],4)} | ours {np.round(ours.tallies['w']['mean'][:,0],4)}", flush=True)
