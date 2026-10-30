import sys; sys.path.insert(0, "/home/claude/mtech/gmc3d")
import numpy as np, openmc
from core import openmc_import, transport
from problems import common
from checks import openmc_ref
def build(mode):
    mats, m = common.one_group_materials("probe", {"mod": (1.0, 0.01), "abs": (0.2, 2.0)})
    sph = openmc.Sphere(r=2.0)
    inner = [openmc.Cell(name="abs", fill=m["abs"], region=-sph),
             openmc.Cell(name="modin", fill=m["mod"], region=+sph)]
    box = openmc.model.RectangularParallelepiped(-10, 10, -10, 10, -10, 10, boundary_type="vacuum")
    if mode == "root":
        cells = [openmc.Cell(fill=m["abs"], region=-sph), openmc.Cell(fill=m["mod"], region=+sph & -box)]
    else:
        u = openmc.Universe(cells=inner)
        if mode == "universe":
            cells = [openmc.Cell(fill=u, region=-box)]
        elif mode == "shifted":
            c = openmc.Cell(fill=u, region=-box); c.translation = (2.0, 1.0, 0.0); cells = [c]
        elif mode == "rotated":
            c = openmc.Cell(fill=u, region=-box); c.rotation = (0.0, 0.0, 30.0); cells = [c]
        elif mode == "offcentre":
            lat = openmc.RectLattice(); lat.lower_left = (-6, -10); lat.pitch = (20, 20); lat.universes = [[u]]
            cells = [openmc.Cell(fill=lat, region=-box)]
        elif mode == "lattice":
            lat = openmc.RectLattice(); lat.lower_left = (-10, -10); lat.pitch = (20, 20); lat.universes = [[u]]
            cells = [openmc.Cell(fill=lat, region=-box)]
        elif mode == "lattice3":
            lat = openmc.RectLattice(); lat.lower_left = (-12, -12); lat.pitch = (8, 8); lat.universes = [[u]*3]*3
            cells = [openmc.Cell(fill=lat, region=-box)]
    t = openmc.Tally(name="cur"); t.filters = [openmc.SurfaceFilter([sph])]; t.scores = ["current"]
    t2 = openmc.Tally(name="ab"); t2.scores = ["absorption"]; t2.filters = [openmc.MaterialFilter([m["abs"]])]
    return openmc.Model(openmc.Geometry(cells), mats, common.settings(common.point_source((5.0, 0.3, 0.2))), [t, t2])
for mode in ("shifted", "rotated", "offcentre"):
    model = build(mode)
    ref = openmc_ref.run(model, f"{sys.argv[1]}/inner2_{mode}", 20_000, 20, seed=3, reuse=False)
    ours = transport.run(openmc_import.load(build(mode)), 100_000, 8, seed=2)
    print(f"{mode:9s} OpenMC current {ref.tallies['cur']['mean'][0,0]:+.4f} abs {ref.tallies['ab']['mean'][0,1]:.4f} | "
          f"ours current {ours.tallies['cur']['mean'][0,0]:+.4f} abs {ours.tallies['ab']['mean'][0,1]:.4f}", flush=True)
