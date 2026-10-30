import sys; sys.path.insert(0, "/home/claude/mtech/gmc3d")
import numpy as np, openmc, openmc.stats as st
from core import openmc_import, transport
from problems import common
from checks import openmc_ref
def build(pos, d):
    mats, m = common.one_group_materials("rotb", {"mod": (1e-6, 0.01)})
    x0 = openmc.XPlane(0.0, boundary_type="periodic"); y0 = openmc.YPlane(0.0, boundary_type="periodic")
    x0.periodic_surface = y0
    wall = openmc.ZCylinder(r=10.0, boundary_type="vacuum")
    zb, zt = openmc.ZPlane(-5.0, boundary_type="vacuum"), openmc.ZPlane(5.0, boundary_type="vacuum")
    cells = [openmc.Cell(fill=m["mod"], region=+x0 & +y0 & -wall & +zb & -zt)]
    t = openmc.Tally(name="w"); t.filters = [openmc.SurfaceFilter([x0, y0])]; t.scores = ["current"]
    s = openmc.IndependentSource(space=st.Point(pos), angle=st.Monodirectional(d))
    return openmc.Model(openmc.Geometry(cells), mats, common.settings(s), [t])
for pos, d, lab in (((0.5, 5.0, 0.0), (-1.0, 0.0, 0.0), "out via x0"), ((5.0, 0.5, 0.0), (0.0, -1.0, 0.0), "out via y0")):
    model = build(pos, d)
    ref = openmc_ref.run(model, f"{sys.argv[1]}/rotb_{lab[-2:]}", 2000, 5, seed=3, reuse=False)
    ours = transport.run(openmc_import.load(build(pos, d)), 2000, 5, seed=2)
    print(f"{lab}: OpenMC [x0, y0] {np.round(ref.tallies['w']['mean'][:,0],4)} | ours {np.round(ours.tallies['w']['mean'][:,0],4)}", flush=True)
