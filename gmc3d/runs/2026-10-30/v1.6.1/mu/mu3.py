import sys, warnings
sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import numpy as np, openmc
from problems import common
from core import openmc_import as OI, transport as TR
mats, m = common.one_group_materials("mut3", {"a": (0.5, 1e-6)})
s = openmc.Sphere(r=6.0, boundary_type="vacuum")
c = openmc.Cell(fill=m["a"], region=-s)
t = openmc.Tally(name="mu"); t.filters = [openmc.CollisionFilter([0, 1]), openmc.MuFilter([-1, -0.5, 0, 0.5, 1])]
t.scores = ["flux"]
st = common.settings(common.point_source((0, 0, 0)))
md = openmc.Model(openmc.Geometry([c]), mats, st, openmc.Tallies([t]))
r = TR.run(OI.load(md), int(sys.argv[2]), 4, seed=2, nchunk=int(sys.argv[1]))
print(np.round(r.tallies["mu"]["mean"][:, 0], 4), r.counters)
