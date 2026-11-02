import sys, warnings
sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import numpy as np, openmc
from numba import njit
from problems import common
from checks import openmc_ref
from core import openmc_import as OI, transport as TR

ST, SA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5, float(sys.argv[3]) if len(sys.argv) > 3 else 0.125
def model():
    mats, m = common.one_group_materials("mut", {"a": (ST - SA, SA)})
    s = openmc.Sphere(r=6.0, boundary_type="vacuum")
    c = openmc.Cell(fill=m["a"], region=-s)
    t = openmc.Tally(name="mu"); t.filters = [openmc.MuFilter([-1, -0.5, 0, 0.5, 1])]
    t.scores = ["flux"]
    st = common.settings(common.point_source((0, 0, 0)))
    return openmc.Model(openmc.Geometry([c]), mats, st, openmc.Tallies([t]))

@njit
def analog(n, seed):
    np.random.seed(seed)
    out = np.zeros(4); mu = 0.0
    for _ in range(n):
        x = y = z = 0.0
        c = 2*np.random.random()-1; ph = 2*np.pi*np.random.random(); s = np.sqrt(1-c*c)
        u, v, w = s*np.cos(ph), s*np.sin(ph), c
        while True:
            d = -np.log(np.random.random())/ST
            b = x*u+y*v+z*w; cc = x*x+y*y+z*z-36.0
            dl = -b + np.sqrt(b*b-cc)
            k = min(int((mu+1)*2), 3)
            if dl < d:
                out[k] += dl; break
            out[k] += d
            x += d*u; y += d*v; z += d*w
            if np.random.random() < SA/ST: break
            c = 2*np.random.random()-1; ph = 2*np.pi*np.random.random(); s = np.sqrt(1-c*c)
            nu, nv, nw = s*np.cos(ph), s*np.sin(ph), c
            mu = u*nu+v*nv+w*nw
            u, v, w = nu, nv, nw
    return out/n

which = sys.argv[1]
if which == "analog":
    print("analog", np.round(analog(2_000_000, 1), 4))
elif which == "ours":
    r = TR.run(OI.load(model()), 50000, 20, seed=2)
    print("ours", np.round(r.tallies["mu"]["mean"][:, 0], 4), np.round(r.tallies["mu"]["se"][:, 0], 4))
else:
    ref = openmc_ref.run(model(), "omc", 100000, 20, seed=5)
    print("openmc", np.round(ref.tallies["mu"]["mean"][:, 0], 4), np.round(ref.tallies["mu"]["se"][:, 0], 4))
