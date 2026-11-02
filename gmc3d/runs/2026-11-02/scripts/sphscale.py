"""Does the spherical-mesh tally cost grow with the NUMBER OF BINS, or
with the number of EDGE SURFACES squared?

`_mesh_segment` (Cartesian) steps wall to wall and updates one axis, O(1)
a step.  `_next_cut` (curved) re-solves every r sphere, every phi
half-plane and every theta cone on every step.  If that is what costs,
refining r must grow our time faster than linearly in bins, while
OpenMC's grows linearly.  Only the tally definition changes here.
"""
import json, sys, time
sys.path.insert(0, ".")
import numpy as np
import openmc
from checks import openmc_ref
from core import openmc_import, transport as TR
from problems import PROBLEMS

N, B, REF_B = 100_000, 5, 400
out = {}


def build(nr):
    m = PROBLEMS["sphere"]()
    src = [t for t in m.tallies if t.name == "sphmesh"][0]
    mesh = src.filters[0].mesh
    lo, hi = float(mesh.r_grid[0]), float(mesh.r_grid[-1])
    new = openmc.SphericalMesh(r_grid=np.linspace(lo, hi, nr + 1),
                               theta_grid=mesh.theta_grid,
                               phi_grid=mesh.phi_grid, origin=mesh.origin)
    t = openmc.Tally(name="sphmesh")
    t.filters = [openmc.MeshFilter(new)]
    t.scores = ["flux", "absorption"]
    m.tallies = openmc.Tallies([t])
    return m, int(t.num_bins)


for nr in (1, 2, 5, 10, 20, 40):
    m, nb = build(nr)
    pr = openmc_import.load(m)
    TR.run(pr, 1000, 1, seed=1)
    o = 1e30
    for k in range(2):
        t0 = time.perf_counter()
        TR.run(pr, N, B, seed=9)
        o = min(o, time.perf_counter() - t0)
    m2, _ = build(nr)
    m2.settings.survival_biasing = True
    m2.settings.cutoff = {"weight": 0.25, "weight_avg": 1.0}
    p = openmc_ref.run(m2, f"/tmp/ss_{nr}", max(N * B // REF_B, 1), REF_B,
                       seed=101, reuse=False).timing["openmc_transport_s"]
    out[nr] = {"ours_s": o, "openmc_s": p, "bins": nb}
    print(f"SS r_bins {nr:3d}  bins {nb:5d}  ours {o:7.2f} s   "
          f"openmc {p:7.2f} s", flush=True)
    json.dump(out, open("/tmp/claude-0/-home-claude-mtech/"
                        "59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad/"
                        "sphscale.json", "w"), indent=1)
