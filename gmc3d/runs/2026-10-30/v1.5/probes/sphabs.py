import sys; sys.path.insert(0, "/home/claude/mtech/gmc3d")
from core import tallies as T
import numpy as np, openmc
from numba import njit, prange
from core import openmc_import, transport
from problems import PROBLEMS
from checks import openmc_ref
model = PROBLEMS["sphere"](sig_s=1e-9, sig_a=0.1)
model.tallies = [t for t in model.tallies if t.name == "sphmesh"]
pr = openmc_import.load(model)
L = transport.physics(pr)[1]
@njit(parallel=True)
def exact(L, nd, nt, R, sa):
    out = np.zeros((nd, 80))
    ga = np.pi * (3 - np.sqrt(5.0))
    for i in prange(nd):
        zc = 1 - 2 * (i + 0.5) / nd; rr = np.sqrt(1 - zc*zc)
        u, v, w = rr*np.cos(ga*i), rr*np.sin(ga*i), zc
        dt = R / nt
        for k in range(nt):
            t = (k + 0.5) * dt
            b = T._point_bin(L, 0, t*u, t*v, t*w)
            if b >= 0:
                out[i, b] += np.exp(-sa*t) * dt
    return out.sum(axis=0) / nd
ex = exact(L, 400_000, 4000, 10.0, 0.1)
r = transport.run(pr, 1_000_000, 10, seed=4)
ours, ose = r.tallies["sphmesh"]["mean"][:, 0], r.tallies["sphmesh"]["se"][:, 0]
ref = openmc_ref.run(model, "data/openmc/probe_sphabs", 100_000, 40, seed=11, reuse=False)
om, ose2 = ref.tallies["sphmesh"]["mean"][:, 0], ref.tallies["sphmesh"]["se"][:, 0]
ok = ex > 1e-6
zo = (ours - ex)[ok] / ose[ok]; zr = (om - ex)[ok] / ose2[ok]
print(f"ours vs exact: chi2/dof {np.mean(zo**2):.2f} max|z| {np.abs(zo).max():.2f}")
print(f"OpenMC vs exact: chi2/dof {np.mean(zr**2):.2f} max|z| {np.abs(zr).max():.2f}")
np.set_printoptions(precision=4, suppress=True, linewidth=150)
for name, a in (("exact", ex), ("ours", ours), ("openmc", om)):
    print(name, a.reshape(4,4,5)[0][:, :2].ravel())
