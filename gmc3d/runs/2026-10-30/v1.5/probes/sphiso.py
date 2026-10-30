import sys; sys.path.insert(0, "/home/claude/mtech/gmc3d")
from core import tallies as T
import numpy as np, openmc
from numba import njit, prange
from core import openmc_import, transport
from problems import PROBLEMS
from checks import openmc_ref
@njit(parallel=True)
def exact(L, nb, nd, nt, R, sa):
    out = np.zeros((nd, nb)); ga = np.pi * (3 - np.sqrt(5.0))
    for i in prange(nd):
        zc = 1 - 2 * (i + 0.5) / nd; rr = np.sqrt(1 - zc*zc)
        u, v, w = rr*np.cos(ga*i), rr*np.sin(ga*i), zc
        dt = R / nt
        for k in range(nt):
            t = (k + 0.5) * dt
            b = T._point_bin(L, 0, t*u, t*v, t*w)
            if b >= 0: out[i, b] += np.exp(-sa*t) * dt
    return out.sum(axis=0) / nd
cases = {
 "A origin0 theta":  dict(theta_grid=[0, 0.8, np.pi/2, 2.4, np.pi], phi_grid=[0, 2*np.pi], origin=(0,0,0)),
 "B offset phi":     dict(theta_grid=[0, np.pi], phi_grid=np.linspace(0, 2*np.pi, 5), origin=(1.0,-0.5,0.5)),
 "C offset theta":   dict(theta_grid=[0, 0.8, np.pi/2, 2.4, np.pi], phi_grid=[0, 2*np.pi], origin=(1.0,-0.5,0.5)),
 "D offset r only":  dict(theta_grid=[0, np.pi], phi_grid=[0, 2*np.pi], origin=(1.0,-0.5,0.5)),
 "E z-offset theta": dict(theta_grid=[0, 0.8, np.pi/2, 2.4, np.pi], phi_grid=[0, 2*np.pi], origin=(0,0,0.5)),
 "F x-offset theta": dict(theta_grid=[0, 0.8, np.pi/2, 2.4, np.pi], phi_grid=[0, 2*np.pi], origin=(1.0,0,0)),
}
for j, (name, kw) in enumerate(cases.items()):
    model = PROBLEMS["sphere"](sig_s=1e-9, sig_a=0.1)
    mesh = openmc.SphericalMesh(r_grid=np.linspace(0, 12, 6), **kw)
    t = openmc.Tally(name="s"); t.filters = [openmc.MeshFilter(mesh)]; t.scores = ["flux"]
    model.tallies = [t]
    pr = openmc_import.load(model)
    L = transport.physics(pr)[1]
    nb = pr.tallies[0].nbins
    ex = exact(L, nb, 200_000, 4000, 10.0, 0.1)
    ref = openmc_ref.run(model, f"{sys.argv[1]}/omc_iso{j}", 100_000, 20, seed=11, reuse=False)
    om, se = ref.tallies["s"]["mean"][:, 0], ref.tallies["s"]["se"][:, 0]
    ok = (ex > 1e-6)
    z = (om - ex)[ok] / se[ok]
    print(f"{name:18s} OpenMC vs exact: chi2/dof {np.mean(z**2):9.1f}  max|z| {np.abs(z).max():7.1f}", flush=True)
