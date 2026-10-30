import sys; sys.path.insert(0, "/home/claude/mtech/gmc3d")
from core import tallies as T
import numpy as np, openmc
from numba import njit, prange
from core import openmc_import, transport
from problems import PROBLEMS, common
from checks import openmc_ref
@njit(parallel=True)
def exact(L, nb, nd, nt, R, sa, s0, s1, s2):
    out = np.zeros((nd, nb)); ga = np.pi * (3 - np.sqrt(5.0))
    for i in prange(nd):
        zc = 1 - 2 * (i + 0.5) / nd; rr = np.sqrt(1 - zc*zc)
        u, v, w = rr*np.cos(ga*i), rr*np.sin(ga*i), zc
        # distance to the sphere of radius R about 0 from s
        b = s0*u + s1*v + s2*w; c = s0*s0 + s1*s1 + s2*s2 - R*R
        tmax = -b + np.sqrt(b*b - c)
        dt = tmax / nt
        for k in range(nt):
            t = (k + 0.5) * dt
            bb = T._point_bin(L, 0, s0 + t*u, s1 + t*v, s2 + t*w)
            if bb >= 0: out[i, bb] += np.exp(-sa*t) * dt
    return out.sum(axis=0) / nd
src = (-1.0, 0.5, -0.5)
for j, (name, kw) in enumerate({
    "theta, origin 0, source off": dict(theta_grid=[0, 0.8, np.pi/2, 2.4, np.pi], phi_grid=[0, 2*np.pi]),
    "theta only pi/2 (plane)": dict(theta_grid=[0, np.pi/2, np.pi], phi_grid=[0, 2*np.pi]),
    "theta only 0.8 (one cone)": dict(theta_grid=[0, 0.8, np.pi], phi_grid=[0, 2*np.pi]),
}.items()):
    model = PROBLEMS["sphere"](sig_s=1e-9, sig_a=0.1)
    model.settings.source = common.point_source(src)
    mesh = openmc.SphericalMesh(r_grid=np.linspace(0, 10, 6), **kw)
    t = openmc.Tally(name="s"); t.filters = [openmc.MeshFilter(mesh)]; t.scores = ["flux"]
    model.tallies = [t]
    pr = openmc_import.load(model)
    L = transport.physics(pr)[1]
    nb = pr.tallies[0].nbins
    ex = exact(L, nb, 200_000, 4000, 10.0, 0.1, *src)
    ours = transport.run(pr, 200_000, 10, seed=5)
    ref = openmc_ref.run(model, f"{sys.argv[1]}/omc_iso2_{j}", 100_000, 20, seed=11, reuse=False)
    for lab, rr in (("ours", ours), ("OpenMC", ref)):
        m, se = rr.tallies["s"]["mean"][:, 0], rr.tallies["s"]["se"][:, 0]
        ok = ex > 1e-6
        z = (m - ex)[ok] / np.maximum(se[ok], 1e-300)
        print(f"{name:28s} {lab:6s} vs exact: chi2/dof {np.mean(z**2):9.1f}  max|z| {np.abs(z).max():7.1f}", flush=True)
