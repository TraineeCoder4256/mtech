import sys; sys.path.insert(0, "/home/claude/mtech/gmc3d")
from core import tallies as T
import numpy as np, openmc
from numba import njit
@njit
def bins_of(L, pts):
    out = np.empty(pts.shape[0], np.int64)
    for i in range(pts.shape[0]):
        out[i] = T._point_bin(L, 0, pts[i,0], pts[i,1], pts[i,2])
    return out
from core.openmc_import import _mesh, Filter, Tally

def layout(m):
    box, edges, mt, org = _mesh(type("t", (), {"id": 1}), m)
    f = Filter("mesh", int(np.prod(box[2])), mesh=box, edges=edges, mesh_type=mt, origin=org)
    tl = Tally(1, "m", "mesh", ["flux"], [f])
    return T.compile_layout([tl], 1, 1), int(np.prod(box[2]))

cyl = openmc.CylindricalMesh(r_grid=[0, 2, 5, 9], phi_grid=np.linspace(0, 2*np.pi, 7),
                             z_grid=[-6, -1, 3, 8], origin=(1.0, -0.5, 0.3))
cylq = openmc.CylindricalMesh(r_grid=[1, 4, 7], phi_grid=[0.3, 1.2, 2.5],
                             z_grid=[-3, 0, 3], origin=(0.0, 0.0, 0.0))
sph = openmc.SphericalMesh(r_grid=[0, 3, 6, 10], theta_grid=[0, 0.7, np.pi/2, 2.2, np.pi],
                           phi_grid=np.linspace(0, 2*np.pi, 5), origin=(-1.0, 0.5, 2.0))
sphq = openmc.SphericalMesh(r_grid=[2, 5, 8], theta_grid=[0.4, 1.3, 2.0],
                           phi_grid=[1.0, 3.0, 5.0], origin=(0, 0, 0))
rng = np.random.default_rng(1)
for name, m in (("cyl", cyl), ("cyl-part", cylq), ("sph", sph), ("sph-part", sphq)):
    L, nb = layout(m)
    worst = 0.0; worst_room = 0.0
    for trial in range(300):
        p = rng.uniform(-12, 12, 3); d = rng.normal(size=3); d /= np.linalg.norm(d)
        length = rng.uniform(1, 30)
        row = np.zeros(L.n_entries)
        T._curved_segment(row, L, 0, L.offset[0], 0.0, 0.0, 1.0, *p, *d, length)
        got = row[L.offset[0]::4][:nb]
        n = 200_000
        ts = (np.arange(n) + 0.5) / n * length
        ref = np.zeros(nb)
        pts = p + ts[:, None] * d
        bins = bins_of(L, pts)
        ok = bins >= 0
        np.add.at(ref, bins[ok], length / n)
        worst = max(worst, np.abs(got - ref).max())
        # room: a ball of that radius about a random point holds one bin
        c = rng.uniform(-12, 12, 3)
        r = T.mesh_room(L, *c)
        b0 = T._point_bin(L, 0, *c)
        q = rng.normal(size=(2000, 3)); q /= np.linalg.norm(q, axis=1)[:, None]
        q = c + 0.999 * r * q * rng.uniform(0, 1, (2000, 1)) ** (1/3)
        bb = bins_of(L, q)
        if (bb != b0).any(): worst_room += 1
        # tightness: is some point at 1.05 r in another bin?
    print(f"{name}: worst |walk - brute| = {worst:.2e} cm over 300 flights "
          f"(brute step {30/200000:.1e}); room violations {worst_room}")
