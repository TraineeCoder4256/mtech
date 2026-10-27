"""Flux maps: our mesh tallies next to OpenMC's, and where they disagree.

Used by: run.py (`python run.py maps`).

For every test problem: OpenMC (the reference), then ours with a ball
backend (the trained network by default) under the 'profile' mesh rule.
One PNG per problem, three rows:

    GMC        our flux per voxel (log colour scale)
    OpenMC     the same, on the same colour scale
    z          (ours - OpenMC) / combined error, voxel by voxel, -4 to +4:
               white where they agree, red where ours is high, blue low.
               Honest noise looks like salt and pepper; a real error shows
               up as a coloured patch or ring.

3D meshes are shown as the three mid-planes through the mesh (XY, XZ, YZ);
the 7 x 7 x 1 lattice mesh as one plane; the slab's 60 depth bins as a
curve with error bars and z underneath.  The voxels OpenMC never scored
(outside the geometry) are blank.  The numbers behind the pictures
(chi2/dof, max |z|) are those of checks/compare.py and go in maps.json.

`plot` works from plain tally dicts (mean and se arrays plus the mesh), so
a map can also be drawn from a saved run (run.py solve writes them as
JSON), which is how the v1.0 'before' maps were made.
"""

from pathlib import Path

import numpy as np

from core import openmc_import, transport
from problems import PROBLEMS
from . import compare, openmc_ref, validate

ZMAX = 4.0


def _grid(t, col=0):
    dim = np.asarray(t["mesh"][2])
    return np.asarray(t["mean"])[:, col].reshape(dim[::-1]), \
        np.asarray(t["se"])[:, col].reshape(dim[::-1])


def plot(ours, ref, path, title, label="GMC"):
    """ours, ref: tally dicts with 'mesh', 'mean', 'se'.  Writes a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    a, a_se = _grid(ours)
    b, b_se = _grid(ref)
    z, ok = compare.z_scores(a.ravel(), a_se.ravel(), b.ravel(),
                             b_se.ravel())
    z = z.reshape(a.shape)
    ok = ok.reshape(a.shape)
    zk = z[ok]
    stats = (f"chi2/dof {np.mean(zk ** 2):.3f}, max|z| "
             f"{np.max(np.abs(zk)):.2f}, |z|>3 in "
             f"{100 * np.mean(np.abs(zk) > 3):.2f}% of {zk.size} voxels")
    ll, ur, dim = (np.asarray(v, float) for v in ours["mesh"])
    nz, ny, nx = a.shape

    if ny == 1 and nz == 1:                         # a depth profile
        x = ll[0] + (np.arange(nx) + 0.5) * (ur[0] - ll[0]) / nx
        fig, (ax, axz) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        ax.errorbar(x, b[0, 0], 2 * b_se[0, 0], fmt="o", ms=3, color="k",
                    label="OpenMC (2 sigma)")
        ax.errorbar(x, a[0, 0], 2 * a_se[0, 0], fmt="-", lw=1.2,
                    color="tab:red", label=f"{label} (2 sigma)")
        ax.set_yscale("log")
        ax.set_ylabel("flux per source particle")
        ax.legend()
        axz.bar(x, np.where(ok[0, 0], z[0, 0], 0), width=0.8 * (x[1] - x[0]),
                color=np.where(np.abs(z[0, 0]) > 3, "tab:red", "tab:gray"))
        for h in (-3, 3):
            axz.axhline(h, ls=":", color="k", lw=0.8)
        axz.set_ylim(-ZMAX - 1, ZMAX + 1)
        axz.set_ylabel("z")
        axz.set_xlabel("x (cm)")
        fig.suptitle(f"{title}\n{stats}", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return stats

    planes = [("XY", lambda g: g[nz // 2], (0, 1))]
    if nz > 1:
        planes += [("XZ", lambda g: g[:, ny // 2, :], (0, 2)),
                   ("YZ", lambda g: g[:, :, nx // 2], (1, 2))]
    pos = b[ok & (b > 0)] if (ok & (b > 0)).any() else np.array([1.0])
    norm = LogNorm(vmin=max(pos.min(), pos.max() * 1e-6), vmax=pos.max())
    fig, axes = plt.subplots(3, len(planes), squeeze=False,
                             figsize=(4.2 * len(planes) + 1, 11))
    for c, (pname, cut, (h, v)) in enumerate(planes):
        ext = (ll[h], ur[h], ll[v], ur[v])
        mask = ~cut(ok)
        for r, (g, cmap, nrm, lab) in enumerate((
                (a, "viridis", norm, label), (b, "viridis", norm, "OpenMC"),
                (z, "RdBu_r", None, "z"))):
            img = np.ma.array(cut(g), mask=mask)
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap,
                           norm=nrm, vmin=None if nrm else -ZMAX,
                           vmax=None if nrm else ZMAX,
                           interpolation="nearest")
            ax.set_title(f"{lab}, {pname} mid-plane", fontsize=9)
            ax.set_xlabel("xyz"[h] + " (cm)")
            ax.set_ylabel("xyz"[v] + " (cm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="flux" if nrm else "z")
    fig.suptitle(f"{title}\n{stats}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return stats


def mesh_name(results):
    for name, t in results.tallies.items():
        if t["filter"] == "mesh":
            return name
    return None


def run(names, backend, outdir, workdir, particles=100_000, batches=20,
        seed=5, r_star=3.0, log=print):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in names:
        model = PROBLEMS[name]()
        pr = openmc_import.load(model)
        ref = openmc_ref.run(model, f"{workdir}/openmc_{name}",
                             particles * batches // validate.REF_BATCHES,
                             validate.REF_BATCHES, seed=seed + 100)
        res = transport.run(pr, particles, batches, seed=seed,
                            backend=backend, r_star=r_star)
        cmp = compare.compare(res, ref)
        m = mesh_name(res)
        stats = plot(res.tallies[m], {**res.tallies[m], **ref.tallies[m]},
                     outdir / f"{name}.png",
                     f"{name}: {backend.name}, R* = {r_star:g}, 'profile' "
                     f"mesh rule, {particles * batches:,} histories each",
                     label=f"GMC ({backend.name})")
        log(f"\n== {name}: {res.counters['balls']:,} balls, transport "
            f"{res.timing['transport_s']:.1f} s + backend "
            f"{res.timing['backend_s']:.1f} s")
        log(compare.report(cmp, f"   {backend.name} vs OpenMC"))
        log(f"   map: {outdir / (name + '.png')}  ({stats})")
        out[name] = {"vs_openmc": cmp, "counters": res.counters,
                     "timing": res.timing, "map": f"{name}.png"}
    return out
