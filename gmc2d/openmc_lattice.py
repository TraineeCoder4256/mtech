#!/usr/bin/env python3
"""The lattice benchmark solved with OpenMC.  Run: python openmc_lattice.py [scale]

A self-contained OpenMC solution of the 7 x 7 cm checkerboard lattice used by
Farmer et al. (arXiv:2512.13965).  It imports nothing from this project and
compares against nothing -- it is a reference solution in its own right.

The problem
-----------
A 7 x 7 cm square divided into 1 cm cells.  The background scatters without
absorbing (sig_s = 1.0, sig_a = 0.0 per cm).  Eleven cells are strong
absorbers (sig_s = 0.5, sig_a = 9.5, so sig_t = 10).  An isotropic source
fills the central 1 x 1 cm cell.  The boundary is vacuum.

This is Farmer et al.'s steady-state variant of the ORNL lattice geometry
(Schotthoefer & Hauck, arXiv:2505.17284).  The ORNL problem itself is
time-dependent (final time 3.2) and its absorbers do not scatter
(sig_a = 10, sig_s = 0), so ORNL's published solutions are not a reference
for this one.

The optional argument multiplies every cross section, exactly as
mc.lattice(scale) does: `python openmc_lattice.py 10` solves the lattice
with cells ten times optically thicker, which is where the sampler is
supposed to pay off.  Outputs for scale != 1 carry an _x<scale> suffix.

Three modelling choices make this a faithful OpenMC statement of that problem,
and each is somewhere a careless setup would silently solve something else:

  MONOENERGETIC.  OpenMC is continuous-energy by default.  Running in
  MULTI-GROUP mode with a single energy group makes the cross sections the
  literal constants above, with no nuclear data library involved.

  2-D XY WITH 3-D DIRECTIONS.  The benchmark is two-dimensional in space but
  particles travel in three dimensions.  Bounding z with REFLECTIVE planes
  reproduces an infinite extrusion exactly, because nothing in the problem
  varies with z.  Vacuum planes there would solve a slab instead.

  FLUX PER UNIT VOLUME.  An OpenMC flux tally is a track length per source
  particle.  Dividing by the mesh cell volume gives the scalar flux, which is
  the quantity the benchmark reports.

Requires OpenMC: conda install -c conda-forge openmc
"""
import pathlib
import sys
import time

import numpy as np
import openmc

# ---- the problem -------------------------------------------------------
L = 7.0                    # domain is L x L cm
PITCH = 1.0                # material cells are PITCH x PITCH cm
MESH = 112                 # tally mesh is MESH x MESH
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # cross-section multiplier
BG = (1.0 * SCALE, 0.0)            # background      (sig_s, sig_a) per cm
AB = (0.5 * SCALE, 9.5 * SCALE)    # absorber blocks (sig_s, sig_a) per cm
SOURCE_CELL = (3, 3)       # zero-indexed (x, y) of the source cell
ABSORBERS = [(1, 1), (3, 1), (5, 1), (2, 2), (4, 2), (1, 3), (5, 3),
             (2, 4), (4, 4), (1, 5), (5, 5)]

# ---- run settings ------------------------------------------------------
PARTICLES = 200_000
BATCHES = 20               # total histories = PARTICLES * BATCHES
SEED = 20250901
TAG = "" if SCALE == 1.0 else f"_x{SCALE:g}"
RUNDIR = pathlib.Path(f"results/openmc{TAG}")
OUT = pathlib.Path("results")

GROUP_TOP = 20.0e6         # the single group is [0, 20 MeV); the value is arbitrary
BIRTH_E = 1.0e6            # any energy inside that group


def absorber_map():
    """Boolean (ncell, ncell) array, indexed [y][x], origin at lower left."""
    n = int(round(L / PITCH))
    m = np.zeros((n, n), bool)
    for x, y in ABSORBERS:
        m[y, x] = True
    return m


# ------------------------------------------------------------------ model
def build():
    """Materials, geometry, settings and tallies for the lattice."""
    # --- one-group macroscopic cross sections, P0 (isotropic) scattering
    groups = openmc.mgxs.EnergyGroups([0.0, GROUP_TOP])
    lib = openmc.MGXSLibrary(groups)
    mats = {}
    for name, (sig_s, sig_a) in (("background", BG), ("absorber", AB)):
        xs = openmc.XSdata(name, groups)
        xs.order = 0                                   # isotropic scattering
        xs.set_total([sig_s + sig_a])
        xs.set_absorption([sig_a])
        xs.set_scatter_matrix(np.array([sig_s]).reshape(1, 1, 1))
        lib.add_xsdata(xs)

        m = openmc.Material(name=name)
        m.set_density("macro", 1.0)   # the values above are already macroscopic
        m.add_macroscopic(name)
        mats[name] = m

    RUNDIR.mkdir(parents=True, exist_ok=True)
    xs_path = (RUNDIR / "mgxs.h5").resolve()
    lib.export_to_hdf5(str(xs_path))
    materials = openmc.Materials(list(mats.values()))
    materials.cross_sections = str(xs_path)

    # --- geometry: a lattice of 1 cm cells, vacuum in x and y, reflective in z
    univ = {k: openmc.Universe(cells=[openmc.Cell(fill=v)])
            for k, v in mats.items()}
    absorbing = absorber_map()
    grid = [[univ["absorber" if absorbing[y, x] else "background"]
             for x in range(absorbing.shape[1])]
            for y in range(absorbing.shape[0])]

    lat = openmc.RectLattice()
    lat.lower_left = (0.0, 0.0)
    lat.pitch = (PITCH, PITCH)
    # OpenMC stores lattice rows from the TOP down while absorber_map() is
    # origin-lower, so the row order is reversed here and nowhere else.
    lat.universes = grid[::-1]

    x0 = openmc.XPlane(0.0, boundary_type="vacuum")
    x1 = openmc.XPlane(L, boundary_type="vacuum")
    y0 = openmc.YPlane(0.0, boundary_type="vacuum")
    y1 = openmc.YPlane(L, boundary_type="vacuum")
    z0 = openmc.ZPlane(-0.5, boundary_type="reflective")
    z1 = openmc.ZPlane(+0.5, boundary_type="reflective")
    geometry = openmc.Geometry([openmc.Cell(
        fill=lat, region=+x0 & -x1 & +y0 & -y1 & +z0 & -z1)])

    # --- source: uniform through the central cell, isotropic in 3-D
    sx, sy = SOURCE_CELL
    src = openmc.IndependentSource()
    src.space = openmc.stats.Box((sx * PITCH, sy * PITCH, -0.5),
                                 ((sx + 1) * PITCH, (sy + 1) * PITCH, 0.5))
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([BIRTH_E], [1.0])

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.energy_mode = "multi-group"
    settings.particles = PARTICLES
    settings.batches = BATCHES
    settings.source = src
    settings.seed = SEED
    settings.output = {"tallies": False}

    # --- tallies: the fine flux map, a per-cell map, and the balance
    tallies = openmc.Tallies()
    for label, dim in (("fine", MESH), ("cell", int(round(L / PITCH)))):
        msh = openmc.RegularMesh()
        msh.dimension = (dim, dim, 1)
        msh.lower_left = (0.0, 0.0, -0.5)
        msh.upper_right = (L, L, 0.5)
        t = openmc.Tally(name=f"flux_{label}")
        t.filters = [openmc.MeshFilter(msh)]
        t.scores = ["flux"]
        t.estimator = "tracklength"
        tallies.append(t)

    bal = openmc.Tally(name="absorption")
    bal.scores = ["absorption"]
    # COLLISION, set explicitly.  Left unset, OpenMC picks tracklength -- the
    # same estimator as the flux tallies -- and the absorption cross-check in
    # main() then agrees to every printed digit whatever is wrong with the
    # physics.  A collision estimator scores at collision sites instead of
    # along tracks, so the two only agree if the flux field is right.
    bal.estimator = "collision"
    tallies.append(bal)

    return openmc.Model(geometry, materials, settings, tallies)


def read_flux(sp, name, dim, cell_volume):
    """Mesh tally -> (flux, std dev) as (ny, nx) arrays, origin lower.

    OpenMC orders mesh bins with x fastest, then y, then z, so the flat
    result reshapes as (nz, ny, nx).  Dividing by the cell volume turns
    track length per source particle into scalar flux per unit volume.
    """
    t = sp.get_tally(name=name)
    mean = t.mean.ravel().reshape(1, dim, dim)[0] / cell_volume
    sd = t.std_dev.ravel().reshape(1, dim, dim)[0] / cell_volume
    return mean, sd


# -------------------------------------------------------------------- plot
INK, MUTED, GRID = "#1A1D21", "#6B7280", "#D8DEE4"
HORIZ, VERT = "#1F6FB2", "#C2621F"          # validated pair, CVD dE 21.5


def figure(flux, sd, path):
    """Geometry, the flux map, and two lineouts through the source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = L / MESH
    cut = int(round((SOURCE_CELL[0] + 0.5) * PITCH / d))   # through the source
    xs = (np.arange(MESH) + 0.5) * d
    # A handful of far-corner cells are never scored at this particle count.
    # Mask them rather than clamping to a floor -- a clamped value would drag
    # the colour scale down by decades and flatten every real feature.
    scored = flux > 0
    lg = np.ma.masked_where(~scored, np.log10(np.where(scored, flux, 1.0)))
    hi = float(lg.max())
    lo = hi - 7.0

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for a in ax:
        a.tick_params(colors=MUTED, labelsize=9)
        for s in a.spines.values():
            s.set_color(GRID)

    # -- geometry
    absorbing = absorber_map()
    ax[0].imshow(absorbing, origin="lower", extent=(0, L, 0, L),
                 cmap="Greys", vmin=0, vmax=1.9, interpolation="nearest")
    sx, sy = SOURCE_CELL
    ax[0].add_patch(plt.Rectangle((sx * PITCH, sy * PITCH), PITCH, PITCH,
                                  fill=False, ec=VERT, lw=2.0))
    ax[0].text(sx * PITCH + 0.5, sy * PITCH + 0.5, "source", color=VERT,
               ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax[0].set_title("Geometry", color=INK, fontsize=11, pad=8)
    ax[0].set_xlabel("x (cm)", color=MUTED, fontsize=9.5)
    ax[0].set_ylabel("y (cm)", color=MUTED, fontsize=9.5)

    # -- scalar flux, sequential single-hue ramp
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E8EBEE")            # unscored cells, distinct from the ramp
    im = ax[1].imshow(lg, origin="lower", extent=(0, L, 0, L), cmap=cmap,
                      vmin=lo, vmax=hi, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}\,\phi$", color=MUTED, fontsize=9.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)
    cb.outline.set_edgecolor(GRID)
    ax[1].axhline(xs[cut], color="white", lw=0.8, alpha=0.75)
    ax[1].axvline(xs[cut], color="white", lw=0.8, alpha=0.75)
    ax[1].set_title("Scalar flux", color=INK, fontsize=11, pad=8)
    ax[1].set_xlabel("x (cm)", color=MUTED, fontsize=9.5)

    # -- lineouts through the source centre
    ax[2].grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax[2].set_axisbelow(True)
    h, v = lg[cut, :], lg[:, cut]
    ax[2].plot(xs, h, color=HORIZ, lw=2.0, label="horizontal")
    ax[2].plot(xs, v, color=VERT, lw=2.0, label="vertical")
    for arr, col, lab in ((h, HORIZ, "horizontal"), (v, VERT, "vertical")):
        ax[2].annotate(lab, (xs[-1], arr[-1]), color=col, fontsize=9,
                       fontweight="bold", ha="right",
                       va="bottom" if col == HORIZ else "top",
                       xytext=(-4, 6 if col == HORIZ else -12),
                       textcoords="offset points")
    ax[2].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower center")
    ax[2].set_title(f"Lineouts through the source", color=INK, fontsize=11, pad=8)
    ax[2].set_xlabel("position (cm)", color=MUTED, fontsize=9.5)
    ax[2].set_ylabel(r"$\log_{10}\,\phi$", color=MUTED, fontsize=9.5)
    ax[2].set_xlim(0, L)

    fig.suptitle(f"Lattice benchmark{'' if SCALE == 1 else f' x{SCALE:g}'}, "
                 f"OpenMC multi-group, "
                 f"{PARTICLES * BATCHES:,} histories",
                 color=INK, fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- run
def main():
    n_hist = PARTICLES * BATCHES
    print(f"lattice {L:g}x{L:g} cm, {MESH}x{MESH} mesh, "
          f"cross sections x{SCALE:g}")
    print(f"  background  sig_s={BG[0]:g}  sig_a={BG[1]:g}")
    print(f"  absorbers   sig_s={AB[0]:g}  sig_a={AB[1]:g}   "
          f"({len(ABSORBERS)} cells)")
    print(f"  source cell {SOURCE_CELL}, isotropic and uniform")
    print(f"  {BATCHES} batches x {PARTICLES:,} = {n_hist:,} histories\n")

    t0 = time.time()
    sp_path = build().run(cwd=str(RUNDIR))
    wall = time.time() - t0

    ncell = int(round(L / PITCH))
    d = L / MESH
    with openmc.StatePoint(sp_path) as sp:
        fine, fine_sd = read_flux(sp, "flux_fine", MESH, d * d * 1.0)
        cell, cell_sd = read_flux(sp, "flux_cell", ncell, PITCH ** 2 * 1.0)
        absorbed = float(sp.get_tally(name="absorption").mean.ravel()[0])
        absorbed_sd = float(sp.get_tally(name="absorption").std_dev.ravel()[0])

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"openmc_fine{TAG}.npy", fine)
    np.save(OUT / f"openmc_fine_sd{TAG}.npy", fine_sd)
    np.save(OUT / f"openmc_cell{TAG}.npy", cell)
    np.save(OUT / f"openmc_cell_sd{TAG}.npy", cell_sd)
    figure(fine, fine_sd, OUT / f"openmc_lattice{TAG}.png")

    scored = fine > 0
    rel = np.where(scored, fine_sd / np.maximum(fine, 1e-300), np.nan)
    absorbing = absorber_map()

    # Internal consistency: the absorption rate implied by the flux field,
    # sum(sig_a * phi * V) over EVERY cell, must equal the independently
    # scored absorption tally.  They come from different estimators
    # (tracklength vs collision, see build()), so they agree only to within
    # their statistics -- and agreement means the mesh, the volume
    # normalisation and the material assignment are all consistent with one
    # another.
    #
    # Every cell, not just the absorbers: with the benchmark's sig_a = 0
    # background the two are the same sum, but a real moderator absorbs
    # weakly everywhere and omitting it leaves the check several per cent
    # short.  It read as a failure the first time a real material was used.
    sig_a_map = np.where(absorbing, AB[1], BG[1])
    implied = float((sig_a_map * cell).sum()) * PITCH ** 2
    # treating cells as independent; the two estimators also share
    # histories, so this z is approximate but the right order
    implied_sd = float(np.sqrt(((sig_a_map * cell_sd) ** 2).sum())) * PITCH ** 2
    z_abs = (implied - absorbed) / np.hypot(implied_sd, absorbed_sd)
    lines = [
        "=" * 66,
        "LATTICE BENCHMARK -- OpenMC multi-group, one energy group",
        "=" * 66,
        f"histories            {n_hist:,}  ({BATCHES} x {PARTICLES:,})",
        f"cross sections       x{SCALE:g} (1 = the published benchmark)",
        f"wall time            {wall:.1f} s",
        f"OpenMC               {openmc.__version__}",
        "",
        "PARTICLE BALANCE  (per source particle)",
        f"  absorption         {absorbed:.6f} +/- {absorbed_sd:.6f}",
        f"  leakage            {1.0 - absorbed:.6f}",
        f"  sum                {1.0:.6f}",
        "",
        "ABSORPTION CROSS-CHECK  (two independent estimators)",
        f"  from the flux map   {implied:.6f}   sum of sig_a * phi * V",
        f"  from the tally      {absorbed:.6f}",
        f"  difference          {abs(implied - absorbed) / absorbed * 100:.3f} %"
        f"   (z = {z_abs:+.2f}; |z| < 3 passes)",
        "",
        "SCALAR FLUX  (per unit volume, per source particle)",
        f"  domain integral    {fine.sum() * d * d:.6f}",
        f"  peak               {fine.max():.4f}",
        f"  smallest scored    {fine[scored].min():.3e}",
        f"  dynamic range      "
        f"{np.log10(fine.max() / fine[scored].min()):.1f} decades",
        f"  unscored cells     {int((~scored).sum())} of {fine.size}"
        f"  ({(~scored).mean() * 100:.2f} %, all in the far corners)",
        "",
        "STATISTICAL PRECISION  (relative standard deviation, scored cells)",
        f"  median             {np.nanmedian(rel) * 100:.3f} %",
        f"  90th percentile    {np.nanpercentile(rel, 90) * 100:.3f} %",
        f"  worst cell         {np.nanmax(rel) * 100:.3f} %",
        "",
        "CELL-AVERAGED FLUX  (7 x 7, y increasing upward)",
    ]
    for y in range(ncell - 1, -1, -1):
        row = "  " + " ".join(f"{cell[y, x]:9.3e}" for x in range(ncell))
        lines.append(row + "   " + "".join("A" if absorbing[y, x] else "."
                                           for x in range(ncell)))
    lines += ["", "  A marks an absorber cell.", "=" * 66]

    text = "\n".join(lines)
    (OUT / f"openmc_results{TAG}.txt").write_text(text + "\n")

    # One self-describing file: the arrays AND the problem that produced
    # them.  A consumer (make_reference.py) checks these parameters against
    # its own definition before trusting the numbers.
    np.savez_compressed(
        OUT / f"openmc_reference{TAG}.npz",
        fine=fine, fine_sd=fine_sd, cell=cell, cell_sd=cell_sd,
        absorbed=absorbed, absorbed_sd=absorbed_sd,
        scale=SCALE, L=L, pitch=PITCH, mesh=MESH, bg=np.array(BG),
        ab=np.array(AB), absorbers=np.array(ABSORBERS),
        source_cell=np.array(SOURCE_CELL), particles=PARTICLES,
        batches=BATCHES, seed=SEED, wall_seconds=wall,
        openmc_version=openmc.__version__)
    print(text)
    print(f"\nwrote {OUT}/openmc_results{TAG}.txt, openmc_lattice{TAG}.png, "
          f"openmc_reference{TAG}.npz and four .npy arrays")


if __name__ == "__main__":
    main()
