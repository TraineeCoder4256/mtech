#!/usr/bin/env python3
"""Independent check of the lattice with OpenMC.  Run: python openmc_lattice.py

Why this exists
---------------
mc.py is our own solver, so "mc.py agrees with mc.py" proves nothing about
the physics.  validate.py already checks it against three exact analytic
identities, which is strong but local.  This file adds the other kind of
evidence: a production transport code, written by other people, solving the
SAME problem and producing a flux field we can difference cell by cell.

Getting one-to-one comparability needs four things to line up exactly, and
each one is a place where a careless setup would silently produce a
different problem:

  1. MONOENERGETIC.  OpenMC is continuous-energy by default.  We run it in
     MULTI-GROUP mode with a single group, so the cross sections are the
     constants sig_s and sig_a and nothing is interpolated from a library.

  2. 2-D XY WITH 3-D DIRECTIONS.  Our particles have 2-D positions but
     3-D direction cosines (see the Omega_z discussion in mc.py).  OpenMC
     is 3-D, so we bound z with two REFLECTIVE planes.  For a problem that
     is invariant in z this is exact, not an approximation: it reproduces
     an infinite extrusion, which is precisely our geometry.

  3. NORMALISATION.  mc.py returns tally / (dx * dy * n) -- track length
     per unit volume per source particle, with unit depth in z.  OpenMC's
     flux tally is already per source particle, so we divide by the mesh
     cell volume dx * dy * dz with dz = 1.  The two are then the same
     quantity in the same units.

  4. MESH.  Two meshes are tallied.  The 112 x 112 one matches mc.py's
     fine grid.  The 7 x 7 one matches the macro-cell grid the generative
     sampler works on -- solve.py can only produce a cell-averaged flux,
     so comparing it against the fine field would be meaningless.

Cross sections and layout are read from mc.py itself rather than retyped,
so the two solvers cannot drift apart.

Requires OpenMC (conda install -c conda-forge openmc).  Nothing else in
the project depends on this file; it is a check, not part of the pipeline.
"""
import pathlib
import sys
import time

import numpy as np

import mc

# ---- settings ----------------------------------------------------------
PARTICLES = 200_000        # per batch
BATCHES = 20               # total histories = PARTICLES * BATCHES
MC_RUNS = 5                # our-MC repeats, for a per-cell error bar
SEED = 20250901
RUNDIR = pathlib.Path("results/openmc")
OUT = pathlib.Path("results")

GROUP_TOP = 20.0e6         # single group [0, 20 MeV); the value is arbitrary
BIRTH_E = 1.0e6            # any energy inside the group


# ------------------------------------------------------------------ model
def build(problem):
    """Assemble the OpenMC model for `problem` (from mc.lattice())."""
    import openmc

    sig_s, sig_a = _macro_cross_sections(problem)
    kinds = sorted({(round(s, 12), round(a, 12))
                    for s, a in zip(sig_s.ravel(), sig_a.ravel())})
    names = {k: f"m{i}" for i, k in enumerate(kinds)}

    # --- one-group macroscopic cross sections, P0 (isotropic) scattering
    groups = openmc.mgxs.EnergyGroups([0.0, GROUP_TOP])
    xsdatas, materials = [], {}
    for (s, a), name in names.items():
        xs = openmc.XSdata(name, groups)
        xs.order = 0                                  # isotropic scattering
        xs.set_total([s + a])
        xs.set_absorption([a])
        xs.set_scatter_matrix(np.array([s]).reshape(1, 1, 1))
        xsdatas.append(xs)

        m = openmc.Material(name=name)
        m.set_density("macro", 1.0)   # values above are already macroscopic
        m.add_macroscopic(name)
        materials[name] = m

    RUNDIR.mkdir(parents=True, exist_ok=True)
    xs_path = (RUNDIR / "mgxs.h5").resolve()
    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas(xsdatas)
    lib.export_to_hdf5(str(xs_path))

    mats = openmc.Materials(list(materials.values()))
    mats.cross_sections = str(xs_path)

    # --- geometry: a 7 x 7 lattice of 1 cm cells
    univ = {}
    for name, m in materials.items():
        univ[name] = openmc.Universe(cells=[openmc.Cell(fill=m)])

    L, pitch = problem["L"], problem["pitch"]
    ncell = int(round(L / pitch))
    grid = [[univ[names[(round(sig_s[y, x], 12), round(sig_a[y, x], 12))]]
             for x in range(ncell)]
            for y in range(ncell)]
    lat = openmc.RectLattice()
    lat.lower_left = (0.0, 0.0)
    lat.pitch = (pitch, pitch)
    # OpenMC stores lattice rows from the TOP down; our arrays are
    # origin-lower, so the row order is reversed here and nowhere else.
    lat.universes = grid[::-1]

    x0 = openmc.XPlane(0.0, boundary_type="vacuum")
    x1 = openmc.XPlane(L, boundary_type="vacuum")
    y0 = openmc.YPlane(0.0, boundary_type="vacuum")
    y1 = openmc.YPlane(L, boundary_type="vacuum")
    # reflective in z == infinite extrusion, because nothing varies with z
    z0 = openmc.ZPlane(-0.5, boundary_type="reflective")
    z1 = openmc.ZPlane(+0.5, boundary_type="reflective")
    root = openmc.Universe(cells=[openmc.Cell(
        fill=lat, region=+x0 & -x1 & +y0 & -y1 & +z0 & -z1)])
    geom = openmc.Geometry(root)

    # --- source: uniform in the central cell, isotropic in 3-D
    sx0, sy0, sx1, sy1 = problem["source"]
    try:
        src = openmc.IndependentSource()
    except AttributeError:                       # OpenMC < 0.14
        src = openmc.Source()
    src.space = openmc.stats.Box((sx0, sy0, -0.5), (sx1, sy1, 0.5))
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

    # --- tallies: the fine mesh for mc.py, the macro mesh for the sampler
    tallies = openmc.Tallies()
    meshes = {}
    for label, dim in (("fine", problem["n"]), ("coarse", ncell)):
        msh = openmc.RegularMesh()
        msh.dimension = (dim, dim, 1)
        msh.lower_left = (0.0, 0.0, -0.5)
        msh.upper_right = (L, L, 0.5)
        t = openmc.Tally(name=f"flux_{label}")
        t.filters = [openmc.MeshFilter(msh)]
        t.scores = ["flux"]
        t.estimator = "tracklength"
        tallies.append(t)
        meshes[label] = msh

    # particle balance: absorption + leakage must equal 1 per source particle
    bal = openmc.Tally(name="absorption")
    bal.scores = ["absorption"]
    tallies.append(bal)

    return openmc.Model(geom, mats, settings, tallies), meshes


def _macro_cross_sections(problem):
    """The per-1cm-cell cross sections, checked to be uniform within a cell.

    mc.py stores sig_s and sig_a on the fine 112 x 112 grid.  The OpenMC
    geometry is built from 1 cm material regions, so this collapses the
    fine arrays and asserts nothing was lost -- if a block boundary ever
    stopped landing on a cell edge, this is where it would be caught.
    """
    k = problem["per_cm"]
    out = []
    for key in ("sig_s", "sig_a"):
        a = problem[key]
        ny, nx = a.shape[0] // k, a.shape[1] // k
        blocks = a[:ny * k, :nx * k].reshape(ny, k, nx, k)
        assert blocks.std(axis=(1, 3)).max() < 1e-12, f"{key} varies inside a cell"
        out.append(blocks[:, 0, :, 0].copy())
    return out


# ------------------------------------------------------------- extraction
def read_flux(sp, name, dim, cell_volume):
    """Mesh tally -> (flux, std_dev) as (ny, nx) arrays, origin lower.

    OpenMC orders mesh bins with x fastest, then y, then z, so the flat
    result reshapes as (nz, ny, nx).  Dividing by the cell volume converts
    OpenMC's track length per source particle into the same flux per unit
    volume per source particle that mc.solve() returns.
    """
    t = sp.get_tally(name=name)
    mean = t.mean.ravel().reshape(1, dim, dim)[0] / cell_volume
    sd = t.std_dev.ravel().reshape(1, dim, dim)[0] / cell_volume
    return mean, sd


# ------------------------------------------------------------- comparison
def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare(label, ours, theirs, sd_ours, sd_theirs):
    """Difference two flux fields and say whether it is more than noise."""
    m = theirs > 0
    rel = np.abs(ours[m] - theirs[m]) / theirs[m] * 100.0
    sd = np.sqrt(sd_ours[m] ** 2 + sd_theirs[m] ** 2)
    z = np.where(sd > 0, (ours[m] - theirs[m]) / np.maximum(sd, 1e-300), 0.0)
    print(f"\n  {label}")
    print(f"    relative L2                 {rel_l2(ours, theirs)*100:8.3f} %")
    print(f"    per-cell relative error     median {np.median(rel):6.2f} %"
          f"   90th {np.percentile(rel,90):6.2f} %   max {rel.max():7.2f} %")
    print(f"    |z| > 2 in                  {np.mean(np.abs(z)>2)*100:8.2f} %"
          f" of cells   (about 5 % expected if both are correct)")
    print(f"    total flux ratio            {ours.sum()/theirs.sum():8.6f}")
    return {"rel_l2": rel_l2(ours, theirs), "frac_z2": float(np.mean(np.abs(z) > 2))}


def our_mc(problem, n, runs):
    """Solve with mc.py `runs` times; return the stack of fields.

    The repeats are kept rather than reduced here because the coarse-mesh
    error bar has to be computed from the repeats too.  Propagating the
    fine-mesh error into the coarse mesh would need the fine cells to be
    statistically independent, and they are not -- one particle lays track
    length in many neighbouring cells, so their errors are correlated and
    a sqrt(N) reduction would understate the coarse error bar.
    """
    fields, stats = [], None
    for i in range(runs):
        phi, stats = mc.solve(problem, n, seed=1000 + i)
        fields.append(phi)
    return np.stack(fields), stats


def mean_sem(stack):
    """Ensemble mean and standard error of the mean, over independent runs."""
    return stack.mean(0), stack.std(0, ddof=1) / np.sqrt(stack.shape[0])


# -------------------------------------------------------------------- run
def main():
    try:
        import openmc                                           # noqa: F401
    except ImportError:
        sys.exit("OpenMC is not installed.  conda install -c conda-forge openmc")

    problem = mc.lattice()
    n_hist = PARTICLES * BATCHES
    print(f"lattice 7x7 cm, {problem['n']}x{problem['n']} mesh, "
          f"absorbers sig_s=0.5 sig_a=9.5, background sig_s=1.0 sig_a=0.0")
    print(f"OpenMC: {BATCHES} batches x {PARTICLES:,} = {n_hist:,} histories\n")

    model, meshes = build(problem)
    t0 = time.time()
    sp_path = model.run(cwd=str(RUNDIR))
    t_openmc = time.time() - t0
    print(f"\nOpenMC finished in {t_openmc:.1f} s")

    L, ncell = problem["L"], int(round(problem["L"] / problem["pitch"]))
    d = L / problem["n"]
    with openmc.StatePoint(sp_path) as sp:
        fine, fine_sd = read_flux(sp, "flux_fine", problem["n"], d * d * 1.0)
        coarse, coarse_sd = read_flux(sp, "flux_coarse", ncell,
                                      problem["pitch"] ** 2 * 1.0)
        absorbed = float(sp.get_tally(name="absorption").mean.ravel()[0])

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "openmc_fine.npy", fine)
    np.save(OUT / "openmc_fine_sd.npy", fine_sd)
    np.save(OUT / "openmc_coarse.npy", coarse)
    np.save(OUT / "openmc_coarse_sd.npy", coarse_sd)

    print(f"\nPARTICLE BALANCE (OpenMC)")
    print(f"    absorption per source particle   {absorbed:10.6f}")
    print(f"    leakage    per source particle   {1.0-absorbed:10.6f}")
    print(f"    sum                              {1.0:10.6f}  (exact by construction)")

    # ---- our MC on the same problem, same normalisation
    print(f"\nour mc.py: {MC_RUNS} runs x {n_hist:,} histories")
    t0 = time.time()
    stack, stats = our_mc(problem, n_hist, MC_RUNS)
    t_ours = time.time() - t0
    print(f"    finished in {t_ours:.1f} s, "
          f"{stats['scatters_per_particle']:.2f} scatters per particle")

    ours, ours_sd = mean_sem(stack)
    # the coarse statistics come from coarsening each run, not from
    # propagating the fine error bar -- see our_mc()
    ours_c, ours_c_sd = mean_sem(
        np.stack([mc_coarsen(f, problem["per_cm"]) for f in stack]))
    np.save(OUT / "mc_fine.npy", ours)
    np.save(OUT / "mc_coarse.npy", ours_c)

    print("\nOPENMC  vs  OUR MONTE CARLO")
    compare("fine mesh (112 x 112)", ours, fine, ours_sd, fine_sd)
    compare("macro cells (7 x 7)", ours_c, coarse, ours_c_sd, coarse_sd)

    # ---- the sampler, if evaluate.py has left a field behind
    gmc_path = OUT / "gmc_coarse.npy"
    if gmc_path.exists():
        gmc = np.load(gmc_path)
        print("\nOPENMC  vs  GENERATIVE SAMPLER")
        compare("macro cells (7 x 7)", gmc, coarse,
                np.zeros_like(gmc), coarse_sd)
    else:
        print(f"\n{gmc_path} not found -- run evaluate.py and save the GMC"
              f"\nmacro-cell field there to include the sampler in this table.")


def mc_coarsen(phi, per_cm):
    """Fine grid -> macro cells by volume average (what solve.py compares to)."""
    ny, nx = phi.shape[0] // per_cm, phi.shape[1] // per_cm
    return phi[:ny * per_cm, :nx * per_cm].reshape(
        ny, per_cm, nx, per_cm).mean(axis=(1, 3))


if __name__ == "__main__":
    main()
