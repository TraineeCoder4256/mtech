"""The robustness suite (SRS V1-ROB-4): randomly built models against OpenMC.

Used by: run.py (`python run.py robust`).

Each model is drawn from a seeded random generator, so model k is the same
on every run and a failure can be rebuilt (`build(k)`).  Each one must
PASS against OpenMC with the validation rule (checks/validate.py: no tally
chi2 with p < 0.001, no single value with |z| > 3.29, no lost particles),
for plain Monte Carlo and for exact balls, or be refused cleanly while it
is read (an `Unsupported` naming the feature).  Anything else (a crash, a
lost particle, a wrong answer) is a failure.

What a model is made of, drawn within what gmc3d accepts:

    world       a box, each pair of faces vacuum, reflective, white or
                periodic (at least one pair vacuum)
    objects     1 to 4, each in its own slot along x so none overlap:
                sphere, shell (sphere in sphere), z-cylinder, box, cone
                segment, any-axis cylinder in a sphere, ZTorus, or a 2x2 /
                3x3 RectLattice of pins; some objects are void
    materials   2 to 4, sigma_s 0.05-2 and sigma_a 0.005-0.5 per cm
    source      point, box or spherical shell; isotropic, monodirectional
                or a polar-azimuthal cone; sometimes a cell constraint
    tallies     total; cell; material; a regular mesh; then at random a
                cylindrical or spherical mesh, currents on the world's
                faces, a mesh-surface tally, a cell x energy tally
    refusals    about one model in seven also asks for something gmc3d
                does not do (an albedo, a collision estimator, a fission
                score, a horn torus): it must be refused, naming it

Left out on purpose: the three places OpenMC 0.16 is itself wrong
(runs/2026-10-30/v1.5/openmc_bugs.log): spherical-mesh theta edges at
pi/2, currents on surfaces inside moved universes, and rotational
periodic walls; and the 'scatter' score in a model with a void cell,
which crashes OpenMC 0.16 in multigroup mode (models 15 and 29 of the
first v1.6 run).  Models with them would fail against OpenMC for OpenMC's
error, not ours; unit tests cover them instead.

Histories: 200,000 per code and model (OpenMC in 200 batches); many models
times many tallies at p = 0.001 still expect about one false alarm per
run, so a FAIL is rerun once at 5x histories before it counts.
"""

import time
import traceback

import numpy as np
import openmc
import openmc.stats as st

from ball.oracle import Oracle
from core import openmc_import, transport
from core.openmc_import import Unsupported
from problems import common
from . import compare, openmc_ref, validate

N_MODELS = 30
PARTICLES, BATCHES, REF_BATCHES = 20_000, 10, 200


def _mat_xs(rng, n):
    return {f"m{j}": (float(rng.uniform(0.05, 2.0)),
                      float(rng.uniform(0.005, 0.5))) for j in range(n)}


def _object(rng, kind, x0, half, mats, void_ok=True):
    """(cells, region it occupies) for one object centred at (x0, 0, 0)
    inside a slot of half-width `half` (in x; y and z have more room)."""
    pick = lambda: mats[rng.integers(len(mats))]
    fill = None if (void_ok and rng.random() < 0.15) else pick()
    r = half * rng.uniform(0.5, 0.9)
    if kind == "sphere":
        s = openmc.Sphere(x0=x0, r=r)
        return [openmc.Cell(fill=fill, region=-s)], -s
    if kind == "shell":
        so, si = openmc.Sphere(x0=x0, r=r), openmc.Sphere(x0=x0, r=0.5 * r)
        return [openmc.Cell(fill=pick(), region=-si),
                openmc.Cell(fill=fill, region=+si & -so)], -so
    if kind == "zcyl":
        c = openmc.ZCylinder(x0=x0, r=r)
        lo, hi = openmc.ZPlane(-1.5 * r), openmc.ZPlane(1.5 * r)
        reg = -c & +lo & -hi
        return [openmc.Cell(fill=fill, region=reg)], reg
    if kind == "box":
        reg = openmc.model.RectangularParallelepiped(
            x0 - r, x0 + r, -r, r, -0.7 * r, 0.7 * r)
        return [openmc.Cell(fill=fill, region=-reg)], -reg
    if kind == "cone":
        k = openmc.ZCone(x0=x0, z0=r, r2=0.25)
        lo = openmc.ZPlane(-r)
        reg = -k & +lo & -openmc.ZPlane(r - 1e-3)
        return [openmc.Cell(fill=fill, region=reg)], reg
    if kind == "tiltcyl":
        d = rng.normal(size=3)
        c = openmc.Cylinder(x0=x0, r=0.4 * r, dx=d[0], dy=d[1], dz=d[2])
        s = openmc.Sphere(x0=x0, r=r)
        return [openmc.Cell(fill=pick(), region=-c & -s),
                openmc.Cell(fill=fill, region=+c & -s)], -s
    if kind == "torus":
        t = openmc.ZTorus(x0=x0, a=0.6 * r, b=0.3 * r, c=0.3 * r)
        s = openmc.Sphere(x0=x0, r=r)
        return [openmc.Cell(fill=pick(), region=-t),
                openmc.Cell(fill=fill, region=+t & -s)], -s
    if kind == "lattice":
        n = int(rng.integers(2, 4))
        pitch = 2 * r / n
        pin = openmc.ZCylinder(r=0.35 * pitch)
        a, b = pick(), pick()
        u = openmc.Universe(cells=[openmc.Cell(fill=a, region=-pin),
                                   openmc.Cell(fill=b, region=+pin)])
        lat = openmc.RectLattice()
        lat.lower_left, lat.pitch = (x0 - r, -r), (pitch, pitch)
        lat.universes = [[u] * n for _ in range(n)]
        reg = -openmc.model.RectangularParallelepiped(
            x0 - r, x0 + r, -r, r, -1.2 * r, 1.2 * r)
        return [openmc.Cell(fill=lat, region=reg)], reg
    raise ValueError(kind)


KINDS = ("sphere", "shell", "zcyl", "box", "cone", "tiltcyl", "torus",
         "lattice")


def build(k):
    """Model number k (a fresh openmc.Model) and a one-line description."""
    rng = np.random.default_rng([2026, 10, 30, k])
    nmat = int(rng.integers(2, 5))
    materials, mm = common.one_group_materials(f"robust{k}",
                                               _mat_xs(rng, nmat))
    mats = list(mm.values())
    nobj = int(rng.integers(1, 5))
    half = float(rng.uniform(3.0, 6.0))          # slot half-width
    Lx = nobj * half + 1.0
    Ly = Lz = half * 2.0 + 1.0
    bcs = []
    for axis in range(3):
        b = rng.choice(["vacuum", "reflective", "white", "periodic"],
                       p=[0.55, 0.2, 0.1, 0.15])
        bcs.append(str(b))
    if "vacuum" not in bcs:
        bcs[int(rng.integers(3))] = "vacuum"
    lim = (Lx, Ly, Lz)
    faces = []
    for axis, cls in enumerate((openmc.XPlane, openmc.YPlane,
                                openmc.ZPlane)):
        lo, hi = cls(-lim[axis]), cls(lim[axis])
        lo.boundary_type = hi.boundary_type = bcs[axis]
        if bcs[axis] == "periodic":
            lo.periodic_surface = hi
        faces += [lo, hi]
    world = (+faces[0] & -faces[1] & +faces[2] & -faces[3] & +faces[4] &
             -faces[5])
    kinds = [str(rng.choice(KINDS)) for _ in range(nobj)]
    cells, holes = [], []
    for j, kind in enumerate(kinds):
        x0 = -Lx + 1.0 + half * (2 * j + 1)      # slot j's centre
        cs, reg = _object(rng, kind, x0, half, mats)
        cells += cs
        holes.append(reg)
    bg = openmc.Cell(fill=mats[int(rng.integers(nmat))], region=world)
    for h in holes:
        bg.region = bg.region & ~h
    cells.append(bg)
    geometry = openmc.Geometry(cells)

    # source
    sk = rng.choice(["point", "box", "shell"])
    if sk == "point":
        space = st.Point(tuple(rng.uniform(-0.8, 0.8, 3) * lim))
    elif sk == "box":
        a = rng.uniform(-0.9, 0.0, 3) * lim
        space = st.Box(tuple(a), tuple(a + rng.uniform(0.1, 0.9, 3) *
                                       np.asarray(lim)))
    else:
        space = st.SphericalIndependent(
            st.PowerLaw(0.0, 0.8 * min(lim), 2), st.Uniform(-1.0, 1.0),
            st.Uniform(0.0, 2 * np.pi),
            origin=tuple(rng.uniform(-0.3, 0.3, 3) * lim))
    ak = rng.choice(["iso", "mono", "cone"], p=[0.6, 0.2, 0.2])
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    angle = (st.Isotropic() if ak == "iso" else
             st.Monodirectional(tuple(d)) if ak == "mono" else
             st.PolarAzimuthal(st.Uniform(0.5, 1.0),
                               st.Uniform(0.0, 2 * np.pi),
                               reference_uvw=tuple(d)))
    src = openmc.IndependentSource(space=space, angle=angle,
                                   energy=st.Discrete([1.0e6], [1.0]))
    if rng.random() < 0.2 and len(cells) > 1:
        src.constraints = {"domains": [cells[int(rng.integers(
            len(cells) - 1))], bg]}
    settings = common.settings(src)

    # tallies (no 'scatter' score with a void cell: OpenMC 0.16 crashes on
    # a multigroup track-length scatter tally in void, measured v1.1.3)
    void = any(c.fill is None for c in cells)
    tallies = [common.flux_tally("total", scores=(
                   ("flux", "absorption", "total") if void else
                   ("flux", "absorption", "scatter", "total"))),
               common.flux_tally("cells", openmc.CellFilter(cells)),
               common.flux_tally("mats", openmc.MaterialFilter(mats)),
               common.flux_tally("mesh", common.mesh_filter(
                   (-Lx, -Ly, -Lz), (Lx, Ly, Lz), (6, 4, 4)))]
    if rng.random() < 0.4:
        tallies.append(common.flux_tally("cyl", openmc.MeshFilter(
            openmc.CylindricalMesh(
                r_grid=np.linspace(0, min(Ly, Lz), 4),
                phi_grid=np.linspace(0, 2 * np.pi, 4),
                z_grid=np.linspace(-Lz, Lz, 3),
                origin=tuple(rng.uniform(-1, 1, 3))))))
    if rng.random() < 0.3:
        tallies.append(common.flux_tally("sph", openmc.MeshFilter(
            openmc.SphericalMesh(
                r_grid=np.linspace(0, min(lim), 4),
                theta_grid=[0.0, 1.0, 2.2, np.pi],
                phi_grid=np.linspace(0, 2 * np.pi, 4),
                origin=tuple(rng.uniform(-1, 1, 3))))))
    if rng.random() < 0.5:
        t = openmc.Tally(name="faces")
        t.filters = [openmc.SurfaceFilter(faces)]
        t.scores = ["current"]
        tallies.append(t)
    if rng.random() < 0.3:
        m = openmc.RegularMesh()
        m.lower_left, m.upper_right = (-Lx, -Ly, -Lz), (Lx, Ly, Lz)
        m.dimension = (3, 2, 2)
        t = openmc.Tally(name="meshsurf")
        t.filters = [openmc.MeshSurfaceFilter(m)]
        t.scores = ["current"]
        tallies.append(t)
    if rng.random() < 0.3:
        tallies.append(common.flux_tally("cell_e", None))
        tallies[-1].filters = [openmc.CellFilter(cells[:2]),
                               openmc.EnergyFilter([0.0, 20.0e6])]
    # about one model in seven also asks for something gmc3d does not do,
    # to check that it is refused cleanly while the model is read
    bad = None
    if rng.random() < 0.15:
        bad = str(rng.choice(["albedo", "collision", "fission", "horn"]))
        if bad == "albedo":
            v = next(f for f in faces if f.boundary_type == "vacuum")
            v.albedo = 0.5
        elif bad == "collision":
            tallies[0].estimator = "collision"
        elif bad == "fission":
            tallies[1].scores = ["flux", "fission"]
        else:
            horn = openmc.ZTorus(x0=-Lx + 1.0 + half, a=0.2, b=0.4, c=0.4)
            cells[0].region = cells[0].region & +horn
    desc = (f"{nobj} object(s) {kinds}, {nmat} materials, faces x/y/z "
            f"{bcs}, source {sk}/{ak}"
            + (" + constraint" if src.constraints else "")
            + f", tallies {[t.name for t in tallies]}"
            + (f", UNSUPPORTED {bad}" if bad else ""))
    return openmc.Model(geometry, materials, settings,
                        openmc.Tallies(tallies)), desc


def run_one(k, workdir, scale=1, log=print):
    """Model k against OpenMC.  Returns a dict with 'status' PASS, FAIL,
    REFUSED or ERROR and the details."""
    t0 = time.perf_counter()
    out = {"model": k}
    try:
        model, desc = build(k)
        out["desc"] = desc
        try:
            pr = openmc_import.load(model)
        except Unsupported as e:
            out.update(status="REFUSED", reason=str(e))
            return out
        n = PARTICLES * BATCHES * scale
        ref = openmc_ref.run(model, f"{workdir}/robust_{k}_x{scale}",
                             n // REF_BATCHES, REF_BATCHES,
                             seed=1000 + k)
        res = {}
        for label, be in (("mc", None), ("balls", Oracle())):
            r = transport.run(pr, PARTICLES * scale, BATCHES, seed=k + 1,
                              backend=be, r_star=1.0)
            cmp = compare.compare(r, ref)
            ok, bad = validate.verdict(cmp)
            if r.counters["lost"]:
                ok, bad = False, bad + [f"{r.counters['lost']} lost"]
            res[label] = {"pass": ok, "bad": bad, "cmp": cmp,
                          "lost": r.counters["lost"],
                          "balls": r.counters["balls"]}
        out["runs"] = res
        out["status"] = "PASS" if all(v["pass"] for v in res.values()) \
            else "FAIL"
    except Exception as e:                        # noqa: BLE001
        out.update(status="ERROR", reason=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc())
    finally:
        out["seconds"] = time.perf_counter() - t0
    return out


def suite(workdir, n_models=N_MODELS, log=print):
    rows = []
    for k in range(n_models):
        row = run_one(k, workdir)
        if row["status"] == "FAIL":
            log(f"  model {k}: FAIL at 1x "
                f"({'; '.join(sum((v['bad'] for v in row['runs'].values()), []))})"
                f", rerunning at 5x histories")
            first = row
            row = run_one(k, workdir, scale=5)
            row["first_try"] = {"status": first["status"],
                                "bad": {l: v["bad"] for l, v in
                                        first["runs"].items()}}
        rows.append(row)
        extra = row.get("reason") or "; ".join(
            sum((v["bad"] for v in row.get("runs", {}).values()), []))
        log(f"  model {k:2d}  {row['status']:8} {row['seconds']:5.0f} s  "
            f"{row.get('desc', '')}" + (f"\n             {extra}"
                                        if extra else ""))
    counts = {s: sum(r["status"] == s for r in rows)
              for s in ("PASS", "REFUSED", "FAIL", "ERROR")}
    log(f"\n== robustness: {counts['PASS']} PASS, {counts['REFUSED']} "
        f"refused cleanly, {counts['FAIL']} FAIL, {counts['ERROR']} ERROR "
        f"of {len(rows)} models")
    return {"models": rows, "counts": counts,
            "ok": counts["PASS"] + counts["REFUSED"], "total": len(rows)}
