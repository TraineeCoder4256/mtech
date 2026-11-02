"""v1.2: every kind of source, against OpenMC.

Used by: run.py (`python run.py sourcecheck`).

Each case takes an existing test problem and swaps in one kind of source.
The geometry, materials and tallies are the problem's own, so any
difference from OpenMC comes from where and how the particles are born.
Cases on the cask (three materials, voids, a duct; birth position and
direction both matter for what reaches the detector):

    cartesian     CartesianIndependent: Normal x (sampled from a wider
                  Normal, so every site carries a weight), Uniform y,
                  Tabular z
    cylindrical   CylindricalIndependent, r^1 power law (uniform in area)
    spherical     SphericalIndependent, uniform in a ball, with a
                  PolarAzimuthal angle: mu in [0, 1] about +x (towards the
                  duct), phi only in [0, pi/2] about +y.  An uneven phi
                  is what tests the phi convention.
    pointcloud    PointCloud with uneven strengths.  (A PointCloud bias
                  cannot be checked: OpenMC 0.16 refuses the XML its own
                  Python writes for one.)
    meshspatial   MeshSpatial on a CylindricalMesh, volume-normalised,
                  strength rising with r
    meshsource    MeshSource on a RegularMesh: half the elements
                  isotropic, half beaming along +x
    file          FileSource: 20,000 sites with weights 0.5 and 2
    material      Box around the cavity, liner and some concrete,
                  constrained to the fuel material
    two           two sources of strength 1 and 3 (Point and Box)

and on the nested lattice (domains below the root):

    universe      Box around the rotated universe's lattice elements,
                  constrained to that universe
    deepcell      the same, constrained to the pin cell inside that
                  rotated universe (three levels down)

Every case: OpenMC (400 batches), our plain Monte Carlo and our exact
balls (R* = 1), with the pass rule of checks/validate.py.
"""

import time
from pathlib import Path

import numpy as np
import openmc
import openmc.stats as st

from ball.oracle import Oracle
from core import openmc_import, transport
from problems import PROBLEMS
from . import compare, openmc_ref, validate


def _cask_case(name, workdir):
    model = PROBLEMS["cask"]()
    mats = {m.name: m for m in model.materials}
    if name == "cartesian":
        x = st.Normal(0.0, 6.0)
        x.bias = st.Normal(0.0, 9.0)
        src = openmc.IndependentSource(space=st.CartesianIndependent(
            x, st.Uniform(-14.0, 14.0),
            st.Tabular([-60.0, 0.0, 60.0], [1.0, 3.0, 1.0],
                       interpolation="linear-linear")))
    elif name == "cylindrical":
        src = openmc.IndependentSource(space=st.CylindricalIndependent(
            st.PowerLaw(0.0, 19.0, 1), st.Uniform(0.0, 2 * np.pi),
            st.Uniform(-60.0, 60.0)))
    elif name == "spherical":
        src = openmc.IndependentSource(
            space=st.SphericalIndependent(st.PowerLaw(0.0, 18.0, 2),
                                          st.Uniform(-1.0, 1.0),
                                          st.Uniform(0.0, 2 * np.pi)),
            angle=st.PolarAzimuthal(st.Uniform(0.0, 1.0),
                                    st.Uniform(0.0, 0.5 * np.pi),
                                    reference_uvw=(1.0, 0.0, 0.0),
                                    reference_vwu=(0.0, 1.0, 0.0)))
    elif name == "pointcloud":
        pts = [(0.0, 0.0, 0.0), (15.0, 0.0, 0.0), (0.0, 15.0, 40.0),
               (-10.0, -10.0, -50.0)]
        src = openmc.IndependentSource(space=st.PointCloud(
            pts, strengths=[1.0, 4.0, 2.0, 1.0]))
    elif name == "meshspatial":
        mesh = openmc.CylindricalMesh(r_grid=[0.0, 6.0, 12.0, 19.0],
                                      phi_grid=[0.0, np.pi, 2 * np.pi],
                                      z_grid=[-60.0, 0.0, 60.0])
        strengths = np.tile([1.0, 2.0, 4.0], 4)       # r fastest
        src = openmc.IndependentSource(space=st.MeshSpatial(
            mesh, strengths, volume_normalized=True))
    elif name == "meshsource":
        mesh = openmc.RegularMesh()
        mesh.lower_left, mesh.upper_right = (-14.0, -14.0, -60.0), \
            (14.0, 14.0, 60.0)
        mesh.dimension = (2, 2, 3)
        parts = []
        for k in range(12):
            angle = st.Isotropic() if k % 2 == 0 else \
                st.Monodirectional((1.0, 0.0, 0.0))
            parts.append(openmc.IndependentSource(angle=angle,
                                                  strength=1.0 + k))
        src = openmc.MeshSource(mesh, parts)
    elif name == "file":
        rng = np.random.default_rng(11)
        n = 20_000
        r = 19.0 * np.sqrt(rng.random(n))
        ph = 2 * np.pi * rng.random(n)
        z = rng.uniform(-60.0, 60.0, n)
        mu = rng.uniform(-1.0, 1.0, n)
        a = 2 * np.pi * rng.random(n)
        s = np.sqrt(1 - mu * mu)
        w = np.where(rng.random(n) < 0.5, 0.5, 2.0)
        sites = [openmc.SourceParticle(r=(r[i] * np.cos(ph[i]),
                                          r[i] * np.sin(ph[i]), z[i]),
                                       u=(s[i] * np.cos(a[i]),
                                          s[i] * np.sin(a[i]), mu[i]),
                                       wgt=w[i]) for i in range(n)]
        path = Path(workdir) / "sites" / "cask_sites.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            openmc.write_source_file(sites, str(path))
        src = openmc.FileSource(str(path.resolve()))
    elif name == "material":
        src = openmc.IndependentSource(
            space=st.Box((-30.0, -30.0, -80.0), (30.0, 30.0, 80.0)),
            constraints={"domains": [mats["fuel"]]})
    elif name == "two":
        src = [openmc.IndependentSource(space=st.Point((0.0, 0.0, 30.0)),
                                        strength=1.0),
               openmc.IndependentSource(space=st.Box((-10.0, -10.0, -60.0),
                                                     (10.0, 10.0, 0.0)),
                                        strength=3.0)]
    else:
        raise KeyError(name)
    model.settings.source = src
    return model


def _nested_case(name):
    model = PROBLEMS["nested"]()
    univs = {u.name: u for u in model.geometry.get_all_universes().values()}
    cells = {c.name: c for c in model.geometry.get_all_cells().values()}
    # boxes around the rotated elements, so that OpenMC accepts at least
    # the 5% of sites it insists on (we accept any fraction)
    if name == "universe":
        dom, box = [univs["offset"]], ((-1.0, -1.0, -2.0), (3.0, 1.0, 2.0))
    else:
        dom, box = [cells["tilted-pin"]], ((-0.6, -1.0, -1.4),
                                            (3.0, 1.0, 1.4))
    model.settings.source = openmc.IndependentSource(
        space=st.Box(*box), constraints={"domains": dom})
    return model


CASK_CASES = ("cartesian", "cylindrical", "spherical", "pointcloud",
              "meshspatial", "meshsource", "file", "material", "two")
NESTED_CASES = ("universe", "deepcell")
CASES = CASK_CASES + NESTED_CASES


def build(name, workdir):
    return _cask_case(name, workdir) if name in CASK_CASES else \
        _nested_case(name)


def run(workdir, particles=100_000, batches=20, seed=5, names=None,
        log=print):
    out = {}
    for name in names or CASES:
        model = build(name, workdir)
        pr = openmc_import.load(model)
        t = time.perf_counter()
        n = particles * batches
        ref = openmc_ref.run(model, f"{workdir}/openmc_src_{name}",
                             n // validate.REF_BATCHES, validate.REF_BATCHES,
                             seed=seed + 100)
        t_ref = time.perf_counter() - t
        mc = transport.run(pr, particles, batches, seed=seed)
        ob = transport.run(pr, particles, batches, seed=seed,
                           backend=Oracle(), r_star=1.0)
        c_mc, c_ob = compare.compare(mc, ref), compare.compare(ob, ref)
        (p_mc, bad_mc), (p_ob, bad_ob) = validate.verdict(c_mc), \
            validate.verdict(c_ob)
        log(f"\n== {name}: {n:,} histories each; OpenMC {t_ref:.1f} s; "
            f"{ob.counters['balls']:,} balls; sites drawn again "
            f"{mc.counters['source_redrawn']:,}")
        log(compare.report(c_mc, "   plain MC vs OpenMC"))
        log(compare.report(c_ob, "   exact balls vs OpenMC"))
        for label, ok, bad in (("plain MC", p_mc, bad_mc),
                               ("exact balls", p_ob, bad_ob)):
            log(f"   {label}: {'PASS' if ok else 'FAIL'}"
                + ("" if ok else "  (" + "; ".join(bad) + ")"))
        out[name] = {"pass_mc": p_mc, "pass_balls": p_ob,
                     "mc_vs_openmc": c_mc, "balls_vs_openmc": c_ob,
                     "counters_mc": mc.counters}
    out["all_pass"] = all(v["pass_mc"] and v["pass_balls"]
                          for k, v in out.items() if k != "all_pass")
    log(f"\n== overall: {'PASS' if out['all_pass'] else 'FAIL'}")
    return out
