"""Point detectors checked three ways (v1.6; `python run.py detcheck`).

Used by: run.py.  Writes detcheck.json, which checks/score.py reads.

1. EXACT.  A point source in two absorbing shells that do not scatter: the
   flux at a point is exp(-tau) / (4 pi d^2) exactly, tau summed along
   the straight line through both shells.  Every history scores the same,
   so the answer must agree to rounding.
2. BALLS AGREE WITH PLAIN MC.  In a scattering sphere, the same detectors
   with exact balls (their scatters unseen, scored at profile points) and
   with plain Monte Carlo (every scatter scored).  The claim of the method
   is that the two have the same mean.
3. AGAINST OPENMC.  OpenMC 0.16 has no point detector, so the reference
   is the next best thing it does have: the track-length flux in a small
   void sphere around the point, divided by its volume, from 10 times our
   histories.  In a void the flux is smooth, so the sphere's average
   differs from the point value only at second order in its radius (about
   rho^2 / 10 times the flux's curvature): rho = 0.5 cm for the cavity in
   the medium, 1.5 cm in the outer void, where the flux falls like 1/r^2
   from a source about 13 cm away (a 0.3% difference, below the error).
   Detectors in the scattering medium itself have no such reference
   (a small sphere of material is not smooth enough at the point), so
   they are checked by 2 only.

The medium detectors use an averaging radius r0 = 0.3 cm (see
core/detectors.py: without it the estimator's variance is infinite for a
point inside a scattering material); the void ones use none.
"""

import time

import numpy as np
import openmc

from ball.oracle import Oracle
from core import openmc_import, transport
from problems import common
from . import openmc_ref, validate

RHO_CAV, RHO_OUT = 0.5, 1.5   # cm, OpenMC's reference spheres
REF_SCALE = 10                # OpenMC histories, times ours


def exact_model():
    mats, m = common.one_group_materials("det_exact", {"in": (0.0, 0.2),
                                                       "out": (0.0, 0.05)})
    s1 = openmc.Sphere(r=3.0)
    s2 = openmc.Sphere(r=10.0, boundary_type="vacuum")
    model = openmc.Model(openmc.Geometry([
        openmc.Cell(fill=m["in"], region=-s1),
        openmc.Cell(fill=m["out"], region=+s1 & -s2)]), mats,
        common.settings(common.point_source((0.0, 0.0, 0.0))),
        openmc.Tallies([common.flux_tally("total")]))
    pts = {"beyond_both": (5.0, 1.0, 0.0), "inner": (2.0, 0.0, 0.0),
           "far": (-7.0, -3.0, 2.0)}
    for name, p in pts.items():
        common.point_detector(model, name, p)
    return model, pts


def exact_value(p):
    d = float(np.linalg.norm(p))
    tau = 0.2 * min(d, 3.0) + 0.05 * max(d - 3.0, 0.0)
    return np.exp(-tau) / (4 * np.pi * d * d)


def scatter_model():
    """A scattering sphere (r 10) inside a void sphere (r 16, vacuum), a
    void cavity inside the medium, and small void spheres for OpenMC."""
    mats, m = common.one_group_materials("det_scatter",
                                         {"medium": (0.8, 0.05)})
    med = openmc.Sphere(r=10.0)
    world = openmc.Sphere(r=16.0, boundary_type="vacuum")
    cav = openmc.Sphere(x0=4.0, r=RHO_CAV)            # cavity in the medium
    probe = openmc.Sphere(x0=13.0, y0=2.0, r=RHO_OUT)  # in the outer void
    c_cav = openmc.Cell(name="cavity", region=-cav)
    c_probe = openmc.Cell(name="probe", region=-probe)
    cells = [openmc.Cell(fill=m["medium"], region=-med & +cav), c_cav,
             openmc.Cell(region=+med & -world & +probe), c_probe]
    model = openmc.Model(openmc.Geometry(cells), mats,
                         common.settings(common.point_source((-5.0, 0.0,
                                                              0.0))),
                         openmc.Tallies([common.flux_tally(
                             "spheres", openmc.CellFilter([c_cav, c_probe]),
                             scores=("flux",))]))
    common.point_detector(model, "cavity", (4.0, 0.0, 0.0))
    common.point_detector(model, "outside", (13.0, 2.0, 0.0))
    common.point_detector(model, "medium", (0.0, 5.0, 0.0), r0=0.3)
    common.point_detector(model, "medium_deep", (-2.0, -3.0, 1.0), r0=0.3)
    return model


def _z(a, sa, b, sb):
    return float((a - b) / max(np.hypot(sa, sb), 1e-300))


def run(workdir, particles=100_000, batches=20, seed=7, log=print):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        log(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    log("== 1. exact: point source in two absorbing shells, no scatter")
    model, pts = exact_model()
    r = transport.run(openmc_import.load(model), 2000, 4, seed=seed)
    for name, p in pts.items():
        v, e = r.tallies[name]["mean"][0, 0], exact_value(p)
        rel = abs(v - e) / e
        add(f"exact {name}", rel < 1e-9,
            f"ours {v:.10g}, exact {e:.10g}, rel err {rel:.1e}")

    log(f"\n== 2./3. scattering sphere, {particles * batches:,} histories")
    model = scatter_model()
    pr = openmc_import.load(model)
    t = time.perf_counter()
    mc = transport.run(pr, particles, batches, seed=seed)
    t_mc = time.perf_counter() - t
    t = time.perf_counter()
    ob = transport.run(pr, particles, batches, seed=seed + 1,
                       backend=Oracle(), r_star=1.0)
    t_ob = time.perf_counter() - t
    log(f"   plain MC {t_mc:.0f} s; exact balls {t_ob:.0f} s "
        f"({ob.counters['balls']:,} balls); lost {mc.counters['lost']} / "
        f"{ob.counters['lost']}")
    for name, _, r0 in pr.detectors:
        a, sa = mc.tallies[name]["mean"][0, 0], mc.tallies[name]["se"][0, 0]
        b, sb = ob.tallies[name]["mean"][0, 0], ob.tallies[name]["se"][0, 0]
        z = _z(b, sb, a, sa)
        add(f"balls vs plain MC, {name}", abs(z) <= validate.Z1_MAX,
            f"balls {b:.6g} +- {sb:.2g}, plain {a:.6g} +- {sa:.2g}, "
            f"z {z:+.2f}" + (f" (r0 {r0} cm)" if r0 else ""))

    n = particles * batches
    ref = openmc_ref.run(model, f"{workdir}/detcheck", n // 400,
                         400 * REF_SCALE, seed=seed + 100)
    for k, (name, rho) in enumerate((("cavity", RHO_CAV),
                                     ("outside", RHO_OUT))):
        vol = 4.0 / 3.0 * np.pi * rho ** 3
        e, se = ref.tallies["spheres"]["mean"][k, 0] / vol, \
            ref.tallies["spheres"]["se"][k, 0] / vol
        mine, mine_se = mc.tallies["spheres"]["mean"][k, 0] / vol, \
            mc.tallies["spheres"]["se"][k, 0] / vol
        log(f"   {name}: OpenMC sphere average {e:.6g} +- {se:.2g}; ours "
            f"(track length, same sphere) {mine:.6g} +- {mine_se:.2g}")
        for label, r in (("plain MC", mc), ("exact balls", ob)):
            v, sv = r.tallies[name]["mean"][0, 0], r.tallies[name]["se"][0, 0]
            z = _z(v, sv, e, se)
            add(f"{label} detector vs OpenMC, {name}",
                abs(z) <= validate.Z1_MAX,
                f"detector {v:.6g} +- {sv:.2g}, OpenMC {e:.6g} +- {se:.2g},"
                f" z {z:+.2f}")
    timing = {"plain_mc_s": t_mc, "balls_s": t_ob}
    return {"checks": checks, "timing": timing,
            "counters_mc": mc.counters, "counters_balls": ob.counters}
