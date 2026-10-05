"""The Stage 1 validation suite (SRS S1-VAL-1 to -3).

Used by: run.py (`python run.py validate`).

For every test problem:
    OpenMC (multigroup)          the reference
    ours, plain Monte Carlo      must match OpenMC     -> the solver is right
    ours, exact ball steps       must match OpenMC     -> the driver is right
Plus the analytic checks (checks/analytic.py), with and without balls.

The ball runs use R* = 1 (as many balls as possible, the hardest test of
the driver) and the 'centre' mesh rule, so balls are not shrunk away by the
mesh.  Total, cell and material tallies are exact under either rule; mesh
tallies under 'centre' are expected to differ near steep gradients (that is
research item R4), so they are reported but not used as a pass/fail.
"""

import time

import numpy as np

from ball.oracle import Oracle
from core import openmc_import, transport
from problems import PROBLEMS
from . import analytic, compare, openmc_ref

PROBLEM_ORDER = ("sphere", "slab", "curved", "nested", "lattice3d")


def suite(workdir, particles=100_000, batches=20, seed=5, names=None,
          log=print):
    names = names or PROBLEM_ORDER
    out = {"analytic": [], "problems": {}}

    log("== analytic checks")
    rows = analytic.pure_absorber() + analytic.infinite_medium()
    log(analytic.report(rows))
    log("   ... the same with exact ball steps (R* = 1)")
    rows_b = analytic.pure_absorber(Oracle(), 1.0) + \
        analytic.infinite_medium(Oracle(), 1.0)
    log(analytic.report(rows_b))
    out["analytic"] = [list(r) for r in rows + rows_b]

    for name in names:
        model = PROBLEMS[name]()
        pr = openmc_import.load(model)
        t = time.perf_counter()
        ref = openmc_ref.run(model, f"{workdir}/openmc_{name}", particles,
                             batches, seed=seed + 100)
        t_ref = time.perf_counter() - t
        mc = transport.run(pr, particles, batches, seed=seed)
        orc = Oracle()
        ob = transport.run(pr, particles, batches, seed=seed, backend=orc,
                           r_star=1.0, mesh_rule="centre")
        c_mc, c_ob = compare.compare(mc, ref), compare.compare(ob, ref)
        log(f"\n== {name}: {particles * batches:,} histories each")
        log(f"   time: OpenMC {t_ref:.1f} s (incl. start-up), plain MC "
            f"{mc.timing['transport_s']:.1f} s, exact balls "
            f"{ob.timing['transport_s'] + ob.timing['backend_s']:.1f} s")
        log(f"   exact balls: {ob.counters['balls']:,} balls, "
            f"{orc.scatters:,} scatters inside them; lost particles "
            f"{mc.counters['lost']} / {ob.counters['lost']}")
        log(compare.report(c_mc, "   plain MC vs OpenMC"))
        log(compare.report(c_ob, "   exact balls vs OpenMC"))
        log(f"   conservation: plain MC {analytic.conservation(mc)[1]:.6f}, "
            f"exact balls {analytic.conservation(ob)[1]:.6f}")
        out["problems"][name] = {"mc_vs_openmc": c_mc,
                                 "balls_vs_openmc": c_ob,
                                 "counters_mc": mc.counters,
                                 "counters_balls": ob.counters,
                                 "timing_mc": mc.timing,
                                 "timing_balls": ob.timing,
                                 "timing_openmc": ref.timing}
    return out
