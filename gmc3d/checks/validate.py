"""The validation suite (SRS S1-VAL-1 to -3), with mesh pass/fail since v1.1.

Used by: run.py (`python run.py validate`).

For every test problem:
    OpenMC (multigroup)          the reference
    ours, plain Monte Carlo      must match OpenMC     -> the solver is right
    ours, exact ball steps       must match OpenMC     -> the driver is right
Plus the analytic checks (checks/analytic.py), with and without balls.

The ball runs use R* = 1 (as many balls as possible, the hardest test of
the driver) and the 'profile' mesh rule, so balls stay full-size.  Since
v1.1 EVERY tally counts towards the verdict, mesh tallies included (before
v1.1 the 'centre' rule smeared them, so they were only reported).

A comparison PASSES when no number disagrees by more than chance allows at
a one-in-a-thousand level (P_FAIL):

    a tally with several bins    the sum of z^2 over its N bins, against the
                                 chi-squared distribution with N degrees of
                                 freedom (so 1,000 voxels may sit at
                                 chi2/dof 1.14, 60 depth bins at 1.6)
    a single number              |z| <= 3.29 (leakage, totals, one cell)

This only works if the error bars themselves are right.  OpenMC's come from
its batch means, so it runs the same histories in REF_BATCHES batches
(free, see checks/openmc_ref.py): with 20 batches each OpenMC error bar is
uncertain by ~16% and z follows Student's t with 19 degrees of freedom,
whose z^2 averages 1.12 -- enough to fail every big mesh on its own.
"""

import time

import numpy as np
from scipy import stats

from ball.oracle import Oracle
from core import openmc_import, transport
from problems import PROBLEMS
from . import analytic, compare, openmc_ref

PROBLEM_ORDER = ("sphere", "slab", "curved", "nested", "lattice3d")
VALIDATE_ORDER = PROBLEM_ORDER + ("cask", "tilted", "hexcore",
                                   "sector")
P_FAIL = 1.0e-3
Z1_MAX = float(stats.norm.isf(P_FAIL / 2))          # 3.29
REF_BATCHES = 400


def verdict(cmp):
    """(passed, list of the failing entries) for one compare.compare()."""
    bad = []
    for key, e in cmp.items():
        if "z" in e or "ours" in e:
            z = e["z"] if "z" in e else (e["ours"] - e["ref"]) / max(
                np.hypot(e["ours_se"], e["ref_se"]), 1e-300)
            if abs(z) > Z1_MAX:
                bad.append(f"{key} z {z:+.2f}")
        elif e["bins"] > 0:
            p = stats.chi2.sf(e["chi2_dof"] * e["bins"], e["bins"])
            if p < P_FAIL:
                bad.append(f"{key} chi2/dof {e['chi2_dof']:.3f} over "
                           f"{e['bins']} bins (p {p:.1e})")
    return not bad, bad


def suite(workdir, particles=100_000, batches=20, seed=5, names=None,
          log=print):
    names = names or VALIDATE_ORDER
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
        n = particles * batches
        ref = openmc_ref.run(model, f"{workdir}/openmc_{name}",
                             n // REF_BATCHES, REF_BATCHES, seed=seed + 100)
        t_ref = time.perf_counter() - t
        mc = transport.run(pr, particles, batches, seed=seed)
        orc = Oracle()
        ob = transport.run(pr, particles, batches, seed=seed, backend=orc,
                           r_star=1.0, mesh_rule="profile")
        c_mc, c_ob = compare.compare(mc, ref), compare.compare(ob, ref)
        (p_mc, bad_mc), (p_ob, bad_ob) = verdict(c_mc), verdict(c_ob)
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
        for label, ok, bad in (("plain MC", p_mc, bad_mc),
                               ("exact balls", p_ob, bad_ob)):
            log(f"   {label}: {'PASS' if ok else 'FAIL'}"
                + ("" if ok else "  (" + "; ".join(bad) + ")"))
        out["problems"][name] = {"pass_mc": p_mc, "pass_balls": p_ob,
                                 "mc_vs_openmc": c_mc,
                                 "balls_vs_openmc": c_ob,
                                 "counters_mc": mc.counters,
                                 "counters_balls": ob.counters,
                                 "timing_mc": mc.timing,
                                 "timing_balls": ob.timing,
                                 "timing_openmc": ref.timing}
    out["all_pass"] = all(v["pass_mc"] and v["pass_balls"]
                          for v in out["problems"].values())
    log(f"\n== overall: {'PASS' if out['all_pass'] else 'FAIL'}")
    return out
