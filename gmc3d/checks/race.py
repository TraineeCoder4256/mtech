"""Head to head: OpenMC against our plain Monte Carlo and our ball steps.

Used by: run.py (`python run.py race`).

Every contender solves the same openmc.Model with the same number of
histories on the same 4 cores, and is timed on transport only (start-up,
compilation and file writing are reported apart).  The contenders:

    OpenMC analog        OpenMC's default: a particle is killed when it is
                         absorbed
    OpenMC implicit      OpenMC with survival biasing: absorption lowers the
                         weight instead, with Russian roulette below 0.25,
                         the same weight game as ours.  This is the fair
                         algorithm-for-algorithm comparison
    ours, plain MC       our solver, no balls
    ours + exact walk    balls, the exact walk inside each (the truth)
    ours + network       balls, the trained normalizing flow
    ours + table         balls, the lookup table (for reference)

Faster per history is not the whole story: the two OpenMC modes and ours
have different noise per history.  So each row also gets a figure of merit
(FOM) = 1 / (relative error^2 x seconds).  Higher is better, and the FOM
ratio is how many times faster a contender reaches the same error bar.

A FOM is only as good as the error bar under it, and an error bar estimated
from G independent groups carries about sqrt(2 / G) of its own -- 22% at 40
groups, which is how two honest races on 5 and 6 October 2026 came to
disagree by a factor of two on the slab.  Both sides are now split finely:
ours takes its groups from the thread chunks (nchunk x batches, 1,280 here;
see core/tallies.py), and OpenMC runs the same histories in REF_BATCHES
batches instead of `batches`, which costs no time and does not move its
answer at all (see checks/openmc_ref.py).  That puts both error bars at
about 2%, so a FOM ratio is good to a few per cent rather than a factor.
Timings themselves still repeat to only about 10%.

Ball runs use the 'centre' mesh rule (balls kept full-size; mesh tallies
smeared, research item R4), as in measure.py.  The answers checked are
leakage, total flux and, where a problem has one, its detector.
"""

import time

import numpy as np

from core import openmc_import, rng, transport
from problems import PROBLEMS
from . import compare, openmc_ref

# (label, backend name or None, R*).  R* = 2 is the best measured for the
# exact walk and the table; R* = 10 for the network (measure.log).
CONTENDERS = (("ours, plain MC", None, None),
              ("ours + exact walk", "oracle", 2.0),
              ("ours + network", "network", 3.0),
              ("ours + network", "network", 10.0),
              ("ours + table", "table", 2.0))

# the quantity a user of each problem would care about, beyond leakage
DETECTORS = {"cask": ("cell", 5, "detector flux")}

# batches for the OpenMC reference runs: the same histories, split finely so
# its error bar is as well determined as ours (see the docstring)
REF_BATCHES = 1000


def _quantities(name, r):
    """{label: (mean, se)} of the answers compared in every row."""
    q = {"leakage": (r.leakage, r.leakage_se)}
    t = r.tallies.get("total")
    if t is not None:
        q["total flux"] = (float(t["mean"][0, 0]), float(t["se"][0, 0]))
    if name in DETECTORS:
        tally, b, label = DETECTORS[name]
        t = r.tallies[tally]
        q[label] = (float(t["mean"][b, 0]), float(t["se"][b, 0]))
    return q


def _row(label, name, r, seconds, extra=None):
    q = _quantities(name, r)
    fom = {k: (1.0 / ((se / m) ** 2 * seconds) if m and se else float("nan"))
           for k, (m, se) in q.items()}
    row = {"label": label, "seconds": seconds, "quantities": q, "fom": fom,
           "counters": r.counters, "timing": r.timing,
           "groups": getattr(r, "groups", 0),
           "leakage_se_batches": getattr(r, "leakage_se_batches", 0.0)}
    row.update(extra or {})
    return row


def race(names, backends, particles=100_000, batches=40, seed=11,
         workdir="data/openmc", contenders=CONTENDERS, log=print):
    """backends: {name: backend object} for the names used in
    `contenders`.  Returns {problem: [rows]}."""
    out = {}
    for name in names:
        n = particles * batches
        log(f"\n== {name}: {n:,} histories per contender, "
            f"{particles:,} per batch")
        rows = []
        for sb in (False, True):
            model = PROBLEMS[name]()
            model.settings.survival_biasing = sb
            if sb:
                model.settings.cutoff = {"weight": 0.25, "weight_avg": 1.0}
            tag = "implicit" if sb else "analog"
            ref_b = max(REF_BATCHES, batches)
            ref = openmc_ref.run(model, f"{workdir}/race_{name}_{tag}",
                                 max(n // ref_b, 1), ref_b, seed=seed + 100,
                                 reuse=False)
            rows.append(_row(f"OpenMC {tag}", name, ref,
                             ref.timing["openmc_transport_s"],
                             {"startup_s": ref.timing["wall_s"] -
                              ref.timing["openmc_transport_s"]}))
        pr = openmc_import.load(PROBLEMS[name]())
        t = time.perf_counter()
        transport.run(pr, 1000, 1, seed=seed)             # compile (numba)
        compile_s = time.perf_counter() - t
        for be in backends.values():                      # warm up draws
            be.sample(np.full(16, 5.0), rng.streams(np.uint64(9), 0, 16),
                      np.arange(16))
        for label, b, rs in contenders:
            if b is not None and b not in backends:
                continue
            be = backends.get(b) if b else None
            r = transport.run(pr, particles, batches, seed=seed, backend=be,
                              r_star=rs or 3.0, mesh_rule="centre")
            sec = r.timing["transport_s"] + r.timing["backend_s"]
            lab = label + (f" (R* = {rs:g})" if rs else "")
            rows.append(_row(lab, name, r, sec,
                             {"compile_s": compile_s,
                              "backend_s": r.timing["backend_s"],
                              "balls_per_history": r.counters["balls"] / n,
                              "flights_per_history":
                                  r.counters["flights"] / n}))
        _report(rows, log)
        out[name] = rows
    return out


def _report(rows, log):
    ref = rows[0]                                        # OpenMC analog
    keys = list(ref["quantities"])
    log(f"  {'contender':32s} {'seconds':>8s} {'vs OpenMC':>9s} "
        + " ".join(f"{'FOM x (' + k + ')':>22s}" for k in keys))
    for r in rows:
        log(f"  {r['label']:32s} {r['seconds']:8.2f} "
            f"{ref['seconds'] / r['seconds']:8.2f}x "
            + " ".join(f"{r['fom'][k] / ref['fom'][k]:21.2f}x"
                       for k in keys))
    log("  answers (z against OpenMC analog):")
    for r in rows:
        parts = []
        for k in keys:
            m, se = r["quantities"][k]
            m0, se0 = ref["quantities"][k]
            z = (m - m0) / np.hypot(se, se0) if r is not ref else 0.0
            parts.append(f"{k} {m:.6g} +- {se:.2g} (z {z:+.1f})")
        log(f"    {r['label']:32s} " + "; ".join(parts))
    extra = [r for r in rows if "flights_per_history" in r]
    log("  ours: flights / balls per history, backend seconds: " + "; ".join(
        f"{r['label'].replace('ours', '').strip(' ,+')} "
        f"{r['flights_per_history']:.1f} / {r['balls_per_history']:.2f}, "
        f"{r['backend_s']:.2f} s" for r in extra))
    log(f"  error bars: ours from {extra[0].get('groups', 0):,} chunk groups, "
        f"OpenMC's from {REF_BATCHES:,} batches" if extra else "")
    log(f"  one-off costs: OpenMC start-up {rows[0]['startup_s']:.1f} s, "
        f"our compile {extra[0]['compile_s']:.1f} s" if extra else "")
