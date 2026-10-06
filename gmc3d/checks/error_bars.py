"""Is the error bar honest, and how stable is it? (both estimators)

Used by: run by hand (needs OpenMC, like the rest of checks/).  From gmc3d/:

    python -m checks.error_bars
    python -m checks.error_bars --problems slab cask --runs 30
    python -m checks.error_bars --problems sphere --backend table --r-star 2

Uses: core/transport.py and core/tallies.py (unchanged by this file),
problems/.

WHY.  A mean is only as trustworthy as its error bar, and an error bar is
only as good as the number of independent groups it is estimated from.
tallies.py used to estimate it from the scatter of the batch means -- 40
numbers in a race -- which makes the error bar itself uncertain by about
11% and a figure of merit, which goes as one over its square, by about 24%.
Measured on the slab on 5 and 6 October 2026, two honest runs disagreed by
a factor of two on "how many times faster than OpenMC" purely through that.
tallies.py now uses the thread chunks as the groups instead, which costs
nothing and gives nchunk x batches of them.

This file checks the new estimator the only way an error bar can be
checked: by running the same thing many times and seeing how much the
answer actually moves.

    truth          the standard deviation of the means of `runs`
                   independent runs -- what one run's error bar should be
                   predicting.  Its own uncertainty is 1 / sqrt(2(runs-1)).
    bias           each estimator's average, divided by that truth.  1.00
                   means honest; below 1 means the error bar lies low, which
                   is the dangerous direction.
    stability      the spread of the estimator across runs, relative to its
                   own average.  This is the quantity the fix targets.
    coverage       how often the run's mean landed within 1.96 error bars of
                   the grand mean.  Should be about 95%.

Seeds are spaced 1,000 apart.  A master seed is the generator's starting
state and each particle skips STRIDE numbers ahead of the one before
(core/rng.py), so any two master seeds give effectively disjoint streams;
the spacing is belt and braces.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ball import oracle, table
from core import openmc_import, transport
from problems import PROBLEMS

TABLE_FILE = Path(__file__).resolve().parent.parent / "data" / "ball" / \
    "table.npz"


def _quantities(r):
    """The answers worth checking: leakage and, if present, total flux."""
    q = {"leakage": (r.leakage, r.leakage_se, r.leakage_se_batches)}
    t = r.tallies.get("total")
    if t is not None:
        q["total flux"] = (float(t["mean"][0, 0]), float(t["se"][0, 0]),
                           None)
    return q


def study(names, runs=30, particles=10_000, batches=20, backend=None,
          r_star=2.0, nchunk=32, log=print):
    be = None
    if backend == "table":
        be = table.Table.load(TABLE_FILE)
    elif backend == "oracle":
        be = oracle.Oracle()
    out = {}
    for name in names:
        pr = openmc_import.load(PROBLEMS[name]())
        transport.run(pr, 1000, 1, seed=1)                   # compile
        rows = []
        for k in range(runs):
            r = transport.run(pr, particles, batches, seed=1 + 1000 * k,
                              backend=be, r_star=r_star,
                              mesh_rule="centre", nchunk=nchunk)
            rows.append(r)
        groups = rows[0].groups
        log(f"\n== {name}{'' if be is None else ' + ' + be.name}: "
            f"{runs} runs of {batches} x {particles:,} histories; the error "
            f"bar now comes from {groups:,} chunk groups, the old one from "
            f"{batches} batches")
        means = np.array([r.leakage for r in rows])
        se_c = np.array([r.leakage_se for r in rows])
        se_b = np.array([r.leakage_se_batches for r in rows])
        truth = means.std(ddof=1)
        unc = 1.0 / np.sqrt(2 * (runs - 1))
        grand = means.mean()
        log(f"  leakage {grand:.6g}; the spread of the {runs} run means is "
            f"{truth:.3g} (itself +-{100 * unc:.0f}%)")
        log(f"  {'estimator':18s} {'average':>10s} {'bias':>7s} "
            f"{'stability':>10s} {'coverage':>9s}")
        for lab, se in (("chunks (new)", se_c), ("batches (old)", se_b)):
            cov = np.mean(np.abs(means - grand) < 1.96 * se)
            log(f"  {lab:18s} {se.mean():10.3g} {se.mean() / truth:6.2f}x "
                f"{100 * se.std(ddof=1) / se.mean():9.1f}% "
                f"{100 * cov:8.0f}%")
        log(f"  expected stability: {100 / np.sqrt(2 * (groups - 1)):.1f}% "
            f"from {groups:,} groups, "
            f"{100 / np.sqrt(2 * (batches - 1)):.1f}% from {batches}"
            f" (a rare tally does worse: its groups are not Gaussian)")
        out[name] = {"runs": runs, "particles": particles,
                     "batches": batches, "groups": int(groups),
                     "truth": float(truth), "grand_mean": float(grand),
                     "se_chunks": se_c.tolist(), "se_batches": se_b.tolist(),
                     "means": means.tolist()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--problems", nargs="*", default=["slab", "sphere", "cask"])
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--particles", type=int, default=10_000)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--backend", choices=("mc", "table", "oracle"),
                   default="mc")
    p.add_argument("--r-star", type=float, default=2.0)
    p.add_argument("--nchunk", type=int, default=32)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    res = study(a.problems, a.runs, a.particles, a.batches,
                None if a.backend == "mc" else a.backend, a.r_star,
                a.nchunk)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"\nwritten to {a.json}")


if __name__ == "__main__":
    main()
