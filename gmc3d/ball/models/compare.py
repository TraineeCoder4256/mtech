"""Families side by side: accuracy against the exact walk, and cost per draw.

Used by: run.py (`python run.py compare`).
Uses: checks/ball_metrics.py (Innovation's ball-by-ball score, called
read-only), any backend.

The two numbers that decide whether a model is worth having in a ball:

 1. ACCURACY, ball by ball.  checks/ball_metrics.score: at radii from 1.5
    to 29 mfp, 200,000 exact walks against 200,000 draws -- the mean exit
    cosine, the mean path, the KS distances of both distributions, and the
    surviving weight at three absorption levels (the long-walk tail that
    went wrong in gmc2d's absorber cells).  Each gap also as z = gap /
    standard error; |z| under about 3 is indistinguishable at this size.
    The summary keeps the worst |z| and the worst KS (minus its noise
    level) per backend, so a ranking is one line.

 2. COST per draw, at three batch sizes (1k, 10k, 100k radii per call),
    the best of 3 timings each.  A ball step pays only if a draw costs
    less than the flights it removes, so this is half the verdict.

Every family is scored here by the same code, at the same radii, on the
same random numbers for the exact side.  The exact walk (`oracle`) and the
table can be listed too, as goalposts: the oracle's error is zero by
construction and the table's cost is close to free.
"""

import time

import numpy as np

from checks import ball_metrics

BATCHES = (1_000, 10_000, 100_000)


def cost_per_draw(backend, batches=BATCHES, repeats=3, seed=4242):
    """Best-of-`repeats` seconds per draw at each batch size."""
    from core import rng
    out = {}
    gen = np.random.default_rng(seed)
    for n in batches:
        R = np.exp(gen.uniform(np.log(max(backend.r_min, 1.0)),
                               np.log(min(backend.r_max, 30.0)), n))
        state = rng.streams(np.uint64(seed), 0, n)
        backend.sample(R[:min(n, 256)], state, np.arange(min(n, 256)))
        best = np.inf
        for _ in range(repeats):
            t = time.perf_counter()
            backend.sample(R, state, np.arange(n))
            best = min(best, time.perf_counter() - t)
        out[n] = best / n
    return out


def summary(rows):
    """Worst |z| over every compared quantity, and worst KS excess."""
    zs, ks = [], []
    for r in rows:
        for key, val in r.items():
            if isinstance(val, dict) and "z" in val:
                zs.append(abs(val["z"]))
        ks.append(max(r["ks_mu"], r["ks_s"]) - r["ks_noise_95"])
    return {"worst_abs_z": float(max(zs)), "worst_ks_excess": float(max(ks))}


def run(backends, log=print, n=200_000):
    """backends: {label: backend}.  Returns the results dict."""
    res = {}
    for label, be in backends.items():
        t0 = time.time()
        rows = ball_metrics.score(be, n=n)
        cost = cost_per_draw(be)
        desc = be.describe() if hasattr(be, "describe") else {}
        res[label] = {"rows": rows, "summary": summary(rows),
                      "cost_per_draw_s": cost,
                      "nfe": desc.get("nfe"), "params": desc.get("params"),
                      "steps_done": desc.get("steps_done"),
                      "train_seconds": desc.get("train_seconds")}
        log(ball_metrics.report(rows, f"\n{label} vs the exact walk"))
        log(f"  ({time.time() - t0:.0f} s)")
    log("\nsummary (|z| < ~3 and KS excess <= 0: indistinguishable from "
        "the exact walk at this sample size)")
    log(f"  {'backend':<16}{'weights':>10}{'nfe':>5}{'worst |z|':>11}"
        f"{'KS excess':>11}" +
        "".join(f"{'us/draw @' + format(b, ','):>18}" for b in BATCHES))
    for label, r in res.items():
        p = f"{r['params']:,}" if r["params"] else "-"
        log(f"  {label:<16}{p:>10}{(r['nfe'] or '-'):>5}"
            f"{r['summary']['worst_abs_z']:>11.1f}"
            f"{r['summary']['worst_ks_excess']:>+11.4f}" +
            "".join(f"{1e6 * r['cost_per_draw_s'][b]:>18.3f}"
                    for b in BATCHES))
    return res
