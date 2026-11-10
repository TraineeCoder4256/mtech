"""The acceptance gate for interface v2 models (anisotropic balls).

Used by: run.py (`python run.py gate2 <family>_v2[_seed<k>]`).  Writes
gate2_<name>.json and a log.
Uses: ball/walk.py walk_aniso (the truth), ball/laws.py held_out (laws no
model trained on), a v2 backend (ball/models AnisoBackend), called as the
driver calls it: .sample_aniso(R, mats, ang, streams, idx), whole balls
(from v2.1.4; m2.1 models walk the first scatter exactly and draw only
the k >= 2 exits, which .draw(R, g) alone cannot show).

Agreed 5 Nov 2026 (Innovation/Development protocol): the gate checks
accuracy against the exact walk at every radius, near r_max too, the
weight-factor tail, and cost per draw, and refuses a model whose manifest
names another interface.  A model that fails keeps the previous one.

PER LAW AND RADIUS.  For each held-out law (the test problems' laws) and
R in RADII: N exact walks against N model draws at the same (R, law).
For each output (cos_n, mu, phi, s): the difference of means in
standard errors (z) and the two-sample KS distance.  For each kappa in
KAPPAS (sigma_a / sigma_s): the weight factor E[exp(-kappa s)], the share
of weight that survives the ball, as z.  Errors there go straight into
leakage and absorption.

THE RULE (provisional; Phani decides the final one):
    * every weight-factor |z| <= 3.29 (1 in 1,000 by chance per check),
      at the gate's N
    * every KS distance <= KS_MAX = 0.03.  For comparison, the noise is
      about 0.006 at N = 100,000, and the moment truncation alone costs
      up to 0.018 for a peaked law in a 1 mfp ball
      (runs/2026-11-10/interface-v2/README.md)
    * every draw inside the interface ranges (interface.check_aniso_outputs)
Also REPORTED, not part of the rule (v2.1.5): each weight error as a share
of what the ball absorbs, (E_walk - E_model) / (1 - E_walk), and how many
weight checks a two-condition rule would fail (|z| > 3.29 AND that share
off by more than 0.5%, 1%, 2%).  Where the weight factor barely varies
(R = 1) a pure z rule fails a 0.05% bias; that decision is Phani's.
Means are reported (z) but do not fail a model on their own: with
100,000 draws a mean shift of 1% is many sigma and may not matter to a
problem.  The end-to-end judge is `run.py anisocheck` with the model as
backend, against OpenMC.

COST: microseconds per ball at a batch of 100,000, against the exact walk
at the same radii; random streams are made outside the timers.
"""

import json
import time
from pathlib import Path

import numpy as np

from ball import interface as I, laws as LW, walk
from core import rng, scattering as SC

RADII = (1.0, 1.5, 3.0, 5.0, 10.0, 20.0, 29.0)
KAPPAS = (0.001, 0.01, 0.1)
KS_MAX = 0.03
Z_MAX = 3.29
ABSORB_TOLS = (0.005, 0.01, 0.02)   # reported only (v2.1.5), see run()
OUT = ("cos_n", "mu", "phi", "s")


def _ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _z(a, b):
    se = np.hypot(a.std() / np.sqrt(a.size), b.std() / np.sqrt(b.size))
    return float((b.mean() - a.mean()) / max(se, 1e-300))


def check_manifest(backend):
    """The model's recorded interface must be ANISO_INTERFACE_VERSION."""
    info = getattr(backend, "info", {}) or {}
    v = info.get("interface")
    if v is None and type(backend).__name__ == "AnisoBackend":
        # ball/models/store.read(expect=2) already refused any other
        # version when the model was loaded
        v = I.ANISO_INTERFACE_VERSION
    return v == I.ANISO_INTERFACE_VERSION, v


def run(backend, n=100_000, seed=21, log=print, laws=None):
    ok_manifest, v = check_manifest(backend)
    log(f"model {getattr(backend, 'name', '?')}: manifest interface {v} "
        f"({'OK' if ok_manifest else 'REFUSED: need '}"
        f"{'' if ok_manifest else I.ANISO_INTERFACE_VERSION})")
    out = {"model": getattr(backend, "name", None), "manifest_interface": v,
           "manifest_ok": ok_manifest, "rows": [], "n": n}
    if not ok_manifest:
        out["pass"] = False
        return out
    laws = laws or LW.held_out()
    fails = []
    t_model = t_walk = 0.0
    n_draws = 0
    for name, law in laws.items():
        g = LW.moments(law)
        ang = SC.pack([law])
        log(f"\n== {name}: g1 {g[0]:+.3f}")
        for R in RADII:
            # own streams per (law, radius), so rows share no numbers; the
            # model's from another master seed, so its exact first scatter
            # (walk.first_scatter_aniso, v2.1.3) is not the walk's own.
            # Made outside the timers: rng.streams is a Python loop.
            first = len(out["rows"]) * n
            st = rng.streams(np.uint64(seed), first, n)
            st_model = rng.streams(np.uint64(seed + 1), first, n)
            Rs, mats, idx = np.full(n, R), np.zeros(n, np.int64), np.arange(n)
            t = time.perf_counter()
            ex = walk.walk_many_aniso(Rs, mats, ang, st, idx)[:4]
            t_walk += time.perf_counter() - t
            t = time.perf_counter()
            md = backend.sample_aniso(Rs, mats, ang, st_model, idx)
            t_model += time.perf_counter() - t
            n_draws += n
            md = tuple(np.asarray(a, np.float64) for a in md)
            bad = I.check_aniso_outputs(np.full(n, R), *md)
            row = {"law": name, "R": R,
                   "z": dict(zip(OUT, (_z(a, b) for a, b in zip(ex, md)))),
                   "ks": dict(zip(OUT, (_ks(a, b) for a, b in zip(ex, md)))),
                   "out_of_range": sum(bad.values())}
            wz, ar = {}, {}
            for k in KAPPAS:
                we, wm = np.exp(-k * ex[3]), np.exp(-k * md[3])
                wz[k] = _z(we, wm)
                # the same error as a share of what the ball absorbs
                ar[k] = float((we.mean() - wm.mean()) / (1.0 - we.mean()))
            row["weight_z"] = wz
            row["absorb_rel"] = ar
            why = [f"KS {o} {row['ks'][o]:.3f}" for o in OUT
                   if row["ks"][o] > KS_MAX]
            why += [f"weight z {wz[k]:+.2f} at kappa {k}" for k in KAPPAS
                    if abs(wz[k]) > Z_MAX]
            if row["out_of_range"]:
                why.append(f"{row['out_of_range']} draws out of range")
            row["pass"] = not why
            fails += [f"{name} R {R:g}: {w}" for w in why]
            out["rows"].append(row)
            if any(abs(wz[k]) > Z_MAX for k in KAPPAS):
                log("           absorbed share, model/walk - 1: " + " ".join(
                    f"{ar[k]:+.2%}" for k in KAPPAS))
            log(f"  R {R:4g}  " + "  ".join(
                f"{o} z {row['z'][o]:+6.2f} KS {row['ks'][o]:.3f}"
                for o in OUT) + "  weight z " + " ".join(
                f"{wz[k]:+.2f}" for k in KAPPAS) +
                ("" if row["pass"] else "  FAIL"))
    out["us_per_draw_model"] = 1e6 * t_model / n_draws
    out["us_per_draw_walk"] = 1e6 * t_walk / n_draws
    out["failures"] = fails
    out["pass"] = not fails
    # REPORT ONLY, for Phani's decision on the final rule: the failing
    # weight checks that a two-condition rule (|z| > Z_MAX AND the
    # absorbed share off by more than tol) would still fail
    wf = [(r["weight_z"][k], r["absorb_rel"][k]) for r in out["rows"]
          for k in KAPPAS]
    out["weight_fails_two_condition"] = {
        f"{tol:g}": sum(abs(z) > Z_MAX and abs(a) > tol for z, a in wf)
        for tol in ABSORB_TOLS}
    out["weight_fails_z_only"] = sum(abs(z) > Z_MAX for z, _ in wf)
    log(f"\ncost: model {out['us_per_draw_model']:.2f} us per draw, exact "
        f"walk {out['us_per_draw_walk']:.2f} us (batch {n:,}, these radii)")
    log(f"gate v2: {'PASS' if out['pass'] else 'FAIL'} ({len(fails)} "
        f"failing checks of {len(out['rows']) * (len(OUT) + len(KAPPAS))})")
    for f in fails[:40]:
        log("  " + f)
    log(f"weight checks failing: {out['weight_fails_z_only']} on |z| alone;"
        " with the absorbed share also off by more than " + ", ".join(
            f"{float(t):.1%}: {c}" for t, c in
            out["weight_fails_two_condition"].items()) +
        "  (report only; the rule above decides)")
    return out


def save(out, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    p = Path(folder, f"gate2_{out['model']}.json")
    p.write_text(json.dumps(out, indent=1, default=float))
    return p
