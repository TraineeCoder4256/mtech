"""Interface v2 models (anisotropic balls) against the exact walk.

Used by: runs/2026-11-10/models-v2/ (the first v2 comparison), by hand.
Uses: ball/walk.py (walk_many_aniso, the reference), ball/laws.py (the
held-out laws), backend.AnisoBackend (the models).

This is the MODEL SIDE's check, for choosing between families and for
Phani's comparison.  The acceptance gate stays Innovation's (checks/); it
decides whether transport may use a model by default.

WHAT IS COMPARED.  For each law the models never saw (laws.held_out: the
seven laws of Innovation's anisotropic test problems) and each radius R in
RADII, N exact walks against N model draws:

  mean z      for cos_n, mu, phi and s / R: the gap between the two means
              in standard errors.  Under about 3: the two cannot be told
              apart at this sample size.
  KS excess   for each output, the largest gap between the two cumulative
              distributions minus its 95% noise level (1.36 sqrt(2 / N));
              above 0 means the SHAPE differs, not just the mean.
  weight z    E[exp(-kappa s)] at kappa = 0.01 and 0.1, as v1's
              ball_metrics: what a ball passes on in an absorbing medium.

Two extra rows:
  isotropic   the law g = 0 (a flat table, as the training slice): the
              model must give v1's (mu, s) and the lean of cos_n toward
              the first direction that Innovation measured.
  smoke       Innovation's P3 reference [1, 0.6, 0.3, 0.1] at R = 5:
              cos_n 0.382, mu 0.778, phi 1.864, s 8.91.
"""

import time

import numpy as np

RADII = (1.0, 3.0, 5.0, 10.0, 30.0)
KAPPAS = (0.01, 0.1)
OUTS = ("cos_n", "mu", "phi", "s/R")


def _ks(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / a.size
                               - np.searchsorted(b, x, "right") / b.size)))


def _z(x, y):
    se = np.sqrt(x.var() / x.size + y.var() / y.size)
    return float((y.mean() - x.mean()) / se) if se > 0 else 0.0


def laws():
    """[(name, law)]: the held-out laws, then the isotropic one."""
    from core import scattering as SC
    from .. import laws as LW
    out = list(LW.held_out().items())
    flat = (np.full(SC.DEFAULT_NMU, 0.5),
            np.linspace(0.0, 1.0, SC.DEFAULT_NMU))
    return out + [("isotropic", (SC.TAB, flat))]


def exact(law_list, n, seed=4242):
    """{(name, R): (cos_n, mu, phi, s)} from the exact walk."""
    from core import scattering as SC
    from .. import walk
    ang = SC.pack([law for _, law in law_list])
    out = {}
    for m, (name, _) in enumerate(law_list):
        for j, R in enumerate(RADII):
            cn, mu, phi, s, _k = walk.sample_aniso(
                R, ang, m, n, seed=seed, first=(m * len(RADII) + j) * n)
            out[name, R] = (cn, mu, phi, s)
    return out, ang


def score(backend, law_list, ref, ang, seed=777):
    """One row per (law, R).  Whole balls (backend.sample_aniso, so an
    m2.1 model's exact first scatter is included), on random streams of
    their own (master seed `seed`, not the exact walk's)."""
    from core import rng
    rows = []
    for m, (name, _) in enumerate(law_list):
        for j, R in enumerate(RADII):
            e = ref[name, R]
            n = e[0].size
            st = rng.streams(np.uint64(seed), (m * len(RADII) + j) * n, n)
            d = backend.sample_aniso(np.full(n, R), np.full(n, m), ang, st,
                                     np.arange(n))
            ee = (e[0], e[1], e[2], e[3] / R)
            dd = (d[0], d[1], d[2], d[3] / R)
            noise = 1.36 * np.sqrt(2.0 / n)
            row = {"law": name, "R": R,
                   "z": {o: _z(a, b) for o, a, b in zip(OUTS, ee, dd)},
                   "ks_excess": {o: _ks(a, b) - noise
                                 for o, a, b in zip(OUTS, ee, dd)},
                   "weight": {}, "mean_exact": [float(a.mean()) for a in e],
                   "mean_model": [float(a.mean()) for a in d]}
            for kp in KAPPAS:
                we, wm = np.exp(-kp * e[3]), np.exp(-kp * d[3])
                row["weight"][kp] = (float(wm.mean() / we.mean() - 1),
                                     _z(we, wm))
            rows.append(row)
    return rows


def worst(rows, laws_in=None):
    sel = [r for r in rows if laws_in is None or r["law"] in laws_in]
    z = max(max(abs(v) for v in r["z"].values()) for r in sel)
    zw = max(max(abs(v[1]) for v in r["weight"].values()) for r in sel)
    ks = max(max(r["ks_excess"].values()) for r in sel)
    return max(z, zw), ks


def cost_per_draw(backend, batches=(1000, 10000, 100000), repeats=3):
    """Microseconds per whole ball (sample_aniso), P3 law, R log-uniform
    in [1, 30]; streams made outside the timer, as the driver has them."""
    from core import rng
    ang = _p3_ang()
    out = {}
    for b in batches:
        R = np.exp(np.random.default_rng(0).uniform(0, np.log(30), b))
        mats = np.zeros(b, np.int64)
        st = [rng.streams(np.uint64(99), r * b, b) for r in range(repeats + 1)]
        backend.sample_aniso(R, mats, ang, st[0], np.arange(b))
        best = np.inf
        for r in range(repeats):
            t = time.perf_counter()
            backend.sample_aniso(R, mats, ang, st[r + 1], np.arange(b))
            best = min(best, time.perf_counter() - t)
        out[b] = 1e6 * best / b
    return out


def _p3_ang():
    from core import scattering as SC
    return SC.pack([SC.law("legendre", np.array([1, 0.6, 0.3, 0.1]))])


def report(name, rows, cost, log=print):
    held = [n for n in {r["law"] for r in rows} if n != "isotropic"]
    zh, kh = worst(rows, held)
    zi, ki = worst(rows, ["isotropic"])
    log(f"\n== {name}: held-out laws worst |z| {zh:.1f}, KS excess "
        f"{kh:+.4f};  isotropic worst |z| {zi:.1f}, KS excess {ki:+.4f};  "
        f"us/draw " + " / ".join(f"{v:.2f}" for v in cost.values()))
    log(f"{'law':10s} {'R':>4s} | z: " + " ".join(f"{o:>6s}" for o in OUTS)
        + " | KS excess: " + " ".join(f"{o:>7s}" for o in OUTS)
        + " | weight err % (z) k=0.01, 0.1")
    for r in rows:
        w = "  ".join(f"{100 * e:+6.2f} ({z:+4.1f})"
                      for e, z in r["weight"].values())
        log(f"{r['law']:10s} {r['R']:4g} | "
            + "    " + " ".join(f"{r['z'][o]:+6.1f}" for o in OUTS)
            + " |            " + " ".join(f"{r['ks_excess'][o]:+7.4f}"
                                         for o in OUTS) + f" | {w}")


def smoke(backend, n=200_000, log=print):
    """Innovation's P3 reference point at R = 5 (model means)."""
    d = backend.sample_aniso(np.full(n, 5.0), np.zeros(n, np.int64),
                             _p3_ang())
    m = [float(a.mean()) for a in d]
    log(f"smoke P3 R=5  cos_n {m[0]:.3f} (0.382)  mu {m[1]:.3f} (0.778)  "
        f"phi {m[2]:.3f} (1.864)  s {m[3]:.2f} (8.91)")
    return m


def run(backends, n=100_000, log=print):
    """backends: {name: AnisoBackend}.  Returns {name: summary}."""
    law_list = laws()
    t = time.time()
    ref, ang = exact(law_list, n)
    log(f"exact walks: {len(law_list)} laws x {len(RADII)} radii x {n:,}, "
        f"{time.time() - t:.0f} s")
    out = {}
    for name, be in backends.items():
        rows = score(be, law_list, ref, ang)
        cost = cost_per_draw(be)
        report(name, rows, cost, log)
        sm = smoke(be, log=log)
        held = [x for x, _ in law_list if x != "isotropic"]
        out[name] = {"rows": rows, "cost": cost, "smoke": sm,
                     "worst_held": worst(rows, held),
                     "worst_iso": worst(rows, ["isotropic"]),
                     "describe": {k: v for k, v in be.describe().items()
                                  if k not in ("train",)}}
    return out
