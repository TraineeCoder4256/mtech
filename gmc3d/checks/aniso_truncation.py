"""How much of a scattering law does a ball's exit depend on (interface v2)?

Used by: run by hand (`python -m checks.aniso_truncation`), for the
interface v2 handoff.  Writes nothing but its log.

A v2 model sees a law only through its first L Legendre moments.  Two laws
with the same first L moments then look the same to it.  Here, for laws A
of different shapes, the exact walk is run with A and with B = another
positive law with EXACTLY A's first L moments: the maximum-entropy one,
f(mu) = exp(sum_{l<=L} lambda_l P_l(mu)) (smooth, as different from a
peaked or stepped A as a positive law with those moments can naturally
be), for L = 1, 2, 3, 5, 8, at R = 1, 3, 10, 30.  Both are tables of
401 points (so a peaked law is resolved); the printed dg_{L+1} is how
far B's next moment is from A's.

(A first version used for B the law OpenMC makes from A's first L
moments as Legendre data; its tabular conversion clips negative parts,
which moves g1 by up to 0.4 for peaked laws, so it measured clipping,
not truncation: runs/2026-11-10/interface-v2/truncation_clipped.log.)  For each
output (cos_n, mu, phi, s) it prints the difference of the means in
standard errors (z) and the two-sample KS distance D; the line "same law"
is A against A with other random numbers: the noise floor.
"""

import sys
import time

import numpy as np
from scipy import stats

from ball import laws as LW, walk
from core import scattering as SC

LS = (1, 2, 3, 5, 8)
RADII = (1.0, 3.0, 10.0, 30.0)
OUT = ("cos_n", "mu", "phi", "s")


NPTS = 401


def maxent(g, L, n_gauss=800):
    """The maximum-entropy density on [-1, 1] with E[P_l] = g[l],
    l = 1..L (g[0] = 1), at NPTS equally spaced points: minimise the
    convex dual log Z(lambda) - lambda . g (scipy, exact gradient and
    Hessian, log-sum-exp for stability)."""
    from scipy.optimize import minimize
    xs, ws = np.polynomial.legendre.leggauss(n_gauss)
    P = np.array([np.polynomial.legendre.legval(xs, np.eye(L + 1)[l])
                  for l in range(1, L + 1)])
    t = np.asarray(g[1:L + 1], float)

    def parts(lam):
        a = lam @ P
        amax = a.max()
        f = np.exp(a - amax) * ws
        z = f.sum()
        m = P @ f / z
        return amax + np.log(z), m, (P * f) @ P.T / z - np.outer(m, m)

    res = minimize(lambda l: parts(l)[0] - l @ t, np.zeros(L),
                   jac=lambda l: parts(l)[1] - t,
                   hess=lambda l: parts(l)[2], method="trust-exact",
                   options={"gtol": 1e-11, "maxiter": 2000})
    lam = res.x
    x = np.linspace(-1, 1, NPTS)
    Px = np.array([np.polynomial.legendre.legval(x, np.eye(L + 1)[l])
                   for l in range(1, L + 1)])
    a = lam @ Px
    return np.exp(a - a.max())


def test_laws():
    x = np.linspace(-1, 1, NPTS)
    return {
        "hg 0.3": LW.table(LW.hg_density(0.3, x)),
        "hg 0.6": LW.table(LW.hg_density(0.6, x)),
        "hg 0.9": LW.table(LW.hg_density(0.9, x)),
        "hg2 0.7x0.9+0.3x-0.5": LW.table(0.7 * LW.hg_density(0.9, x) +
                                         0.3 * LW.hg_density(-0.5, x)),
        "histogram fwd": SC.law("histogram", np.array([1, 1, 2, 3, 5, 8,
                                                       13, 20], float)),
        "clipped P3": SC.law("legendre", np.array([1, 0.9, 0.5, -0.4]),
                             tab_points=NPTS),
    }


def _stats(a, b):
    z, d = [], []
    for k in range(4):
        x, y = a[k], b[k]
        se = np.hypot(x.std() / np.sqrt(x.size), y.std() / np.sqrt(y.size))
        z.append((y.mean() - x.mean()) / se)
        d.append(stats.ks_2samp(x, y).statistic)
    return z, d


def run(n=100_000, log=print):
    t0 = time.perf_counter()
    rows = {}
    for name, A in test_laws().items():
        gA = SC.moments(*A, 9)
        log(f"\n== {name}: g1..g8 = " + " ".join(f"{v:+.3f}" for v in gA[1:]))
        for R in RADII:
            ang_a = SC.pack([A])
            a = walk.sample_aniso(R, ang_a, 0, n, seed=11)[:4]
            a2 = walk.sample_aniso(R, ang_a, 0, n, seed=12)[:4]
            z, d = _stats(a, a2)
            log(f"  R {R:4g}  same law      z " +
                " ".join(f"{o} {v:+6.2f}" for o, v in zip(OUT, z)) +
                "   KS D max {:.4f}".format(max(d)))
            for L in LS:
                B = LW.table(maxent(gA, L))
                gB = SC.moments(*B, 9)
                b = walk.sample_aniso(R, SC.pack([B]), 0, n, seed=12)[:4]
                z, d = _stats(a, b)
                rows[(name, R, L)] = (z, d)
                gA9 = SC.moments(*A, 9)
                log(f"  R {R:4g}  L {L}  dg1 {gB[1] - gA[1]:+.4f} dg{L + 1} "
                    f"{gB[L + 1] - gA9[L + 1]:+.3f}  z " +
                    " ".join(f"{o} {v:+6.2f}" for o, v in zip(OUT, z)) +
                    "   KS D max {:.4f}".format(max(d)))
    log(f"\n{n:,} walks per law and radius; {time.perf_counter() - t0:.0f} s")
    return rows


def rate(n=200_000, R=10.0, log=print):
    """Rows per second of data.make_aniso-like walks (training cost)."""
    laws, _ = LW.family(2000, seed=3)
    ang = SC.pack(laws)
    Rs = np.exp(np.random.default_rng(1).uniform(0, np.log(30), n))
    mats = np.random.default_rng(2).integers(0, len(laws), n)
    from core import rng
    st = rng.streams(np.uint64(5), 0, n)
    walk.walk_many_aniso(Rs[:100], mats[:100], ang, st, np.arange(100))
    t = time.perf_counter()
    _, _, _, _, k = walk.walk_many_aniso(Rs, mats, ang, st, np.arange(n))
    dt = time.perf_counter() - t
    st = rng.streams(np.uint64(5), 0, n)
    t = time.perf_counter()
    walk.walk_many(Rs, st, np.arange(n))
    dt1 = time.perf_counter() - t
    log(f"rows/s: anisotropic {n / dt:,.0f} (mean {k.mean():.1f} scatters "
        f"per walk), isotropic v1 {n / dt1:,.0f}; R log-uniform 1-30")
    return n / dt, n / dt1


if __name__ == "__main__":
    rate()
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
