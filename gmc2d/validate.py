#!/usr/bin/env python3
"""Validate the Monte Carlo baseline against EXACT results.  Run: python validate.py

The baseline is the reference everything else is measured against, so it has
to be checked against things that are known exactly -- not against another
code.  A cross-code comparison only tells you two codes agree; an analytic
identity tells you the code is right.

Three identities, chosen because between them they exercise every part of
the solver:

  1. DIRAC / CAUCHY MEAN-CHORD INVARIANCE.  For a convex body in a
     non-absorbing medium, the mean TOTAL path length of a particle entering
     with isotropic incident flux equals the mean chord 4V/S -- and is
     completely independent of how much the particle scatters.  For an
     infinite prism of cross-section W x H that is 2WH/(W+H).
     Exercises: free-path sampling, isotropic scattering, boundary
     detection, path accumulation.  A bug in any of them breaks it.

  2. UNCOLLIDED PROBABILITY.  A particle crosses without scattering with
     probability exp(-d), d being the straight-line chord in mean free paths.
     Exercises: the exponential free-path law and the chord geometry.

  3. PARTICLE BALANCE.  In an optically thick pure absorber nothing escapes,
     so the absorbed weight per source particle is exactly 1.
     Exercises: the full-domain mesh traversal, the track-length estimator,
     implicit capture, and the tally normalisation.

Results are reported as z = (measured - exact) / standard error.  |z| < 3
is a pass.  Note the standard error in test 1 is computed over independent
ENTRY CONDITIONS, not over particles: the particles sharing a condition are
correlated, and treating them as independent understates the error ~3x.
"""
import numpy as np

import mc

# ---- settings ----------------------------------------------------------
SHAPES = [(1., 1.), (3., 3.), (10., 10.), (4., 1.), (1., 4.), (8., 2.)]
N_COND, PER_COND = 40000, 24        # test 1
N_UNCOLLIDED = 2_000_000            # test 2
N_BALANCE = 200000                  # test 3
SEED = 11


def cosine_directions(n, rng):
    """Isotropic incident flux on a face: direction density prop. to Omega_x.
    sqrt(u) is the inverse CDF of that cosine law."""
    ox = np.sqrt(rng.random(n))
    r = np.sqrt(np.maximum(0.0, 1.0 - ox ** 2))
    phi = 2.0 * np.pi * rng.random(n)
    return ox, r * np.cos(phi)


def test_mean_chord(rng):
    print("=" * 76)
    print("1. DIRAC MEAN-CHORD INVARIANCE      exact: mean path = 2WH/(W+H)")
    print("=" * 76)
    print(f"{'W':>5} {'H':>5} {'exact':>8} {'measured':>10} {'rel.err':>9} "
          f"{'z':>7}  verdict")
    ok = True
    for W, H in SHAPES:
        means = np.empty(N_COND)
        xi = rng.random(N_COND)
        ox, oy = cosine_directions(N_COND, rng)
        # entries must cover the whole surface, weighted by face length; a
        # horizontal-face entry into W x H is a left-face entry into H x W
        vertical = rng.random(N_COND) < H / (W + H)
        for j in range(N_COND):
            a, b = (W, H) if vertical[j] else (H, W)
            means[j] = mc.sample_cell(
                PER_COND, a, b, xi=xi[j],
                direction=(max(ox[j], 1e-6), oy[j]), n_blocks=1,
                seed=int(rng.integers(1, 2 ** 31 - 1)))["s"].mean()
        exact = 2 * W * H / (W + H)
        se = means.std(ddof=1) / np.sqrt(N_COND)     # clustered, see docstring
        z = (means.mean() - exact) / se
        ok &= abs(z) < 3
        print(f"{W:>5.1f} {H:>5.1f} {exact:>8.4f} {means.mean():>10.4f} "
              f"{abs(means.mean()-exact)/exact*100:>8.2f}% {z:>+7.2f}  "
              f"{'PASS' if abs(z) < 3 else 'FAIL'}")
    return ok


def test_uncollided():
    print("\n" + "=" * 76)
    print("2. UNCOLLIDED FRACTION              exact: P(k=0) = exp(-chord)")
    print("=" * 76)
    print(f"{'W':>5} {'H':>5} {'chord':>7} {'exact':>10} {'measured':>10} "
          f"{'z':>7}  verdict")
    cases = [(1., 1., .5, .8, .1), (2., 2., .3, .6, -.2), (4., 1., .5, .9, 0.),
             (.5, .5, .5, .7, .3), (3., 1.5, .25, .5, .4), (6., 6., .5, .85, .05)]
    ok = True
    for W, H, xi, ox, oy in cases:
        out = mc.sample_cell(N_UNCOLLIDED, W, H, xi=xi, direction=(ox, oy),
                             n_blocks=8, seed=19)
        ty = (H - xi * H) / oy if oy > 0 else (-xi * H / oy if oy < 0 else np.inf)
        chord = min(W / ox, ty)
        exact = np.exp(-chord)
        meas = (out["k"] == 0).mean()
        z = (meas - exact) / np.sqrt(exact * (1 - exact) / N_UNCOLLIDED)
        ok &= abs(z) < 3
        print(f"{W:>5.1f} {H:>5.1f} {chord:>7.4f} {exact:>10.6f} "
              f"{meas:>10.6f} {z:>+7.2f}  {'PASS' if abs(z) < 3 else 'FAIL'}")
    return ok


def test_balance():
    print("\n" + "=" * 76)
    print("3. PARTICLE BALANCE                 exact: absorbed / source = 1")
    print("=" * 76)
    print(f"{'sigma_a':>8} {'domain (mfp)':>13} {'absorbed/source':>17} "
          f"{'error':>9}  verdict")
    ok = True
    for sa in [2.0, 5.0, 20.0]:
        n = 112
        prob = {"L": 7.0, "n": n, "pitch": 1.0, "per_cm": n // 7,
                "sig_s": np.zeros((n, n)), "sig_a": np.full((n, n), sa),
                "source": (3.0, 3.0, 4.0, 4.0), "source_cell": (3, 3)}
        phi, _ = mc.solve(prob, N_BALANCE, seed=3)
        dx = prob["L"] / n
        absorbed = (prob["sig_a"] * phi).sum() * dx * dx
        err = abs(absorbed - 1.0)
        ok &= err < 1e-3
        print(f"{sa:>8.1f} {7.0*sa:>13.0f} {absorbed:>17.6f} "
              f"{err*100:>8.3f}%  {'PASS' if err < 1e-3 else 'FAIL'}")
    return ok


def main():
    rng = np.random.default_rng(SEED)
    ok = test_mean_chord(rng) & test_uncollided() & test_balance()
    print("\n" + "=" * 76)
    print("ALL TESTS PASSED" if ok else "*** SOME TESTS FAILED ***")
    print("=" * 76)


if __name__ == "__main__":
    main()
