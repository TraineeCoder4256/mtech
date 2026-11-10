"""Anisotropic scattering laws (v2.0): the cosine of a scatter, as OpenMC
samples it in multigroup mode.

Used by: openmc_import.py (builds each material's law from the library),
transport.py (samples the cosine at every scatter), detectors.py (the
law's density towards a detector), ext.py (Legendre moments for the
Legendre and scatter-cosine spherical-harmonics filters).

WHAT OPENMC DOES (read from OpenMC 0.16.0's src/scattdata.cpp, mgxs.cpp,
settings.cpp, math_functions.cpp; one group, so the outgoing group is
always the same one):

  Legendre data   coefficients a_0..a_L of the scatter matrix (truncated
                  to settings.max_order).  The density is

                      f(mu) = sum_l (l + 1/2) (a_l / a_0) P_l(mu)

                  By default (settings.tabular_legendre enable = True)
                  OpenMC does NOT sample this f: it evaluates f at n_mu
                  equally spaced points on [-1, 1] (n_mu = num_points, else
                  33, or 2 for P0 data), sets negative values to 0,
                  renormalises with the trapezoid rule, and samples the
                  resulting piecewise-LINEAR density exactly
                  (ScattDataTabular::sample).  With enable = False it
                  samples f itself by rejection (mu uniform, kept with
                  probability f / M where f > 0), so negative parts are
                  clipped too.
  Tabular data    f given at `order` equally spaced points; same as the
                  converted case (negatives set to 0, trapezoid norm).
  Histogram data  `order` equal bins on [-1, 1]; the bin is chosen by its
                  share, mu uniform inside it.

The tables here reproduce those densities exactly, so the scattering a
particle sees is OpenMC's, not just the same Legendre moments.

PACKING.  Every material's law is one row of a float array `ang`
(transport.Physics.ang):

    ang[m, 0]  kind: ISO, TAB (piecewise linear), HIST, LEG (rejection)
    ang[m, 1]  n: points (TAB), bins (HIST) or coefficients (LEG)
    ang[m, 2]  M, the rejection bound (LEG only)
    ang[m, 3]  Z, the integral of max(f, 0) (LEG only: pdf = max(f, 0)/Z)
    ang[m, 4:4+n]        f at the points / in the bins / a_l / a_0
    ang[m, 4+n:4+2n]     the CDF at the points (TAB) or bin ends (HIST)

An isotropic material is kind ISO, sampled exactly as before v2 (the same
random numbers), so isotropic problems give bit-identical answers.
"""

import numpy as np
from numba import njit

from . import rng

ISO, TAB, HIST, LEG = 0, 1, 2, 3
DEFAULT_NMU = 33            # OpenMC's constants.h
HEAD = 4


# ---------------------------------------------------------------- building
def legendre_density(coef, mu):
    """f(mu) = sum_l (l + 1/2) c_l P_l(mu) (OpenMC's evaluate_legendre)."""
    return np.polynomial.legendre.legval(
        np.asarray(mu, float), (np.arange(len(coef)) + 0.5) * coef)


def law(fmt, data, tab_enable=True, tab_points=None, max_order=None):
    """(kind, values) for one material's one-group scattering.

    fmt is the library's scatter_format ('legendre', 'histogram',
    'tabular'); data the 1D array of its scatter matrix entry [0, 0, :].
    tab_enable / tab_points / max_order are OpenMC's settings.  Returns
    kind ISO for a law that is isotropic in every way OpenMC samples it.
    """
    data = np.asarray(data, float).ravel()
    if fmt == "legendre":
        if max_order is not None:
            data = data[:int(max_order) + 1]
        if data[0] <= 0.0:
            return ISO, None                 # no scattering at all
        c = data / data[0]
        if c.size == 1 or not np.any(c[1:] != 0.0):
            return ISO, None
        if not tab_enable:
            # OpenMC's bound: the largest f on its 1001-point scan, +10%
            # (the scan's own point set: -1, then -1 + (k - 1) dmu, 1)
            n = 1001
            dmu = 2.0 / (n - 1)
            pts = np.r_[-1.0, -1.0 + np.arange(n - 2) * dmu, 1.0]
            m = max(0.0, float(np.max(legendre_density(c, pts)))) * 1.1
            xs, ws = np.polynomial.legendre.leggauss(400)
            z = float(np.sum(ws * np.maximum(legendre_density(c, xs), 0.0)))
            return LEG, (c, m, z)
        n = int(tab_points) if tab_points else DEFAULT_NMU
        return _tabular(legendre_density(c, np.linspace(-1.0, 1.0, n)))
    if fmt == "tabular":
        return _tabular(data)
    if fmt == "histogram":
        if np.any(data < 0.0):
            raise ValueError("histogram scattering with negative bins")
        if np.allclose(data, data[0]):
            return ISO, None
        dmu = 2.0 / data.size
        cdf = np.cumsum(data * dmu)
        f, cdf = data / cdf[-1], cdf / cdf[-1]
        return HIST, (f, cdf)
    raise ValueError(f"scatter_format {fmt!r}")


def _tabular(f):
    f = np.maximum(np.asarray(f, float), 0.0)
    if np.allclose(f, f[0]):
        return ISO, None
    dmu = 2.0 / (f.size - 1)
    cdf = np.r_[0.0, np.cumsum(0.5 * dmu * (f[:-1] + f[1:]))]
    return TAB, (f / cdf[-1], cdf / cdf[-1])


def pack(laws):
    """The `ang` array for a list of (kind, values), one per material."""
    width = HEAD
    for kind, v in laws:
        if kind in (TAB, HIST):
            width = max(width, HEAD + 2 * v[0].size)
        elif kind == LEG:
            width = max(width, HEAD + v[0].size)
    ang = np.zeros((max(1, len(laws)), width))
    for m, (kind, v) in enumerate(laws):
        ang[m, 0] = kind
        if kind in (TAB, HIST):
            n = v[0].size
            ang[m, 1] = n
            ang[m, HEAD:HEAD + n] = v[0]
            ang[m, HEAD + n:HEAD + 2 * n] = v[1]
        elif kind == LEG:
            ang[m, 1] = v[0].size
            ang[m, 2] = v[1]
            ang[m, 3] = v[2]
            ang[m, HEAD:HEAD + v[0].size] = v[0]
    return ang


def density(kind, v, mu):
    """The density of the scattering cosine OpenMC samples (python side,
    for tests and moments)."""
    mu = np.asarray(mu, float)
    if kind == ISO:
        return np.full_like(mu, 0.5)
    if kind == TAB:
        f = v[0]
        return np.interp(mu, np.linspace(-1.0, 1.0, f.size), f)
    if kind == HIST:
        f = v[0]
        k = np.clip(np.floor((mu + 1.0) / (2.0 / f.size)).astype(int), 0,
                    f.size - 1)
        return f[k]
    return np.maximum(legendre_density(v[0], mu), 0.0) / v[2]


def moments(kind, v, order):
    """E[P_l(mu)], l = 0..order, under the density OpenMC samples (exact
    to rounding: Gauss-Legendre on each linear piece or bin)."""
    out = np.zeros(order + 1)
    if kind == ISO:
        out[0] = 1.0
        return out
    if kind in (TAB, HIST):
        n = v[0].size
        edges = np.linspace(-1.0, 1.0, n if kind == TAB else n + 1)
    else:
        edges = np.linspace(-1.0, 1.0, 65)
    xs, ws = np.polynomial.legendre.leggauss(max(16, order + 4))
    for a, b in zip(edges[:-1], edges[1:]):
        x = 0.5 * (b - a) * xs + 0.5 * (a + b)
        fx = density(kind, v, x) * 0.5 * (b - a) * ws
        for l in range(order + 1):
            out[l] += np.sum(fx * np.polynomial.legendre.legval(
                x, np.eye(order + 1)[l]))
    return out


def mean_cosine(kind, v):
    return float(moments(kind, v, 1)[1])


# ---------------------------------------------------------------- sampling
@njit(cache=True)
def sample_mu(ang, m, state, i):
    """A scattering cosine from material m's law (not for kind ISO)."""
    kind = int(ang[m, 0])
    n = int(ang[m, 1])
    if kind == TAB:
        xi = rng.rand(state, i)
        cdf0 = HEAD + n
        dmu = 2.0 / (n - 1)
        k = 0
        c_k = ang[m, cdf0]
        while k < n - 1:                     # ScattDataTabular::sample
            c_k1 = ang[m, cdf0 + k + 1]
            if xi < c_k1:
                break
            c_k = c_k1
            k += 1
        if k > n - 2:
            k = n - 2
        p0, p1 = ang[m, HEAD + k], ang[m, HEAD + k + 1]
        mu0 = -1.0 + k * dmu
        if p0 == p1:
            mu = mu0 + (xi - c_k) / p0
        else:
            frac = (p1 - p0) / dmu
            mu = mu0 + (np.sqrt(max(0.0, p0 * p0 + 2.0 * frac *
                                    (xi - c_k))) - p0) / frac
    elif kind == HIST:
        xi = rng.rand(state, i)
        cdf0 = HEAD + n
        k = 0
        while k < n - 1 and ang[m, cdf0 + k] <= xi:
            k += 1
        mu = -1.0 + (k + rng.rand(state, i)) * (2.0 / n)
    else:                                    # LEG: rejection
        bound = ang[m, 2]
        while True:
            mu = 2.0 * rng.rand(state, i) - 1.0
            f = _leg_f(ang, m, n, mu)
            if f > 0.0 and rng.rand(state, i) * bound <= f:
                break
    if mu < -1.0:
        mu = -1.0
    elif mu > 1.0:
        mu = 1.0
    return mu


@njit(cache=True)
def _leg_f(ang, m, n, mu):
    p_prev, p = 1.0, mu
    f = 0.5 * ang[m, HEAD]
    if n > 1:
        f += 1.5 * ang[m, HEAD + 1] * mu
    for l in range(2, n):
        p_next = ((2 * l - 1) * mu * p - (l - 1) * p_prev) / l
        p_prev, p = p, p_next
        f += (l + 0.5) * ang[m, HEAD + l] * p
    return f


@njit(cache=True)
def pdf(ang, m, mu):
    """The density of the cosine at mu (1/2 for isotropic): what a point
    detector's next-event estimate weights a scatter by."""
    kind = int(ang[m, 0])
    n = int(ang[m, 1])
    if kind == ISO:
        return 0.5
    if kind == TAB:
        dmu = 2.0 / (n - 1)
        k = int((mu + 1.0) / dmu)
        if k > n - 2:
            k = n - 2
        if k < 0:
            k = 0
        r = (mu - (-1.0 + k * dmu)) / dmu
        return (1.0 - r) * ang[m, HEAD + k] + r * ang[m, HEAD + k + 1]
    if kind == HIST:
        k = int((mu + 1.0) / (2.0 / n))
        if k > n - 1:
            k = n - 1
        if k < 0:
            k = 0
        return ang[m, HEAD + k]
    return max(0.0, _leg_f(ang, m, n, mu)) / ang[m, 3]
