"""Scattering laws for training anisotropic ball models (interface v2).

Used by: data.py (make_aniso draws one law per row from a pool made
here), checks/aniso_truncation.py.
Uses: core/scattering.py (laws as OpenMC samples them, their moments).

A law is (kind, values) as core/scattering.law returns it, always one
OpenMC can be given and samples that way: a 33-point table (what OpenMC
makes of Legendre data by default) or a histogram.  What a model sees of
a law is its Legendre moments g_l = E[P_l(mu)], l = 1..N_MOMENTS, of the
density OpenMC samples (interface.ANISO_INPUTS).

THE TRAINING FAMILY (`family`), agreed with Development 10 Nov 2026: a
mix of shapes, each kept only if its mean cosine lies in
[G_MIN, G_MAX] = [-0.5, 0.95]:

    hg        Henyey-Greenstein, g uniform in [G_MIN, G_MAX]
    hg2       two Henyey-Greenstein lobes, one forward and one backward,
              mixed with a random weight (peaked laws with a back-lobe)
    legendre  P1 to P5 data: g_1 uniform, the higher moments g_1^l times
              a random factor in [0.5, 1.5], then OpenMC's tabular
              conversion (negative parts set to 0)
    histogram 4 to 16 equal bins, log-normal heights with a random tilt

THE HELD-OUT SET (`held_out`): the laws of the test problems
(checks/aniso.py).  None of them is a member of the family by
construction (the family is continuous; these are specific laws), and
models are judged on them as unseen conditions.
"""

import numpy as np

from core import scattering as SC

N_MOMENTS = 12           # see runs/2026-11-10/interface-v2/README.md
G_MIN, G_MAX = -0.5, 0.95
KINDS = ("hg", "hg2", "legendre", "histogram")


def hg_density(g, mu):
    """Henyey-Greenstein density of the cosine (integral 1 on [-1, 1])."""
    return 0.5 * (1 - g * g) / (1 + g * g - 2 * g * mu) ** 1.5


def table(f_at_points):
    """A density given at 33 equally spaced points, as OpenMC's 'tabular'
    data: negatives set to 0, renormalised (scattering._tabular)."""
    return SC.law("tabular", np.asarray(f_at_points, float))


def _one(kind, gen):
    x = np.linspace(-1.0, 1.0, SC.DEFAULT_NMU)
    if kind == "hg":
        return table(hg_density(gen.uniform(G_MIN, G_MAX), x))
    if kind == "hg2":
        gf, gb = gen.uniform(0.3, 0.97), gen.uniform(-0.9, -0.2)
        a = gen.uniform(0.5, 0.95)
        return table(a * hg_density(gf, x) + (1 - a) * hg_density(gb, x))
    if kind == "legendre":
        order = gen.integers(1, 6)
        g1 = gen.uniform(G_MIN, G_MAX)
        c = np.r_[1.0, [g1 ** l * gen.uniform(0.5, 1.5)
                        for l in range(1, order + 1)]]
        return SC.law("legendre", c)
    nb = gen.integers(4, 17)
    t = np.linspace(-1, 1, nb)
    h = np.exp(gen.normal(0, 0.7, nb) + gen.uniform(-4, 4) * t)
    return SC.law("histogram", h)


def family(n, seed=1):
    """n laws of the training family: list of (kind, values), and their
    moments (n, N_MOMENTS)."""
    gen = np.random.default_rng([seed, 0xA11])
    laws, g = [], []
    while len(laws) < n:
        law = _one(KINDS[gen.integers(len(KINDS))], gen)
        if law[0] == SC.ISO:
            continue
        m = SC.moments(*law, N_MOMENTS)[1:]
        if G_MIN <= m[0] <= G_MAX:
            laws.append(law)
            g.append(m)
    return laws, np.array(g)


def held_out():
    """{name: (kind, values)}: the test problems' laws (checks/aniso.py),
    for judging models on conditions they never saw."""
    from checks import aniso as A
    out = {}
    for name, law in (("p1", A.P1), ("p3", A.P3), ("clipped", A.CLIPPED),
                      ("backward", A.BACKWARD), ("p5_order2", A.P5[:3]),
                      ("histogram", A.HIST), ("tabular", A.TABULAR)):
        if isinstance(law, tuple):
            out[name] = SC.law(law[0], np.asarray(law[1], float))
        else:
            out[name] = SC.law("legendre", np.asarray(law, float))
    return out


def moments(law):
    """The model input for one law: g_1 .. g_N_MOMENTS."""
    return SC.moments(*law, N_MOMENTS)[1:]
