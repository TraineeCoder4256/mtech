"""Problems whose answers are known exactly (SRS S1-VAL-2).

Used by: checks/validate.py, tests/.

    pure absorber      sphere of radius a, sigma_s = 0, point source at the
                       centre: every particle flies straight out, so
                       leakage = exp(-sigma_a a) and the total flux is
                       (1 - exp(-sigma_a a)) / sigma_a.  With continuous
                       absorption the estimate has ZERO variance, so the
                       match must be to rounding.
    infinite medium    reflective cube: nothing leaks, so every particle is
                       absorbed: absorption = 1 and flux = 1 / sigma_a.
    conservation       any problem without fission: leakage + absorption = 1.

Each check returns (name, value, expected, standard error, z or rel err).
"""

import numpy as np

from core import openmc_import, transport
from problems import PROBLEMS


def pure_absorber(backend=None, r_star=3.0, particles=20_000, batches=10):
    a, sa = 10.0, 0.1
    pr = openmc_import.load(PROBLEMS["sphere"](radius=a, sig_s=0.0,
                                               sig_a=sa))
    r = transport.run(pr, particles, batches, backend=backend,
                      r_star=r_star)
    flux = r.tallies["total"]["mean"][0, 0]
    return [("pure absorber leakage", r.leakage, np.exp(-sa * a),
             r.leakage_se),
            ("pure absorber flux", flux, (1 - np.exp(-sa * a)) / sa,
             r.tallies["total"]["se"][0, 0])]


def infinite_medium(backend=None, r_star=3.0, particles=20_000,
                    batches=10):
    ss, sa = 1.0, 0.1
    pr = openmc_import.load(PROBLEMS["sphere"](radius=5.0, sig_s=ss,
                                               sig_a=sa, box=True))
    r = transport.run(pr, particles, batches, backend=backend,
                      r_star=r_star, mesh_rule="centre")
    t = r.tallies["total"]
    return [("infinite medium flux", t["mean"][0, 0], 1 / sa, t["se"][0, 0]),
            ("infinite medium absorption", t["mean"][0, 1], 1.0,
             t["se"][0, 1]),
            ("infinite medium leakage", r.leakage, 0.0, r.leakage_se)]


def conservation(results):
    """leakage + total absorption for a Results with a 'total' tally."""
    t = results.tallies["total"]
    se = np.hypot(results.leakage_se, t["se"][0, 1])
    return ("conservation", results.leakage + t["mean"][0, 1], 1.0, se)


def report(rows):
    lines = []
    for name, val, exp, se in rows:
        if se > 0:
            tag = f"z {(val - exp) / se:+.2f}"
        else:
            tag = f"rel err {abs(val - exp) / max(abs(exp), 1e-300):.1e}"
        lines.append(f"  {name:28} {val:.8g}   expected {exp:.8g}   "
                     f"se {se:.2g}   {tag}")
    return "\n".join(lines)
