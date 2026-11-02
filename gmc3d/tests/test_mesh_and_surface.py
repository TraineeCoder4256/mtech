"""v1.1: the 'profile' mesh rule and per-surface current tallies.

Run:  python -m pytest -q tests   (from gmc3d/)
"""
import numpy as np
import openmc
import pytest

from ball.oracle import Oracle
from checks import compare
from core import openmc_import, transport
from core.openmc_import import Unsupported
from problems import PROBLEMS


def _slab_with_currents(reflective_too=False):
    model = PROBLEMS["slab"]()
    surfs = model.geometry.get_all_surfaces().values()
    planes = [s for s in surfs if isinstance(s, openmc.XPlane)]
    if reflective_too:
        planes += [s for s in surfs if isinstance(s, openmc.YPlane)][:1]
    t = openmc.Tally(name="sides")
    t.filters = [openmc.SurfaceFilter(sorted(planes, key=lambda s: s.id))]
    t.scores = ["current"]
    model.tallies.append(t)
    return model, sorted(planes, key=lambda s: s.id)


def test_profile_rule_matches_plain_mc_depth_profile():
    """Full-size exact balls, 'profile' rule: the slab's 60-bin depth
    profile agrees with plain MC bin by bin (the old 'centre' rule gave
    chi2/dof in the hundreds here)."""
    pr = openmc_import.load(PROBLEMS["slab"]())
    # the same seed, so both start from the same source sites (v1.2 draws
    # them separately from the transport): the comparison is about how the
    # flux is laid down, not about where particles were born.  The 60 bins
    # are strongly correlated, so a 2-sigma swing of the whole curve
    # between independent runs alone lifts chi2/dof to about 4.
    a = transport.run(pr, 20000, 10, seed=2)
    b = transport.run(pr, 20000, 10, seed=2, backend=Oracle(), r_star=1.0)
    assert b.counters["balls"] > 10000
    ta, tb = a.tallies["depth"], b.tallies["depth"]
    z, ok = compare.z_scores(tb["mean"][:, 0], tb["se"][:, 0],
                             ta["mean"][:, 0], ta["se"][:, 0])
    assert ok.sum() == 60
    assert np.mean(z[ok] ** 2) < 2.0
    assert np.max(np.abs(z[ok])) < 5.0


def test_surface_currents_add_up_to_leakage():
    """Each vacuum plane's current is its signed leakage: + out through
    x = 60 (towards its positive side), - out through x = 0."""
    model, planes = _slab_with_currents()
    pr = openmc_import.load(model)
    for be in (None, Oracle()):
        r = transport.run(pr, 20000, 5, seed=4, backend=be, r_star=2.0)
        cur = dict(zip(r.tallies["sides"]["bin_ids"],
                       r.tallies["sides"]["mean"][:, 0]))
        left, right = cur[planes[0].id], cur[planes[1].id]
        assert left < 0 < right
        assert abs((right - left) - r.leakage) < 1e-12 * r.leakage


def test_current_on_a_reflective_surface_is_exactly_zero():
    """OpenMC scores a reflection +w and then -w (measured, v1.5), so the
    current through a reflective wall is exactly 0; before v1.5 such a
    tally was refused."""
    model, planes = _slab_with_currents(reflective_too=True)
    r = transport.run(openmc_import.load(model), 5000, 4, seed=4)
    cur = dict(zip(r.tallies["sides"]["bin_ids"],
                   r.tallies["sides"]["mean"][:, 0]))
    assert cur[planes[2].id] == 0.0
    assert cur[planes[0].id] < 0 < cur[planes[1].id]
