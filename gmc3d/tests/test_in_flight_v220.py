"""v2.2.0: several batches in flight, and overlapped backend calls, give
the same numbers bit for bit as one batch at a time with the exact walk
(which uses each particle's own random stream).  The full check over six
problems (detectors, anisotropic balls, uneven chunk splits) is
runs/2026-11-16/in-flight/regress.py."""
import numpy as np

from ball.oracle import Oracle
from core import openmc_import, transport
from problems import PROBLEMS


class Deferred(Oracle):
    def submit(self, R, seeds, idx):
        return self.sample(R, seeds, idx)

    def collect(self, ticket):
        return ticket


def _same(a, b):
    assert a.leakage == b.leakage and a.leakage_se == b.leakage_se
    for k in a.tallies:
        assert np.array_equal(a.tallies[k]["mean"], b.tallies[k]["mean"])
        assert np.array_equal(a.tallies[k]["se"], b.tallies[k]["se"])


def test_in_flight_and_overlap_bit_identical():
    pr = openmc_import.load(PROBLEMS["cask"]())
    kw = dict(seed=4, r_star=2.0)
    ref = transport.run(pr, 1000, 5, backend=Oracle(), in_flight=1, **kw)
    _same(ref, transport.run(pr, 1000, 5, backend=Oracle(), in_flight=3,
                             **kw))
    r = transport.run(pr, 1000, 5, backend=Deferred(), in_flight=2,
                      overlap=True, **kw)
    _same(ref, r)
    assert r.settings["overlap"] and r.settings["in_flight"] == 2


def test_bounds_match_chunk_sizes():
    from core import tallies
    lo = transport._bounds(1000, 32, 3)
    sizes = np.diff(lo).reshape(3, 32)
    for q in range(3):
        assert np.array_equal(sizes[q], tallies.chunk_sizes(1000, 32))
    assert lo[0] == 0 and lo[-1] == 3000
