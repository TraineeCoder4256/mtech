"""v2.1.2: the interface v2 acceptance gate (checks/gate_v2.py) passes the
exact walk itself and fails a model that ignores the law."""
import numpy as np

from ball import laws as LW, walk
from checks import gate_v2
from core import rng, scattering as SC

LAWS = {k: v for k, v in LW.held_out().items() if k in ("p3", "histogram")}
FLAT = SC.pack([(SC.TAB, (np.full(3, 0.5), np.array([0.0, 0.5, 1.0])))])


class _Walk:
    """The exact walk dressed as a v2 model (other random numbers)."""
    name, info, aniso = "walk", {"interface": 2}, True

    def __init__(self, ignore_law=False):
        self.calls, self.ignore = 0, ignore_law

    def draw(self, R, g):
        self.calls += 1
        if self.ignore:
            ang = FLAT
        else:
            law = [l for l in LAWS.values()
                   if np.allclose(LW.moments(l), g[0])][0]
            ang = SC.pack([law])
        st = rng.streams(np.uint64(900 + self.calls), 0, R.size)
        return walk.walk_many_aniso(R, np.zeros(R.size, np.int64), ang, st,
                                    np.arange(R.size))[:4]


def test_gate_passes_the_exact_walk():
    out = gate_v2.run(_Walk(), n=20_000, laws=LAWS, log=lambda *a: None)
    assert out["pass"], out["failures"]


def test_gate_fails_a_model_that_ignores_the_law():
    out = gate_v2.run(_Walk(True), n=20_000, laws=LAWS, log=lambda *a: None)
    assert not out["pass"] and len(out["failures"]) > 5


def test_gate_refuses_another_interface():
    w = _Walk()
    w.info = {"interface": 1}
    out = gate_v2.run(w, n=1000, laws=LAWS, log=lambda *a: None)
    assert not out["manifest_ok"] and not out["pass"] and not out["rows"]
