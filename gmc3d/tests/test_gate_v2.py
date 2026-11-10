"""v2.1.2 (v2.1.4: whole balls via sample_aniso): the interface v2 acceptance gate (checks/gate_v2.py) passes the
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

    def sample_aniso(self, R, mats, ang, seeds, idx):
        self.calls += 1
        return walk.walk_many_aniso(R, mats, FLAT if self.ignore else ang,
                                    seeds, idx)[:4]


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
