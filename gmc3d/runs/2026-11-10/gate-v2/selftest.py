import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
import numpy as np
from ball import walk
from core import rng, scattering as SC
from checks import gate_v2
from ball import laws as LW
class WalkAsModel:
    """The exact walk dressed as a v2 model (other random numbers): the
    gate must pass it."""
    name, info, aniso = "walk_as_model", {"interface": 2}, True
    def __init__(self): self.calls = 0
    def draw(self, R, g):
        self.calls += 1
        law = [l for l in LW.held_out().values() if np.allclose(LW.moments(l), g[0])][0]
        st = rng.streams(np.uint64(900 + self.calls), 0, R.size)
        return walk.walk_many_aniso(R, np.zeros(R.size, np.int64), SC.pack([law]), st, np.arange(R.size))[:4]
out = gate_v2.run(WalkAsModel(), n=int(sys.argv[1]))
