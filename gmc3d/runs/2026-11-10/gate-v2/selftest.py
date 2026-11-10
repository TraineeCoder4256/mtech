import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from ball import walk
from checks import gate_v2
class WalkAsModel:
    """The exact walk dressed as a v2 model, called as the gate calls a
    model since v2.1.4 (whole balls, its own streams): the gate must pass
    it."""
    name, info, aniso = "walk_as_model", {"interface": 2}, True
    def sample_aniso(self, R, mats, ang, seeds, idx):
        return walk.walk_many_aniso(R, mats, ang, seeds, idx)[:4]
out = gate_v2.run(WalkAsModel(), n=int(sys.argv[1]))
