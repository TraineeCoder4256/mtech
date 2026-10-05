"""Point location against OpenMC's own (SRS S1-GEO-3, S1-GEO-4).

Random points in every test problem are located by our Numba descent and by
openmc.Geometry.find; the deepest cell must agree everywhere.  The problems
cover unions, complements, cones, void cells, 2D and 3D lattices, an outer
universe, and a translated + rotated universe.
"""
import numpy as np
import openmc
import pytest

from core import geometry as G
from core import openmc_import
from problems import PROBLEMS

BOXES = {"sphere": [(-10, 10)] * 3, "slab": [(0, 60), (-100, 100), (-100, 100)],
         "lattice3d": [(0, 7), (0, 7), (-0.5, 0.5)],
         "curved": [(-35, 35)] * 3, "nested": [(-10, 10)] * 3,
         "cask": [(-62, 100), (-40, 40), (-40, 40)]}     # inside r = 120


def random_points(name, n, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(BOXES[name]).T
    return rng.uniform(lo, hi, (n, 3))


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_locate_matches_openmc(name):
    model = PROBLEMS[name]()
    prob = openmc_import.load(model)
    pts = random_points(name, 3000)
    cells, mats = G.locate_many(prob.geom, pts)
    for p, c in zip(pts, cells):
        path = model.geometry.find(p.copy())
        found = [x for x in path if isinstance(x, openmc.Cell)]
        ref = found[-1].id if found else None
        ours = int(prob.cell_ids[c]) if c >= 0 else None
        assert ours == ref, f"{name}: point {p} -> ours {ours}, OpenMC {ref}"
