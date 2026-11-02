"""v1.6.1: the general scorer (core/ext.py) on the cases its OpenMC check
(`python run.py gapcheck`) found wrong first.

Run:  python -m pytest -q tests   (from gmc3d/)

Each test is an identity or a sign that holds without OpenMC; the
agreement with OpenMC itself is in runs/2026-10-30/v1.6.1/.
"""
import numpy as np
import openmc

from checks import gapcheck as GC
from core import openmc_import, transport
from problems import common

RUN = dict(particles=20_000, batches=4, seed=1)


def _tally(name, filters, scores=("flux",)):
    t = openmc.Tally(name=name)
    t.filters, t.scores = list(filters), list(scores)
    return t


def _sphere(ss, sa, tallies):
    mats, m = common.one_group_materials("t161", {"a": (ss, sa)})
    s = openmc.Sphere(r=6.0, boundary_type="vacuum")
    c = openmc.Cell(name="c", fill=m["a"], region=-s)
    st = common.settings(common.point_source((0, 0, 0)))
    return openmc.Model(openmc.Geometry([c]), mats, st,
                        openmc.Tallies(tallies(c, m)))


def test_lattice_walls_count_as_crossings_for_cell_from():
    """OpenMC scores a CellFrom crossing tally at every lattice wall: a
    one-cell universe left through a wall into its copy is a crossing
    from that cell (current +w, the normal along the step)."""
    model = GC.lattice_crossings_model()
    t = transport.run(openmc_import.load(model), **RUN).tallies
    cur = t["cellfrom_lattice"]["mean"][:, 0]
    assert cur[0] > 0 and cur[1] > 0
    # cell-to-cell: every wall inside a checkerboard joins a to b (a->a and
    # b->b are only the leaks through the box, signed by its planes)
    cc = t["cell_crossings"]["mean"][:, 0].reshape(2, 2)
    assert cc[0, 1] > 0 and cc[1, 0] > 0
    assert abs(cc[0, 0]) < 0.01 * cc[0, 1]


def test_two_phi_bins_tell_the_two_phi_faces_apart():
    """With phi = [0, pi, 2 pi] both phi faces join the same two elements;
    the way a flight crosses must come from phi itself (both faces are
    crossed both ways, so all four phi slots score)."""
    def tallies(c, m):
        sph = openmc.SphericalMesh(r_grid=[0, 6], theta_grid=[0, np.pi],
                                   phi_grid=[0, np.pi, 2 * np.pi])
        return [_tally("f", [openmc.MeshSurfaceFilter(sph)], ("current",))]
    model = _sphere(0.4, 0.1, tallies)
    t = transport.run(openmc_import.load(model), **RUN).tallies
    m = t["f"]["mean"][:, 0].reshape(2, 3, 4)      # element, axis, slot
    assert np.all(m[:, 2, :] > 0)
    # what leaves element 0 through a phi face enters element 1
    assert np.isclose(m[0, 2, 0] + m[0, 2, 2], m[1, 2, 1] + m[1, 2, 3],
                      rtol=1e-12)


def test_mu_filter_carries_the_previous_histories_cosine():
    """OpenMC never resets the scatter cosine between histories, so a Mu
    filter bins an unscattered flight by the cosine the previous history
    ended with.  In a pure scatterer that is the cosine it leaked with,
    which leans forwards (a flight along the last one leaks sooner)."""
    def tallies(c, m):
        return [_tally("u", [openmc.CollisionFilter([0]),
                             openmc.MuFilter([-1, 0, 1])])]
    model = _sphere(0.5, 1e-6, tallies)
    t = transport.run(openmc_import.load(model), **RUN).tallies
    back, fwd = t["u"]["mean"][:, 0]
    assert fwd > 1.5 * back
