"""v1.3: tallies with several filters, cells at any level, partial currents.

Run:  python -m pytest -q tests   (from gmc3d/)

Most checks here are identities that hold history by history (a filled
cell's flux is the sum of what is inside it; the from-cell bins of a
current add up to the net current), so they hold to rounding, not to
within an error bar.  Agreement with OpenMC itself is checked by
`python run.py validate` on the nested and slab problems.
"""
import numpy as np
import openmc
import pytest

from core import openmc_import, transport
from core.openmc_import import Unsupported
from problems import PROBLEMS, common

RUN = dict(particles=20_000, batches=4, seed=1)


def _nested():
    model = PROBLEMS["nested"]()
    cells = {c.name: c for c in model.geometry.get_all_cells().values()}
    mats = {m.name: m for m in model.materials}
    return model, cells, mats


def _tally(name, filters, scores=("flux",)):
    t = openmc.Tally(name=name)
    t.filters, t.scores = list(filters), list(scores)
    return t


def _run(model):
    return transport.run(openmc_import.load(model), **RUN).tallies


def test_filled_cells_score_what_is_inside_them():
    model, c, _ = _nested()
    model.tallies = openmc.Tallies([
        _tally("total", []),
        _tally("outer", [openmc.CellFilter([c["core"], c["reflector"]])]),
        _tally("holder", [openmc.CellFilter([c["holder"]])]),
        _tally("inside", [openmc.CellFilter([c["tilted-pin"],
                                             c["tilted-mod"]])]),
        # a filled cell and a cell inside it, in one filter: both score
        _tally("both", [openmc.CellFilter([c["holder"], c["tilted-pin"]])]),
    ])
    t = _run(model)
    f = lambda n: t[n]["mean"][:, 0]
    assert f("outer").sum() == pytest.approx(f("total")[0], rel=1e-12)
    assert f("holder")[0] == pytest.approx(f("inside").sum(), rel=1e-12)
    assert f("holder")[0] > 0
    assert f("both")[0] == pytest.approx(f("holder")[0], rel=1e-12)
    assert f("both")[1] == pytest.approx(f("inside")[0], rel=1e-12)


def test_mesh_x_material_adds_up_to_each():
    model, _, m = _nested()
    mesh = common.mesh_filter((-10, -10, -10), (10, 10, 10), (5, 5, 5))
    mf = openmc.MaterialFilter(list(m.values()))
    model.tallies = openmc.Tallies([
        _tally("mat", [mf], ("flux", "absorption")),
        _tally("mesh", [mesh], ("flux", "absorption")),
        _tally("mesh_mat", [mesh, mf], ("flux", "absorption")),
        _tally("mat_mesh", [mf, mesh], ("flux", "absorption")),
    ])
    t = _run(model)
    mm = t["mesh_mat"]["mean"].reshape(125, 3, -1)      # mesh slowest
    xm = t["mat_mesh"]["mean"].reshape(3, 125, -1)      # material slowest
    assert np.allclose(mm.sum(axis=0), t["mat"]["mean"], rtol=1e-9)
    assert np.allclose(mm.sum(axis=1), t["mesh"]["mean"], rtol=1e-9,
                       atol=1e-15)
    assert np.allclose(mm, xm.transpose(1, 0, 2), rtol=1e-12, atol=0)


def test_mesh_x_material_with_balls_adds_up():
    """The profile rule's points carry the ball's material."""
    from ball.oracle import Oracle
    model, _, m = _nested()
    mesh = common.mesh_filter((-10, -10, -10), (10, 10, 10), (5, 5, 5))
    mf = openmc.MaterialFilter(list(m.values()))
    model.tallies = openmc.Tallies([
        _tally("mat", [mf]), _tally("mesh_mat", [mesh, mf])])
    t = transport.run(openmc_import.load(model), backend=Oracle(),
                      r_star=1.0, **RUN).tallies
    mm = t["mesh_mat"]["mean"].reshape(125, 3, -1)
    assert np.allclose(mm.sum(axis=0), t["mat"]["mean"], rtol=1e-9)


def test_energy_filter_puts_everything_in_the_groups_bin():
    """One group [0, 20 MeV]: OpenMC gives every particle the midpoint,
    10 MeV (measured)."""
    model, c, _ = _nested()
    cf = openmc.CellFilter([c["core"], c["reflector"]])
    model.tallies = openmc.Tallies([
        _tally("cell", [cf]),
        _tally("cell_e", [cf, openmc.EnergyFilter([0.0, 9.99e6, 20e6])]),
        _tally("cell_e2", [cf, openmc.EnergyFilter([0.0, 10.01e6, 20e6])]),
        _tally("none", [openmc.EnergyFilter([20e6, 30e6])]),
    ])
    t = _run(model)
    ce = t["cell_e"]["mean"][:, 0].reshape(2, 2)
    assert np.all(ce[:, 0] == 0)
    assert np.allclose(ce[:, 1], t["cell"]["mean"][:, 0], rtol=1e-12)
    ce2 = t["cell_e2"]["mean"][:, 0].reshape(2, 2)
    assert np.all(ce2[:, 1] == 0) and np.all(ce2[:, 0] > 0)
    assert np.all(t["none"]["mean"] == 0)


def test_partial_currents_add_up_to_the_net_current():
    model, c, _ = _nested()
    xmax = [s for s in c["core"].region.get_surfaces().values()
            if isinstance(s, openmc.XPlane) and s.x0 == 4.0][0]
    sf = openmc.SurfaceFilter([xmax])
    two = [c["core"], c["reflector"]]
    model.tallies = openmc.Tallies([
        _tally("net", [sf], ("current",)),
        _tally("from", [openmc.CellFromFilter(two), sf], ("current",)),
        _tally("to", [sf, openmc.CellFilter(two)], ("current",)),
    ])
    t = _run(model)
    net = t["net"]["mean"][0, 0]
    fr, to = t["from"]["mean"][:, 0], t["to"]["mean"][:, 0]
    assert fr.sum() == pytest.approx(net, rel=1e-12)
    assert to.sum() == pytest.approx(net, rel=1e-12)
    # leaving the core is always outwards (+x), entering it inwards
    assert fr[0] > 0 and fr[1] < 0
    assert fr[0] == pytest.approx(to[1], rel=1e-12)
    assert fr[1] == pytest.approx(to[0], rel=1e-12)


def test_vacuum_face_cell_and_cell_from_are_the_cell_left():
    t = _run(PROBLEMS["slab"]())
    for name in ("sides_from", "sides_cell"):
        assert np.array_equal(t[name]["mean"], t["sides"]["mean"])


@pytest.mark.parametrize("filters, words", [
    (lambda c: [openmc.UniverseFilter([1])], "UniverseFilter"),
    (lambda c: [openmc.CellFromFilter([c["core"]])], "CellFromFilter"),
    (lambda c: [openmc.CellFilter([c["core"]]),
                openmc.CellFilter([c["reflector"]])], "two CellFilters"),
    (lambda c: [openmc.SurfaceFilter(list(
        c["core"].region.get_surfaces().values())[:1]), common.mesh_filter(
        (-10, -10, -10), (10, 10, 10), (2, 2, 2))], "MeshFilter"),
])
def test_other_filter_sets_go_to_the_general_scorer(filters, words):
    """Refused before v1.6.1; now scored by core/ext.py (checked against
    OpenMC by checks/gapcheck.py)."""
    model, c, _ = _nested()
    scores = ("current",) if "Surface" in words or words == "MeshFilter" \
        else ("flux",)
    model.tallies = openmc.Tallies([_tally("x", filters(c), scores)])
    if words == "UniverseFilter":
        model.tallies[0].filters = [openmc.UniverseFilter(
            [model.geometry.root_universe])]
    pr = openmc_import.load(model)
    assert pr.tallies[0].ext
    r = transport.run(pr, 2000, 2, seed=1)
    assert np.all(np.isfinite(r.tallies["x"]["mean"]))
