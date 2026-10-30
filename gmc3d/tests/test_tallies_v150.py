"""v1.5: cylindrical and spherical meshes, currents on any surface, mesh
surface currents.

Run:  python -m pytest -q tests   (from gmc3d/)

Agreement with OpenMC's own transport is checked by `python run.py
validate` (sphere, curved: curved meshes; lattice3d, tilted, sector,
nested: currents on reflective, white and periodic walls and inside
universes; slab, nested: mesh surface currents).  The checks here are
the ones OpenMC cannot make: the walk against brute force, and a theta
edge at pi/2 (the plane z = 0), which OpenMC 0.16 scores wrongly.
"""
import numpy as np
import openmc
import pytest
from numba import njit, prange

from ball.oracle import Oracle
from core import openmc_import, transport
from core import tallies as T
from core.openmc_import import Filter, Tally, Unsupported
from problems import PROBLEMS, common

RNG = np.random.default_rng(150)


def _layout(mesh, kind="mesh"):
    box, edges, mtype, origin = openmc_import._mesh(
        type("t", (), {"id": 1}), mesh)
    n = int(np.prod(box[2]))
    if kind == "meshsurface":
        n *= 12
    f = Filter(kind, n, mesh=box, edges=edges, mesh_type=mtype,
               origin=origin)
    L, X = T.compile_layouts([Tally(1, "m", kind, ["flux"], [f])], 1, 1)
    return L, X, n


@njit(cache=True)
def _bins_of(L, X, pts):
    out = np.empty(pts.shape[0], np.int64)
    for i in range(pts.shape[0]):
        out[i] = T._point_bin(L, X, 0, pts[i, 0], pts[i, 1], pts[i, 2])
    return out


CURVED = {
    "cyl": openmc.CylindricalMesh(
        r_grid=[0, 2, 5, 9], phi_grid=np.linspace(0, 2 * np.pi, 7),
        z_grid=[-6, -1, 3, 8], origin=(1.0, -0.5, 0.3)),
    "cyl-part": openmc.CylindricalMesh(
        r_grid=[1, 4, 7], phi_grid=[0.3, 1.2, 2.5], z_grid=[-3, 0, 3]),
    "sph": openmc.SphericalMesh(
        r_grid=[0, 3, 6, 10], theta_grid=[0, 0.7, np.pi / 2, 2.2, np.pi],
        phi_grid=np.linspace(0, 2 * np.pi, 5), origin=(-1.0, 0.5, 2.0)),
    "sph-part": openmc.SphericalMesh(
        r_grid=[2, 5, 8], theta_grid=[0.4, 1.3, 2.0],
        phi_grid=[1.0, 3.0, 5.0]),
}


@pytest.mark.parametrize("name", CURVED)
def test_curved_walk_matches_brute_force(name):
    """Each flight's track length per bin equals what 20,000 points along
    it say (to the points' spacing)."""
    L, X, nb = _layout(CURVED[name])
    for _ in range(60):
        p = RNG.uniform(-12, 12, 3)
        d = RNG.normal(size=3)
        d /= np.linalg.norm(d)
        length = RNG.uniform(1, 30)
        row = np.zeros(L.n_entries)
        T._curved_segment(row, L, X, 0, L.offset[0], 0.0, 0.0, 1.0, *p, *d,
                          length)
        got = row[L.offset[0]::T.NS][:nb]
        n = 20000
        ts = (np.arange(n) + 0.5) / n * length
        b = _bins_of(L, X, p + ts[:, None] * d)
        ref = np.zeros(nb)
        np.add.at(ref, b[b >= 0], length / n)
        assert np.abs(got - ref).max() < 2.0 * length / n


@pytest.mark.parametrize("name", CURVED)
def test_curved_room_keeps_a_ball_in_one_bin(name):
    """The 'cap' rule's room: no point of the ball is in another bin, and
    for a centre inside the mesh the room is tight (a ball 5% bigger
    usually reaches another bin).  Outside, the room can be smaller than
    it need be: the nearest edge surface may be a plane's or a cone's
    extension beyond the mesh, which bounds no bin there."""
    L, X, _ = _layout(CURVED[name])
    tight = inside = 0
    for _ in range(400):
        c = RNG.uniform(-9, 9, 3)
        r = T.mesh_room(L, X, *c)
        b0 = T._point_bin(L, X, 0, *c)
        q = RNG.normal(size=(3000, 3))
        q /= np.linalg.norm(q, axis=1)[:, None]
        assert (_bins_of(L, X, c + 0.999 * r * q) == b0).all()
        if b0 >= 0:
            inside += 1
            tight += (_bins_of(L, X, c + 1.05 * r * q) != b0).any()
    assert inside > 10 and tight > 0.75 * inside


@njit(cache=True, parallel=True)
def _uncollided(L, X, nb, nd, nt, R, sa):
    """Exact uncollided flux per bin from a point source at the origin in
    a sphere of radius R, pure absorber sa: directions on a Fibonacci
    sphere, each path cut in nt steps."""
    out = np.zeros((nd, nb))
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in prange(nd):
        zc = 1.0 - 2.0 * (i + 0.5) / nd
        rr = np.sqrt(1.0 - zc * zc)
        u, v, w = rr * np.cos(ga * i), rr * np.sin(ga * i), zc
        dt = R / nt
        for k in range(nt):
            t = (k + 0.5) * dt
            b = T._point_bin(L, X, 0, t * u, t * v, t * w)
            if b >= 0:
                out[i, b] += np.exp(-sa * t) * dt
    return out.sum(axis=0) / nd


def test_theta_bins_with_an_edge_at_pi_over_2_are_exact():
    """The case OpenMC 0.16 gets wrong (its answer is up to 360 sigma off
    this one, measured 30 Oct 2026, whatever the origin; cones elsewhere
    it gets right): ours agrees with the exact answer, about an offset
    origin too."""
    model = PROBLEMS["sphere"](sig_s=1e-9, sig_a=0.1)
    mesh = openmc.SphericalMesh(
        r_grid=np.linspace(0, 12, 6),
        theta_grid=[0, 0.8, np.pi / 2, 2.4, np.pi],
        phi_grid=np.linspace(0, 2 * np.pi, 5), origin=(1.0, -0.5, 0.5))
    t = openmc.Tally(name="s")
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ["flux"]
    model.tallies = [t]
    pr = openmc_import.load(model)
    _, L, _, X = transport.physics(pr)
    ex = _uncollided(L, X, 80, 100_000, 2000, 10.0, 0.1)
    r = transport.run(pr, 200_000, 10, seed=4)
    m, se = r.tallies["s"]["mean"][:, 0], r.tallies["s"]["se"][:, 0]
    ok = ex > 1e-6
    z = (m - ex)[ok] / se[ok]
    assert np.mean(z ** 2) < 1.6
    assert np.abs(z).max() < 4.5


# ------------------------------------------------------------ currents
def _spheres(layout):
    """Absorbing spheres in a moderator box (vacuum), as a probe for
    currents inside universes.  layout: 'root', 'shifted' (one universe,
    translated) or 'lattice' (a 3 x 3 lattice of the universe).  The net
    current into the spheres must be minus their absorption: there is no
    source inside them.  OpenMC 0.16 gets this right for 'root' only
    (shifted: -0.290 against 0.312, lattice: +0.169 against 0.493;
    measured 30 Oct 2026, runs/2026-10-30/v1.5/openmc_bugs.log)."""
    mats, m = common.one_group_materials(
        "probe", {"mod": (1.0, 0.01), "abs": (0.2, 2.0)})
    sph = openmc.Sphere(r=2.0)
    box = openmc.model.RectangularParallelepiped(
        -10, 10, -10, 10, -10, 10, boundary_type="vacuum")
    pin = openmc.Cell(name="abs", fill=m["abs"], region=-sph)
    if layout == "root":
        mod = openmc.Cell(name="mod", fill=m["mod"], region=+sph & -box)
        cells = [pin, mod]
    else:
        mod = openmc.Cell(name="modin", fill=m["mod"], region=+sph)
        u = openmc.Universe(cells=[pin, mod])
    if layout == "root":
        pass
    elif layout == "shifted":
        c = openmc.Cell(fill=u, region=-box)
        c.translation = (2.0, 1.0, 0.0)
        cells = [c]
    else:
        lat = openmc.RectLattice()
        lat.lower_left, lat.pitch = (-12, -12), (8, 8)
        lat.universes = [[u] * 3] * 3
        cells = [openmc.Cell(fill=lat, region=-box)]
    cur = openmc.Tally(name="cur")
    cur.filters = [openmc.SurfaceFilter([sph])]
    cur.scores = ["current"]
    part = openmc.Tally(name="part")
    part.filters = [openmc.CellFromFilter([pin, mod]),
                    openmc.SurfaceFilter([sph])]
    part.scores = ["current"]
    ab = openmc.Tally(name="ab")
    ab.filters = [openmc.MaterialFilter([m["abs"]])]
    ab.scores = ["absorption"]
    return openmc.Model(openmc.Geometry(cells), mats, common.settings(
        common.point_source((5.0, 0.3, 0.2))), [cur, part, ab])


@pytest.mark.parametrize("layout", ["root", "shifted", "lattice"])
def test_inner_currents_conserve_weight(layout):
    """Net current into the absorbing spheres = - their absorption, at any
    universe level, and the partial currents add up to the net."""
    pr = openmc_import.load(_spheres(layout))
    for be in (None, Oracle()):
        r = transport.run(pr, 20000, 8, seed=3, backend=be, r_star=1.0)
        c, cse = r.tallies["cur"]["mean"][0, 0], r.tallies["cur"]["se"][0, 0]
        a, ase = r.tallies["ab"]["mean"][0, 1], r.tallies["ab"]["se"][0, 1]
        assert c < 0
        assert abs(c + a) < 4.0 * np.hypot(cse, ase)
        part = r.tallies["part"]["mean"][:, 0]
        assert part[0] > 0 > part[1]           # out of the sphere, into it
        assert part.sum() == pytest.approx(c, rel=1e-12)


def _quarter(pos, d):
    """A quarter cylinder with rotational periodic walls x = 0, y = 0, a
    near-void medium (sigma_a 0.01) and a beam from pos along d."""
    mats, m = common.one_group_materials("rotb", {"mod": (1e-6, 0.01)})
    x0 = openmc.XPlane(0.0, boundary_type="periodic")
    y0 = openmc.YPlane(0.0, boundary_type="periodic")
    x0.periodic_surface = y0
    wall = openmc.ZCylinder(r=10.0, boundary_type="vacuum")
    zb = openmc.ZPlane(-5.0, boundary_type="vacuum")
    zt = openmc.ZPlane(5.0, boundary_type="vacuum")
    cells = [openmc.Cell(fill=m["mod"],
                         region=+x0 & +y0 & -wall & +zb & -zt)]
    t = openmc.Tally(name="w")
    t.filters = [openmc.SurfaceFilter([x0, y0])]
    t.scores = ["current"]
    src = openmc.IndependentSource(space=openmc.stats.Point(pos),
                                   angle=openmc.stats.Monodirectional(d))
    return openmc.Model(openmc.Geometry(cells), mats, common.settings(src),
                        [t])


@pytest.mark.parametrize("pos, d, want", [
    # leaves through x = 0 once: + w on y = 0, where it comes back in
    ((0.5, 5.0, 0.0), (-1.0, 0.0, 0.0), (0.0, np.exp(-0.005))),
    # twice (after the first turn it heads for x = 0 again)
    ((0.3, 5.4, 0.0), (-0.6, -0.8, 0.0),
     (0.0, np.exp(-0.005) + np.exp(-0.01 * 6.75))),
    ((5.0, 0.5, 0.0), (0.0, -1.0, 0.0), (np.exp(-0.005), 0.0)),
])
def test_rotational_wall_current_is_scored_where_the_particle_comes_in(
        pos, d, want):
    """Signed by the direction against the wall it comes back through.
    OpenMC 0.16 signs it against the wall it LEFT, after turning the
    direction: -1, +1 and -1 + 0.94 here (measured), which is no current
    of either wall; for translational walls (parallel normals) the two
    agree, and tilted.py checks those against OpenMC."""
    r = transport.run(openmc_import.load(_quarter(pos, d)), 2000, 2, seed=1)
    got = r.tallies["w"]["mean"][:, 0]
    assert got == pytest.approx(want, rel=2e-3, abs=1e-12)


def test_white_and_reflective_walls_score_zero():
    pr = openmc_import.load(PROBLEMS["tilted"]())
    r = transport.run(pr, 10000, 4, seed=6)
    walls = r.tallies["walls"]["mean"][:, 0]
    assert walls[2] == 0.0 and walls[3] == 0.0       # the white y faces
    assert walls[0] != 0.0 and walls[1] != 0.0       # the periodic x faces
    pr = openmc_import.load(PROBLEMS["lattice3d"]())
    r = transport.run(pr, 10000, 4, seed=6)
    assert (r.tallies["mirrors"]["mean"][:, 0] == 0.0).all()


# ------------------------------------------------- mesh surface currents
@njit(cache=True)
def _raw_bins(L, pts):
    """Each point's bin on each axis, not clamped (-1 or n outside)."""
    out = np.empty((pts.shape[0], 3), np.int64)
    for i in range(pts.shape[0]):
        for a in range(3):
            out[i, a] = T._bin(L, 0, a, pts[i, a])
    return out


def test_mesh_faces_match_brute_force():
    """Every element change along a flight, found from 20,000 points, is
    one 'out' and/or one 'in' in the face kernel, weighted exp(-sa t)."""
    m = openmc.RectilinearMesh()
    m.x_grid, m.y_grid, m.z_grid = [-4, -1, 1, 4], [-3, 0, 2, 3], [-2, 2]
    L, X, nb = _layout(m, "meshsurface")
    dim = L.mesh_dim[0]
    sa = 0.07
    for _ in range(200):
        p = RNG.uniform(-6, 6, 3)
        d = RNG.normal(size=3)
        d /= np.linalg.norm(d)
        length = RNG.uniform(1, 15)
        row = np.zeros(L.n_entries)
        T.score_mesh_faces(row, L, X, sa, 1.0, *p, *d, length)
        got = row[L.offset[0]::T.NS][:nb]
        n = 20000
        ts = (np.arange(n) + 0.5) / n * length
        c = _raw_bins(L, p + ts[:, None] * d)
        inside = ((c >= 0) & (c < dim)).all(axis=1)
        ref = np.zeros(nb)
        for k in np.flatnonzero((c[1:] != c[:-1]).any(axis=1)):
            if not (inside[k] or inside[k + 1]):
                continue                    # a wall's extension outside
            ax = int(np.flatnonzero(c[k] != c[k + 1])[0])
            up = d[ax] > 0
            wt = np.exp(-sa * 0.5 * (ts[k] + ts[k + 1]))
            for j, side in ((k, 2 if up else 0), (k + 1, 1 if up else 3)):
                if inside[j]:
                    e = (c[j, 2] * dim[1] + c[j, 1]) * dim[0] + c[j, 0]
                    ref[e * 12 + 4 * ax + side] += wt
        assert np.abs(got - ref).max() < 1e-3


def test_mesh_faces_balance_and_balls_agree():
    """In the slab, what leaves one element through a face enters the
    next (exactly, flight by flight), nothing enters from outside, and
    balls (capped at the faces) agree with plain MC."""
    pr = openmc_import.load(PROBLEMS["slab"]())
    a = transport.run(pr, 20000, 8, seed=7)
    f = a.tallies["faces1d"]["mean"][:, 0].reshape(6, 4)
    assert np.allclose(f[:-1, 2], f[1:, 1], rtol=1e-12)
    assert np.allclose(f[1:, 0], f[:-1, 3], rtol=1e-12)
    assert f[0, 1] == 0.0 and f[-1, 3] == 0.0
    b = transport.run(pr, 20000, 8, seed=7, backend=Oracle(), r_star=1.0)
    assert b.counters["balls"] > 10000
    ta, tb = a.tallies["faces1d"], b.tallies["faces1d"]
    ok = ta["se"][:, 0] > 0
    z = (ta["mean"][ok, 0] - tb["mean"][ok, 0]) / np.hypot(
        ta["se"][ok, 0], tb["se"][ok, 0])
    assert np.abs(z).max() < 4.0


def test_mesh_surface_filter_takes_no_other_filter():
    model = PROBLEMS["slab"]()
    m = openmc.RegularMesh()
    m.lower_left, m.upper_right, m.dimension = (0,), (60,), (6,)
    t = openmc.Tally(name="x")
    t.filters = [openmc.MeshSurfaceFilter(m), openmc.CellFilter(
        list(model.geometry.get_all_cells().values()))]
    t.scores = ["current"]
    model.tallies.append(t)
    with pytest.raises(Unsupported, match="MeshSurfaceFilter"):
        openmc_import.load(model)
