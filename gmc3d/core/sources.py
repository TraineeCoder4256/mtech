"""Where particles are born, which way they go, and with what weight.

Used by: openmc_import.py (builds a `SourceSampler` from the model) and
transport.py (asks it for one batch of source sites before each batch).

One mechanism for every source type (since v1.2).  Before each batch,
Python draws the batch's sites -- position, direction, weight -- from the
model's own OpenMC source objects, and the transport starts its particles
from that list.  OpenMC's one-dimensional distributions (`Uniform`,
`Discrete`, `Normal`, `PowerLaw`, `Tabular`, `Mixture`, `Maxwell`, `Watt`,
...) already know how to sample themselves in Python, including their
biasing (`bias=`), which hands back a weight with every sample.  What OpenMC
keeps in C++ -- how those pieces combine into a position or a direction --
is written here, following OpenMC's definitions:

    Point, Box                    the point; uniform in the box
    CartesianIndependent          x, y, z each from its own distribution
    CylindricalIndependent        r, phi, z about `origin`
    SphericalIndependent          r, cos(theta), phi about `origin`
    PointCloud                    one of the points, chosen by strength
    MeshSpatial                   a mesh element chosen by strength (times
                                  its volume if `volume_normalized`), then
                                  a uniform point inside it
    Isotropic, Monodirectional    as named
    PolarAzimuthal                mu from the reference direction u, phi
                                  from the reference v:  mu u + sqrt(1 -
                                  mu^2) (cos(phi) v + sin(phi) (u x v))
    FileSource                    a site of the file, picked uniformly
                                  (with replacement), with its own weight
    MeshSource                    an element chosen by its source's
                                  strength, a uniform point inside it, and
                                  that source's angle

Several sources: each particle picks one with probability proportional to
its strength.  A biased choice (PointCloud / MeshSpatial `bias`) carries the
weight (true probability) / (biased probability).

Constraints.  `domains` (cells, materials or universes) are checked the way
OpenMC checks them: a cell or universe matches at ANY level of the point's
path, a material at the deepest.  A site outside the geometry, or outside
the domains, is drawn again.  Measured against OpenMC 0.16 on 29 Oct 2026
(multigroup, fixed source): `rejection_strategy: 'kill'` gives the same
answer as 'resample' for domains, and `energy_bounds` and `time_bounds`
change nothing.  gmc3d does the same, and names those settings in a warning.

Weights and the leakage.  Also measured on 29 Oct 2026: with a FileSource
whose sites all weigh 2, OpenMC's tallies double but its leakage does not.
Its tallies are per source PARTICLE, its global leakage per unit of source
WEIGHT.  `transport.run` divides the leakage by the batch's mean source
weight so the two codes report the same thing.

Strength.  Measured the same day: a single source of strength 3 gives
three times the tallies of strength 1 (same leakage), and two sources of
strength 1 twice those of two of strength 0.5.  So OpenMC starts every
particle with weight = the SUM of the sources' strengths (a MeshSource's
strength is the sum of its elements'), times any biasing weight.  The same
here.  Before v1.2 gmc3d ignored strength; every test problem uses 1.

Energy and time are sampled only for the weights a biased distribution
gives them; one energy group and a steady state use neither value.
"""

from dataclasses import dataclass

import numpy as np
import openmc
import openmc.stats as st

from . import geometry as G

MAX_TRIES = 1000            # rounds of drawing before giving up


class _Unsupported(ValueError):
    pass


@dataclass
class Sites:
    """One batch of source sites."""
    xyz: np.ndarray          # (n, 3)
    uvw: np.ndarray          # (n, 3) unit vectors
    wgt: np.ndarray          # (n,)
    iso: np.ndarray          # (n,) bool: direction drawn isotropically
    redrawn: int             # sites drawn again (outside geometry/domains)


def _seed(rng):
    return int(rng.integers(0, 2**31 - 1))


def _uni(dist, n, rng):
    """(values, weights) from an OpenMC univariate distribution."""
    x, w = dist.sample(n, seed=_seed(rng))
    return np.asarray(x, float), np.asarray(w, float)


def _choose(p_true, p_bias, n, rng):
    """Indices drawn with probabilities p_bias, and their weights."""
    p_true = np.asarray(p_true, float) / np.sum(p_true)
    q = p_true if p_bias is None else np.asarray(p_bias, float) / \
        np.sum(p_bias)
    idx = rng.choice(q.size, size=n, p=q)
    w = np.ones(n) if p_bias is None else p_true[idx] / q[idx]
    return idx, w


# ------------------------------------------------------------------ meshes
def _mesh_cells(mesh, where):
    """Element sampler for a mesh: (volumes, f(idx, rng) -> points)."""
    if isinstance(mesh, openmc.RegularMesh):
        if len(mesh.dimension) != 3:
            raise _Unsupported(f"{where}: mesh {mesh.id} is not 3D")
        ll = np.asarray(mesh.lower_left, float)
        ur = np.asarray(mesh.upper_right, float)
        edges = [np.linspace(ll[a], ur[a], int(mesh.dimension[a]) + 1)
                 for a in range(3)]
        kind = "box"
    elif isinstance(mesh, openmc.RectilinearMesh):
        edges = [np.asarray(g, float) for g in
                 (mesh.x_grid, mesh.y_grid, mesh.z_grid)]
        kind = "box"
    elif isinstance(mesh, openmc.CylindricalMesh):
        edges = [np.asarray(g, float) for g in
                 (mesh.r_grid, mesh.phi_grid, mesh.z_grid)]
        kind = "cyl"
    elif isinstance(mesh, openmc.SphericalMesh):
        edges = [np.asarray(g, float) for g in
                 (mesh.r_grid, mesh.theta_grid, mesh.phi_grid)]
        kind = "sph"
    else:
        raise _Unsupported(f"{where}: mesh {mesh.id} is a "
                           f"{type(mesh).__name__}; sources handle "
                           f"Regular (3D), Rectilinear, Cylindrical and "
                           f"Spherical meshes")
    dims = [e.size - 1 for e in edges]
    # element index: first axis fastest (OpenMC's order)
    lo = [np.empty(0)] * 3
    hi = [np.empty(0)] * 3
    grids = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
    ids = [g.ravel(order="F") for g in grids]
    for a in range(3):
        lo[a], hi[a] = edges[a][ids[a]], edges[a][ids[a] + 1]
    origin = np.asarray(getattr(mesh, "origin", (0.0, 0.0, 0.0)), float)
    if kind == "box":
        vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    elif kind == "cyl":
        vol = 0.5 * (hi[0] ** 2 - lo[0] ** 2) * (hi[1] - lo[1]) * \
            (hi[2] - lo[2])
    else:
        vol = (hi[0] ** 3 - lo[0] ** 3) / 3.0 * \
            (np.cos(lo[1]) - np.cos(hi[1])) * (hi[2] - lo[2])

    def points(idx, rng):
        n = idx.size
        r = rng.random((n, 3))
        a, b = [x[idx] for x in lo], [x[idx] for x in hi]
        if kind == "box":
            return np.column_stack([a[k] + (b[k] - a[k]) * r[:, k]
                                    for k in range(3)])
        if kind == "cyl":      # uniform in volume: r^2 uniform
            rr = np.sqrt(a[0] ** 2 + (b[0] ** 2 - a[0] ** 2) * r[:, 0])
            ph = a[1] + (b[1] - a[1]) * r[:, 1]
            zz = a[2] + (b[2] - a[2]) * r[:, 2]
            return origin + np.column_stack([rr * np.cos(ph),
                                             rr * np.sin(ph), zz])
        rr = np.cbrt(a[0] ** 3 + (b[0] ** 3 - a[0] ** 3) * r[:, 0])
        ct = np.cos(a[1]) + (np.cos(b[1]) - np.cos(a[1])) * r[:, 1]
        ph = a[2] + (b[2] - a[2]) * r[:, 2]
        stt = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
        return origin + np.column_stack([rr * stt * np.cos(ph),
                                         rr * stt * np.sin(ph), rr * ct])
    return vol, points


# ------------------------------------------------------------------ pieces
def _space(sp, where):
    """A spatial distribution -> f(n, rng) -> (points, weights)."""
    if sp is None:
        sp = st.Point()
    if isinstance(sp, st.Point):
        p = np.asarray(sp.xyz, float)
        return lambda n, rng: (np.tile(p, (n, 1)), np.ones(n))
    if isinstance(sp, st.Box):
        if sp.only_fissionable:
            raise _Unsupported(f"{where}: Box(only_fissionable=True); "
                               f"stage 1 has no fissile material")
        lo = np.asarray(sp.lower_left, float)
        hi = np.asarray(sp.upper_right, float)
        return lambda n, rng: (lo + (hi - lo) * rng.random((n, 3)),
                               np.ones(n))
    if isinstance(sp, st.CartesianIndependent):
        def f(n, rng):
            (x, wx), (y, wy), (z, wz) = (_uni(d, n, rng)
                                         for d in (sp.x, sp.y, sp.z))
            return np.column_stack([x, y, z]), wx * wy * wz
        return f
    if isinstance(sp, st.CylindricalIndependent):
        o = np.asarray(sp.origin, float)

        def f(n, rng):
            (r, wr), (ph, wp), (z, wz) = (_uni(d, n, rng)
                                          for d in (sp.r, sp.phi, sp.z))
            return o + np.column_stack([r * np.cos(ph), r * np.sin(ph),
                                        z]), wr * wp * wz
        return f
    if isinstance(sp, st.SphericalIndependent):
        o = np.asarray(sp.origin, float)

        def f(n, rng):
            (r, wr), (mu, wm), (ph, wp) = (_uni(d, n, rng) for d in
                                           (sp.r, sp.cos_theta, sp.phi))
            s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
            return o + np.column_stack([r * s * np.cos(ph),
                                        r * s * np.sin(ph), r * mu]), \
                wr * wm * wp
        return f
    if isinstance(sp, st.PointCloud):
        pts = np.asarray(sp.positions, float)
        s = np.ones(len(pts)) if sp.strengths is None else sp.strengths

        def f(n, rng):
            idx, w = _choose(s, sp.bias, n, rng)
            return pts[idx], w
        return f
    if isinstance(sp, st.MeshSpatial):
        vol, points = _mesh_cells(sp.mesh, where)
        s = np.ones(vol.size) if sp.strengths is None else \
            np.asarray(sp.strengths, float).ravel()
        if s.size != vol.size:
            raise _Unsupported(f"{where}: MeshSpatial has {s.size} "
                               f"strengths for {vol.size} elements")
        p = s * vol if sp.volume_normalized else s
        b = None if sp.bias is None else np.asarray(sp.bias, float).ravel()
        if b is not None and sp.volume_normalized:
            b = b * vol

        def f(n, rng):
            idx, w = _choose(p, b, n, rng)
            return points(idx, rng), w
        return f
    raise _Unsupported(f"{where}: source space {type(sp).__name__}")


def _isotropic(n, rng):
    mu = 2.0 * rng.random(n) - 1.0
    ph = 2.0 * np.pi * rng.random(n)
    s = np.sqrt(1.0 - mu * mu)
    return np.column_stack([s * np.cos(ph), s * np.sin(ph), mu])


def _angle(an, where):
    """An angular distribution -> (f(n, rng) -> (dirs, weights), iso)."""
    if an is None or (isinstance(an, st.Isotropic) and
                      getattr(an, "bias", None) is None):
        return (lambda n, rng: (_isotropic(n, rng), np.ones(n))), True
    if isinstance(an, st.Monodirectional):
        d = np.asarray(an.reference_uvw, float)
        d = d / np.linalg.norm(d)
        return (lambda n, rng: (np.tile(d, (n, 1)), np.ones(n))), False
    if isinstance(an, st.PolarAzimuthal):
        u = np.asarray(an.reference_uvw, float)
        u = u / np.linalg.norm(u)
        v = np.asarray(an.reference_vwu, float)
        v = v - np.dot(v, u) * u
        if np.linalg.norm(v) < 1e-12:
            raise _Unsupported(f"{where}: PolarAzimuthal reference_vwu is "
                               f"parallel to reference_uvw")
        v = v / np.linalg.norm(v)
        w = np.cross(u, v)

        def f(n, rng):
            (mu, wm), (ph, wp) = _uni(an.mu, n, rng), _uni(an.phi, n, rng)
            s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
            d = mu[:, None] * u + (s * np.cos(ph))[:, None] * v + \
                (s * np.sin(ph))[:, None] * w
            return d, wm * wp
        return f, False
    raise _Unsupported(f"{where}: source angle {type(an).__name__}")


def _extra_weight(dist):
    """Weights of an energy or time distribution (1 unless biased)."""
    if dist is None or getattr(dist, "bias", None) is None:
        return lambda n, rng: np.ones(n)
    return lambda n, rng: _uni(dist, n, rng)[1]


# ----------------------------------------------------------------- sources
class _Independent:
    def __init__(self, s, where):
        if str(s.particle) not in ("neutron", "ParticleType.NEUTRON"):
            raise _Unsupported(f"{where}: particle '{s.particle}'; stage 1 "
                               f"transports neutrons")
        self.space = _space(s.space, where)
        self.angle, self.iso = _angle(s.angle, where)
        self.ew, self.tw = _extra_weight(s.energy), _extra_weight(s.time)

    def draw(self, n, rng):
        xyz, w1 = self.space(n, rng)
        uvw, w2 = self.angle(n, rng)
        w = w1 * w2 * self.ew(n, rng) * self.tw(n, rng)
        return xyz, uvw, w, np.full(n, self.iso)


class _Mesh:
    def __init__(self, s, where):
        vol, self.points = _mesh_cells(s.mesh, where)
        if len(s.sources) != vol.size:
            raise _Unsupported(f"{where}: MeshSource has {len(s.sources)} "
                               f"sources for {vol.size} elements")
        self.parts = []
        for k, e in enumerate(s.sources):
            if not isinstance(e, openmc.IndependentSource):
                raise _Unsupported(f"{where}: element {k} source is a "
                                   f"{type(e).__name__}")
            if e.space is not None:
                raise _Unsupported(f"{where}: element {k} source sets a "
                                   f"space; the mesh element is the space")
            if e.constraints:
                raise _Unsupported(f"{where}: element {k} source has its "
                                   f"own constraints")
            self.parts.append(_Independent(e, f"{where} element {k}"))
        self.strength = np.array([e.strength for e in s.sources], float)

    def draw(self, n, rng):
        idx, w = _choose(self.strength, None, n, rng)
        xyz = self.points(idx, rng)
        uvw = np.empty((n, 3))
        iso = np.empty(n, bool)
        for k in np.unique(idx):
            sel = idx == k
            m = int(sel.sum())
            _, d, wk, isok = self.parts[k].draw(m, rng)
            uvw[sel], w[sel], iso[sel] = d, w[sel] * wk, isok
        return xyz, uvw, w, iso


class _File:
    def __init__(self, s, where):
        path = str(s.path)
        if not path.endswith(".h5"):
            raise _Unsupported(f"{where}: FileSource '{path}'; gmc3d reads "
                               f"OpenMC's .h5 source files")
        sites = openmc.read_source_file(path)
        bad = {str(p.particle) for p in sites} - {"neutron",
                                                   "ParticleType.NEUTRON"}
        if bad:
            raise _Unsupported(f"{where}: FileSource holds {sorted(bad)} "
                               f"particles; stage 1 transports neutrons")
        self.xyz = np.array([p.r for p in sites], float)
        uvw = np.array([p.u for p in sites], float)
        self.uvw = uvw / np.linalg.norm(uvw, axis=1)[:, None]
        self.wgt = np.array([p.wgt for p in sites], float)

    def draw(self, n, rng):
        idx = rng.integers(0, self.wgt.size, n)
        return self.xyz[idx], self.uvw[idx], self.wgt[idx].copy(), \
            np.zeros(n, bool)


# ----------------------------------------------------------------- sampler
class SourceSampler:
    """All of a model's sources, ready to give batches of sites.

    Raises openmc_import.Unsupported (via `make`) for what it cannot do;
    `warnings` lists the constraints it accepts but, like OpenMC in
    multigroup mode, does not apply."""

    def __init__(self, model, geom, cell_index, mat_index, univ_index):
        srcs = model.settings.source
        srcs = srcs if isinstance(srcs, (list, tuple)) else [srcs]
        self.geom = geom
        self.parts, self.domains, self.warnings = [], [], []
        strength = []
        for k, s in enumerate(srcs):
            where = f"source {k}"
            if isinstance(s, openmc.IndependentSource):
                part = _Independent(s, where)
            elif isinstance(s, openmc.MeshSource):
                part = _Mesh(s, where)
            elif isinstance(s, openmc.FileSource):
                part = _File(s, where)
            else:
                raise _Unsupported(f"{where}: {type(s).__name__} (user C++ "
                                   f"code cannot be read)"
                                   if isinstance(s, openmc.CompiledSource)
                                   else f"{where}: {type(s).__name__}")
            self.parts.append(part)
            self.domains.append(self._constraints(s, where, cell_index,
                                                  mat_index, univ_index))
            strength.append(s.strength)
        self.strength = np.asarray(strength, float)

    def _constraints(self, s, where, cell_index, mat_index, univ_index):
        c = dict(s.constraints or {})
        if getattr(s, "domains", None) and "domains" not in c:
            c["domains"] = s.domains
        if c.pop("fissionable", False):
            raise _Unsupported(f"{where}: constraint fissionable=True; "
                               f"stage 1 has no fissile material")
        for key in ("energy_bounds", "time_bounds"):
            if c.pop(key, None) is not None:
                self.warnings.append(
                    f"{where}: constraint '{key}' is not used (one energy "
                    f"group, steady state; OpenMC 0.16 in multigroup mode "
                    f"does not apply it either)")
        if c.pop("rejection_strategy", "resample") == "kill":
            self.warnings.append(
                f"{where}: rejection_strategy 'kill' acts as 'resample' "
                f"(as OpenMC 0.16 does for domains)")
        # OpenMC keeps 'domains' as a type and a list of ids
        dtype, ids = c.pop("domain_type", None), c.pop("domain_ids", [])
        for d in c.pop("domains", None) or []:
            dtype, ids = type(d).__name__.lower(), list(ids) + [d.id]
        if c:
            raise _Unsupported(f"{where}: constraints {sorted(c)}")
        if not ids:
            return None
        index = {"cell": cell_index, "material": mat_index,
                 "universe": univ_index}.get(dtype)
        if index is None:
            raise _Unsupported(f"{where}: domain type '{dtype}'")
        cells, mats, univs = [], [], []
        {"cell": cells, "material": mats, "universe": univs}[dtype].extend(
            index[int(i)] for i in ids)
        return (np.array(cells, np.int64), np.array(mats, np.int64),
                np.array(univs, np.int64))

    def _ok(self, xyz, dom):
        """Inside the geometry and (if any) inside the domains."""
        cells, univs, mats = G.paths_many(self.geom, xyz)
        ok = mats != G.LOST
        if dom is not None:
            c, m, u = dom
            hit = np.isin(cells, c).any(axis=1) | np.isin(mats, m) | \
                np.isin(univs, u).any(axis=1)
            ok &= hit
        return ok

    def sample(self, n, rng):
        """n sites.  rng: a numpy Generator (one per batch)."""
        if self.strength.size == 1:
            which = np.zeros(n, np.int64)
        else:
            which = rng.choice(self.strength.size, size=n,
                               p=self.strength / self.strength.sum())
        xyz, uvw = np.empty((n, 3)), np.empty((n, 3))
        wgt, iso = np.empty(n), np.empty(n, bool)
        redrawn = 0
        for k in np.unique(which):
            todo = np.flatnonzero(which == k)
            drawn, kept = 0, 0
            for _ in range(MAX_TRIES):
                # draw enough candidates to fill the rest, given the share
                # accepted so far (a candidate is used once or not at all)
                share = max(kept / drawn, 1e-3) if drawn else 1.0
                m = int(min(max(1.2 * todo.size / share, 64), 50 * n))
                p, d, w, i = self.parts[k].draw(m, rng)
                ok = np.flatnonzero(self._ok(p, self.domains[k]))
                drawn, kept = drawn + m, kept + ok.size
                ok = ok[:todo.size]
                put, todo = todo[:ok.size], todo[ok.size:]
                xyz[put], uvw[put], wgt[put], iso[put] = p[ok], d[ok], \
                    w[ok], i[ok]
                if todo.size == 0:
                    break
            else:
                raise RuntimeError(
                    f"source {k}: {todo.size} of {n} sites still outside "
                    f"the geometry or its domains after {drawn:,} draws")
            redrawn += drawn - kept
        wgt *= self.strength.sum()      # OpenMC's convention, see top
        return Sites(xyz, uvw, wgt, iso, redrawn)


def make(model, geom, cell_index, mat_index, univ_index, unsupported):
    """SourceSampler, turning its refusals into `unsupported`."""
    try:
        return SourceSampler(model, geom, cell_index, mat_index, univ_index)
    except _Unsupported as e:
        raise unsupported(str(e)) from None
