"""Shared helpers for writing test problems with OpenMC's own API.

Used by: every file in problems/.

Stage 1 is one energy group, so a material is just two numbers: its
scattering and absorption cross sections (1/cm).  OpenMC needs those in a
multigroup library file; `one_group_materials` writes that file and returns
matching openmc.Material objects, so a problem file reads like any other
OpenMC input.
"""

from pathlib import Path

import numpy as np
import openmc

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "data" / "mgxs"


# The scattering law every problem's materials get (v2.0): None =
# isotropic, as before; otherwise set by `scattering_law` below.
_LAW = {"law": None, "tag": ""}


class scattering_law:
    """`with common.scattering_law(law, tag): model = build()` gives every
    material of the problem built inside that law, and writes its library
    as data/mgxs/<name>_<tag>.h5 (so the isotropic one is untouched).

    law: Legendre moments [1, E P_1, E P_2, ...] (OpenMC's 'legendre'
    format; a_l / a_0 = E[P_l(mu)]), ("histogram", bins) for OpenMC's
    histogram format (equal bins on [-1, 1], any positive heights),
    ("tabular", f) for its tabular format (f at equally spaced points on
    [-1, 1]), or a dict {material name: law} for one law per material
    (a material not named stays isotropic).
    """

    def __init__(self, law, tag):
        self.law, self.tag = law, tag

    def __enter__(self):
        self.old = dict(_LAW)
        _LAW.update(law=self.law, tag=self.tag)
        return self

    def __exit__(self, *exc):
        _LAW.update(self.old)


def _scattering(d, ss, mname):
    """Set d's format and order; return its scatter matrix (OpenMC needs
    the order before any cross section is set)."""
    law = _LAW["law"]
    if isinstance(law, dict):
        law = law.get(mname)
    if law is None:
        d.order = 0
        return np.array([[[ss]]])
    if isinstance(law, tuple) and law[0] == "histogram":
        h = np.asarray(law[1], float)
        d.scatter_format = "histogram"
        d.order = h.size
        return np.array([[ss * h / h.sum()]])
    if isinstance(law, tuple) and law[0] == "tabular":
        f = np.asarray(law[1], float)
        d.scatter_format = "tabular"
        d.order = f.size
        return np.array([[ss * f / np.trapezoid(f, dx=2.0 / (f.size - 1))]])
    a = np.asarray(law, float)
    d.order = a.size - 1
    return np.array([[ss * a / a[0]]])


def one_group_materials(name, xs):
    """xs = {material name: (sigma_s, sigma_a)} in 1/cm.

    Writes data/mgxs/<name>.h5 and returns (openmc.Materials, {name: Material}).
    """
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    lib = openmc.MGXSLibrary(groups)
    mats = {}
    if _LAW["tag"]:
        name = f"{name}_{_LAW['tag']}"
    for mname, (ss, sa) in xs.items():
        d = openmc.XSdata(mname, groups)
        sm = _scattering(d, ss, mname)
        d.set_total([ss + sa])
        d.set_absorption([sa])
        d.set_scatter_matrix(sm)
        lib.add_xsdata(d)
        m = openmc.Material(name=mname)
        m.set_density("macro", 1.0)
        m.add_macroscopic(mname)
        mats[mname] = m
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    path = LIBRARY_DIR / f"{name}.h5"
    lib.export_to_hdf5(str(path))
    materials = openmc.Materials(list(mats.values()))
    materials.cross_sections = str(path)
    return materials, mats


def settings(source, particles=100_000, batches=20):
    """Fixed-source, one-group settings shared by every problem."""
    s = openmc.Settings()
    s.run_mode = "fixed source"
    s.energy_mode = "multi-group"
    s.particles = particles
    s.batches = batches
    s.source = source
    return s


def point_source(xyz, direction=None):
    """Isotropic point source, or one direction if `direction` is given."""
    angle = (openmc.stats.Monodirectional(direction) if direction is not None
             else openmc.stats.Isotropic())
    return openmc.IndependentSource(space=openmc.stats.Point(xyz),
                                    angle=angle,
                                    energy=openmc.stats.Discrete([1.0e6],
                                                                 [1.0]))


def box_source(lower_left, upper_right):
    """Isotropic source uniform in a box."""
    return openmc.IndependentSource(
        space=openmc.stats.Box(lower_left, upper_right),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([1.0e6], [1.0]))


def flux_tally(name, filt=None, scores=("flux", "absorption")):
    t = openmc.Tally(name=name)
    if filt is not None:
        t.filters = [filt]
    t.scores = list(scores)
    return t


def mesh_filter(lower_left, upper_right, dimension):
    m = openmc.RegularMesh()
    m.lower_left, m.upper_right, m.dimension = lower_left, upper_right, \
        dimension
    return openmc.MeshFilter(m)


def point_detector(model, name, xyz, r0=0.0):
    """Ask gmc3d for the flux at one point (gmc3d's own; OpenMC 0.16 has no
    point detector, so the reference run ignores it).  Scored with the
    next-event estimator (core/detectors.py).  r0 > 0 (cm) averages
    contributions from closer than r0 to keep the variance finite: biased
    by O(r0), see that file.  Returns the model."""
    if not hasattr(model, "gmc3d_detectors"):
        model.gmc3d_detectors = []
    model.gmc3d_detectors.append((str(name), tuple(float(c) for c in xyz),
                                  float(r0)))
    return model
