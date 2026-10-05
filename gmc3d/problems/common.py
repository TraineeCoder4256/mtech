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


def one_group_materials(name, xs):
    """xs = {material name: (sigma_s, sigma_a)} in 1/cm.

    Writes data/mgxs/<name>.h5 and returns (openmc.Materials, {name: Material}).
    """
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    lib = openmc.MGXSLibrary(groups)
    mats = {}
    for mname, (ss, sa) in xs.items():
        d = openmc.XSdata(mname, groups)
        d.order = 0
        d.set_total([ss + sa])
        d.set_absorption([sa])
        d.set_scatter_matrix(np.array([[[ss]]]))
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
