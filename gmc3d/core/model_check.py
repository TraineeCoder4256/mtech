"""What gmc3d will do with a model, without running it (SRS V1-ROB-6).

Used by: run.py (`python run.py check <model>`).

`load()` stops at the FIRST thing it refuses.  This goes through the model
part by part (settings, surfaces, periodic pairs, materials, cells and
lattices, sources, tallies) with the same importer functions, so the
report lists EVERY refusal at once, each with the importer's own message,
plus every warning and an inventory of what will be used.  Last it calls
`load()` itself, so a model reported as accepted really loads.

A model is given as:
    a problem name           (python run.py check slab)
    a Python file            that defines `model` (an openmc.Model) or a
                             function `build()` / `model()` returning one
    a model.xml file, or a folder holding model.xml or the separate
                             geometry / materials / settings / tallies xml
"""

from collections import Counter
from contextlib import contextmanager
import runpy
import warnings
from pathlib import Path

import numpy as np
import openmc

from . import openmc_import as I

Unsupported = I.Unsupported


def load_model(spec):
    """A problem name, .py file, model.xml or folder -> openmc.Model."""
    from problems import PROBLEMS
    if spec in PROBLEMS:
        return PROBLEMS[spec]()
    p = Path(spec)
    if p.suffix == ".py":
        ns = runpy.run_path(str(p), run_name="gmc3d_check")
        for key in ("model", "build"):
            obj = ns.get(key)
            if isinstance(obj, openmc.Model):
                return obj
            if callable(obj):
                out = obj()
                if isinstance(out, openmc.Model):
                    return out
        raise SystemExit(f"{p}: defines no openmc.Model named `model` and "
                         f"no build() / model() returning one")
    if p.is_dir():
        if (p / "model.xml").exists():
            return _macroscopics(openmc.Model.from_model_xml(
                p / "model.xml"), p / "model.xml")
        return _macroscopics(openmc.Model.from_xml(
            p / "geometry.xml", p / "materials.xml", p / "settings.xml",
            p / "tallies.xml" if (p / "tallies.xml").exists() else None),
            p / "materials.xml")
    if p.suffix == ".xml":
        return _macroscopics(openmc.Model.from_model_xml(p), p)
    raise SystemExit(f"{spec}: not a problem name, .py, .xml or folder")


def _macroscopics(model, xml):
    """OpenMC 0.16's Material.from_xml_element drops <macroscopic>
    (measured: a model written with export_to_model_xml comes back with
    no data in its materials), so put them back from the file."""
    import xml.etree.ElementTree as ET
    by_id = {m.id: m for m in model.materials}
    for el in ET.parse(xml).getroot().iter("material"):
        mac = el.find("macroscopic")
        m = by_id.get(int(el.get("id")))
        if mac is not None and m is not None and m._macroscopic is None:
            m.add_macroscopic(mac.get("name"))
    return model


class Report:
    def __init__(self):
        self.used, self.refused, self.warned = [], [], []
        self.loads = False

    @contextmanager
    def part(self, where):
        """Run a check; an Unsupported becomes a refusal line."""
        try:
            yield
        except Unsupported as e:
            msg = str(e)
            if msg not in self.refused:
                self.refused.append(msg)
        except Exception as e:                     # noqa: BLE001
            self.refused.append(f"{where}: could not be read "
                                f"({type(e).__name__}: {e})")

    def lines(self):
        out = ["USED"] + [f"  {u}" for u in self.used]
        out += ["REFUSED" + (" (nothing)" if not self.refused else "")]
        out += [f"  {r}" for r in self.refused]
        out += ["WARNED" + (" (nothing)" if not self.warned else "")]
        out += [f"  {w}" for w in self.warned]
        out.append("VERDICT  " + (
            "accepted: gmc3d will run this model" if self.loads else
            "refused: gmc3d will not run this model until the items under "
            "REFUSED are changed"))
        return out


def _counts(items):
    return ", ".join(f"{n} {k}" for k, n in sorted(Counter(items).items()))


def check(model):
    """Report for an openmc.Model (see the module docstring)."""
    rep = Report()
    s = model.settings
    with rep.part("settings"):
        rep.warned += I._settings(model)
    rep.used.append(f"settings: {s.run_mode or 'fixed source'}, "
                    f"{s.particles} particles x {s.batches} batches, "
                    f"energy mode {s.energy_mode}")

    geo = model.geometry
    surfs = sorted(geo.get_all_surfaces().values(), key=lambda x: x.id)
    for x in surfs:
        with rep.part(f"surface {x.id}"):
            I._surface(x)
    rep.used.append(f"surfaces: {_counts(type(x).__name__ for x in surfs)}")
    rep.used.append("boundaries: " + _counts(x.boundary_type for x in surfs))
    with rep.part("periodic surfaces"):
        I._periodic_shifts(surfs)

    with rep.part("materials"):
        mats = I._materials(model)
        for k, name in enumerate(mats.names):
            rep.used.append(
                f"material {mats.ids[k]} '{name}': sigma_s "
                f"{mats.sig_s[k]:.4g}, sigma_a {mats.sig_a[k]:.4g} /cm")

    cells = sorted(geo.get_all_cells().values(), key=lambda c: c.id)
    lats = sorted(geo.get_all_lattices().values(), key=lambda l: l.id)
    univs = geo.get_all_universes()
    rep.used.append(f"cells: {len(cells)} ({_counts(c.fill_type for c in cells)})"
                    f"; universes: {len(univs)}; lattices: "
                    f"{_counts(type(l).__name__ for l in lats) or 'none'}")
    for lat in lats:
        if not isinstance(lat, (openmc.RectLattice, openmc.HexLattice)):
            rep.refused.append(f"lattice {lat.id} is a {type(lat).__name__}"
                               f"; RectLattice and HexLattice are handled")
    for c in cells:
        if c.fill_type not in ("material", "distribmat", "void", "universe",
                               "lattice"):
            rep.refused.append(f"cell {c.id} fill type '{c.fill_type}'")

    srcs = s.source if isinstance(s.source, (list, tuple)) else [s.source]
    for k, src in enumerate(srcs):
        desc = type(src).__name__
        if isinstance(src, openmc.IndependentSource):
            desc += (f" (space {type(src.space).__name__}, angle "
                     f"{type(src.angle).__name__}, strength {src.strength})")
        rep.used.append(f"source {k}: {desc}")

    surf_index = {x.id: i for i, x in enumerate(surfs)}
    cell_index = {c.id: i for i, c in enumerate(cells)}
    try:
        mats_ = I._materials(model)
    except Exception:                               # noqa: BLE001
        mats_ = I.Materials([], [], np.zeros(0), np.zeros(0), np.zeros(0))
    mat_index = {m.id: i for i, m in enumerate(
        list(model.materials) or geo.get_all_materials().values())}
    for t in model.tallies:
        with rep.part(f"tally {t.id}"):
            tw = []
            I._check_tally(t, tw)
            tl = I._tally(t, model, mats_, cell_index, mat_index, surf_index)
            rep.warned += tw
            rep.used.append(f"tally {t.id} '{t.name}': filters "
                            f"[{tl.filter}], scores {list(t.scores)}, "
                            f"{tl.nbins} bins")

    # Sources need the compiled geometry: load the model without its
    # tallies (refused tallies are already listed above), then whole.
    saved = model.tallies
    try:
        model.tallies = openmc.Tallies()
        with rep.part("source"), warnings.catch_warnings():
            warnings.simplefilter("ignore", I.IgnoredSetting)
            pr = I.load(model)
            rep.warned += [w for w in pr.warnings if w not in rep.warned]
    finally:
        model.tallies = saved
    with rep.part("model"), warnings.catch_warnings():
        warnings.simplefilter("ignore", I.IgnoredSetting)
        I.load(model)
        rep.loads = not rep.refused
    return rep
