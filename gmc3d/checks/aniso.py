"""Anisotropic scattering checked (v2.0; `python run.py anisocheck`).

Used by: run.py.  Writes anisocheck.json, which checks/score.py reads.

Every way OpenMC 0.16 lets a one-group multigroup library scatter
anisotropically, on the problems of problems/ given that law (OpenMC
reference, compared with checks/validate.py's rule), plus:

    mixed        one anisotropic material among isotropic ones: balls must
                 still be used in the isotropic cells, and the answer agree
    filters      the scatter-cosine tallies (LegendreFilter, scatter-cosine
                 SphericalHarmonicsFilter, MuFilter) against OpenMC
    detectors    point detectors in a void (cavity, outside) against
                 OpenMC's small-sphere flux, as checks/detcheck.py
    displacement EXACT: a beam into an infinite medium (a vacuum sphere of
                 60 mfp, leakage ~ e^-35) is absorbed on average at
                       E[x] = 1 / (sigma_t - sigma_s mu_bar)
                 along the beam (every flight moves 1/sigma_t on average
                 along the last direction, whose mean cosine to the beam
                 shrinks by mu_bar per scatter, while the particle survives
                 with c = sigma_s / sigma_t), with mu_bar the mean cosine
                 of the law OpenMC samples (core/scattering.moments).  The
                 law is the only thing that moves E[x]: P3 2.000 cm against
                 0.909 isotropic.

v2.0 had no balls in anisotropic materials.  v2.1 walks them exactly
(ball/walk.walk_aniso): every case runs with exact balls too (held to one
voxel where a mesh tally exists), and the *_balls cases repeat seven of
them without mesh tallies, so the anisotropic balls keep their full size.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import openmc

from ball.oracle import Oracle
from core import openmc_import, scattering as SC, transport
from problems import PROBLEMS, common
from . import compare, detcheck, openmc_ref, validate

PARTICLES, BATCHES = 50_000, 20      # ours; OpenMC the same, in 400 batches

P1 = [1.0, 0.3]
P3 = [1.0, 0.6, 0.3, 0.1]
CLIPPED = [1.0, 0.9, 0.5, -0.4]          # f(0) < 0: OpenMC clips it to 0
BACKWARD = [1.0, -0.3]
P5 = [1.0, 0.7, 0.4, 0.2, 0.1, 0.05]     # run with settings.max_order = 2
HIST = ("histogram", [1, 1, 2, 3, 5, 8, 13, 20])
TABULAR = ("tabular", [0.2, 0.3, 0.6, 1.2, 2.0])

# name: (problem, law, settings changes)
CASES = {
    "sphere_p3": ("sphere", P3, {}),
    "nested_clipped": ("nested", CLIPPED, {}),
    "cask_p1": ("cask", P1, {}),
    "hexcore_histogram": ("hexcore", HIST, {}),
    "tilted_tabular": ("tilted", TABULAR, {}),
    "sphere_rejection": ("sphere", P3, {"tabular_legendre":
                                        {"enable": False}}),
    "sphere_points_9": ("sphere", CLIPPED, {"tabular_legendre":
                                            {"enable": True,
                                             "num_points": 9}}),
    "lattice3d_max_order": ("lattice3d", P5, {"max_order": 2}),
    "curved_backward": ("curved", BACKWARD, {}),
    "mixed_cask": ("cask", {"concrete": P3}, {}),
}


# v2.1: the same problems with every mesh tally removed, so that balls in
# anisotropic materials keep their full size (with a mesh an anisotropic
# ball is held inside one voxel, transport._track)
# (lattice3d is left out: its cells are too thin for a ball of 1 mfp even
# when isotropic, so it would test nothing)
FULL_BALLS = ("sphere_p3", "nested_clipped", "cask_p1",
              "hexcore_histogram", "tilted_tabular", "curved_backward",
              "mixed_cask")


def _no_meshes(model):
    model.tallies = openmc.Tallies([
        t for t in model.tallies
        if not any(isinstance(f, (openmc.MeshFilter, openmc.MeshSurfaceFilter,
                                  openmc.MeshBornFilter))
                   for f in t.filters)])
    return model


def _build(problem, law, tag, settings):
    with common.scattering_law(law, tag):
        model = PROBLEMS[problem]()
    for k, v in settings.items():
        setattr(model.settings, k, v)
    return model


def _histogram_scatter_fix(ref, model):
    """OpenMC 0.16 scores 'scatter' (and nu-scatter) for histogram data
    order / 2 times too high (GAPS.md 2b): a material's macroscopic data
    is rebuilt from ScattDataHistogram::get_matrix, which returns
    scattxs * f(mu) with f a density (sum f dmu = 1, so sum f = 1 / dmu),
    and the rebuilt scattxs is that sum.  Transport uses total and
    absorption, so flux and absorption are right; only the score is off.
    Undo it on the reference's scatter column so the rest compares."""
    orders = {x.order for x in openmc.MGXSLibrary.from_hdf5(
        model.materials.cross_sections).xsdatas
        if x.scatter_format == "histogram"}
    if not orders:
        return None
    (order,) = orders                 # one histogram size per problem here
    k = 2.0 / order
    col = openmc_import.SCORES.index("scatter")
    for t in ref.tallies.values():
        if "mean" in t and t["mean"].ndim == 2 and t["mean"].shape[1] > col:
            t["mean"][:, col] *= k
            t["se"][:, col] *= k
    return order / 2.0


def _compare(name, model, workdir, particles, batches, log, balls=True,
             seed=3, need_balls=False):
    pr = openmc_import.load(model)
    n = particles * batches
    ref = openmc_ref.run(model, f"{workdir}/aniso_{name}",
                         n // validate.REF_BATCHES, validate.REF_BATCHES,
                         seed=seed + 100)
    raw_scatter = float(ref.tallies["total"]["mean"][0, 2]) \
        if "total" in ref.tallies else None
    factor = _histogram_scatter_fix(ref, model)
    if factor:
        log(f"  {name:22s} OpenMC's histogram scatter score divided by "
            f"{factor:g} (its bug, GAPS.md 2b): total scatter "
            f"{raw_scatter:.6g} -> {raw_scatter / factor:.6g}")
    kinds = sorted({int(k) for k, _ in pr.mats.laws})
    out = {"kinds": kinds, "openmc_scatter_divided_by": factor,
           "mean_cosines": [SC.mean_cosine(k, v) for k, v in pr.mats.laws]}
    runs = (("mc", None),) + ((("balls", Oracle()),) if balls else ())
    for label, be in runs:
        t = time.perf_counter()
        r = transport.run(pr, particles, batches, seed=seed, backend=be,
                          r_star=1.0, mesh_rule="profile")
        cmp = compare.compare(r, ref)
        ok, bad = validate.verdict(cmp)
        if r.counters["lost"]:
            ok, bad = False, bad + [f"{r.counters['lost']} lost"]
        if label == "balls" and need_balls and not r.counters["balls"]:
            ok, bad = False, bad + ["no balls were used"]
        out[label] = {"pass": ok, "bad": bad, "balls": r.counters["balls"],
                      "seconds": time.perf_counter() - t, "cmp": cmp}
        log(f"  {name:22s} {label:5s} {'PASS' if ok else 'FAIL'}  balls "
            f"{r.counters['balls']:,}  " + "; ".join(bad))
    out["pass"] = all(out[lab]["pass"] for lab, _ in runs)
    return out


def filters_model():
    model = _build("sphere", P3, "p3", {})
    t = []
    for name, f in (("legendre", openmc.LegendreFilter(3)),
                    ("sph_scatter", openmc.SphericalHarmonicsFilter(2)),
                    ("mu", openmc.MuFilter(np.linspace(-1, 1, 9)))):
        if name == "sph_scatter":
            f.cosine = "scatter"
        tal = openmc.Tally(name=name)
        tal.filters, tal.scores = [f], ["scatter"]
        t.append(tal)
    model.tallies = openmc.Tallies(list(model.tallies) + t)
    return model


def _detectors(workdir, particles, batches, log):
    with common.scattering_law(P3, "p3"):
        model = detcheck.scatter_model()
    pr = openmc_import.load(model)
    n = particles * batches
    ref = openmc_ref.run(model, f"{workdir}/aniso_detectors",
                         n // validate.REF_BATCHES,
                         validate.REF_BATCHES * detcheck.REF_SCALE, seed=107)
    r = transport.run(pr, particles, batches, seed=7)
    out, ok = {}, True
    for k, (name, rho) in enumerate((("cavity", detcheck.RHO_CAV),
                                     ("outside", detcheck.RHO_OUT))):
        vol = 4.0 / 3.0 * np.pi * rho ** 3
        e = ref.tallies["spheres"]["mean"][k, 0] / vol
        se = ref.tallies["spheres"]["se"][k, 0] / vol
        v, sv = r.tallies[name]["mean"][0, 0], r.tallies[name]["se"][0, 0]
        z = float((v - e) / np.hypot(sv, se))
        good = abs(z) <= validate.Z1_MAX
        ok &= good
        out[name] = {"value": v, "se": sv, "openmc": e, "openmc_se": se,
                     "z": z, "pass": bool(good)}
        log(f"  detector {name:13s} mc    {'PASS' if good else 'FAIL'}  "
            f"{v:.5g} +- {sv:.2g} vs OpenMC sphere {e:.5g} +- {se:.2g} "
            f"(z {z:+.2f})")
    out["pass"] = bool(ok)
    return out


def displacement_model(law, tag):
    ss, sa, R = 1.0, 0.1, 60.0
    with common.scattering_law(law, tag):
        mats, m = common.one_group_materials("aniso_disp",
                                             {"medium": (ss, sa)})
    s = openmc.Sphere(r=R, boundary_type="vacuum")
    tal = openmc.Tally(name="xmom")
    tal.filters = [openmc.SpatialLegendreFilter(1, "x", -R, R)]
    tal.scores = ["absorption"]
    model = openmc.Model(
        openmc.Geometry([openmc.Cell(fill=m["medium"], region=-s)]), mats,
        common.settings(common.point_source((0.0, 0.0, 0.0),
                                            direction=(1.0, 0.0, 0.0))),
        openmc.Tallies([common.flux_tally("total"), tal]))
    return model, ss, sa, R


def _displacement(log, particles=20_000, batches=20):
    out, ok = {}, True
    for tag, law in (("iso", None), ("p1", P1), ("p3", P3),
                     ("clipped", CLIPPED), ("backward", BACKWARD),
                     ("hist", HIST), ("tab", TABULAR)):
        model, ss, sa, R = displacement_model(law, tag if law else "")
        pr = openmc_import.load(model)
        mu_bar = SC.mean_cosine(*pr.mats.laws[0])
        exact = 1.0 / (ss + sa - ss * mu_bar)
        r = transport.run(pr, particles, batches, seed=9)
        # SpatialLegendre scores P_1(x / R) = x / R; column 1 absorption
        v = r.tallies["xmom"]["mean"][1, 1] * R
        sv = r.tallies["xmom"]["se"][1, 1] * R
        z = float((v - exact) / sv)
        good = abs(z) <= validate.Z1_MAX
        ok &= good
        out[tag] = {"mu_bar": mu_bar, "exact": exact, "value": v, "se": sv,
                    "z": z, "pass": bool(good), "leakage": r.leakage}
        log(f"  displacement {tag:9s} {'PASS' if good else 'FAIL'}  "
            f"mu_bar {mu_bar:+.4f}: E[x] {v:.4f} +- {sv:.4f} cm, exact "
            f"{exact:.4f} (z {z:+.2f}), leakage {r.leakage:.1e}")
    out["pass"] = bool(ok)
    return out


def suite(workdir, particles=PARTICLES, batches=BATCHES, names=None,
          log=print):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    rows = {}

    def guard(name, f):
        if names and name not in names:
            return
        t0 = time.perf_counter()
        try:
            rows[name] = f()
        except Exception as e:                    # noqa: BLE001
            rows[name] = {"pass": False,
                          "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()}
            log(f"  {name:22s} ERROR {type(e).__name__}: {e}")
        rows[name]["seconds"] = time.perf_counter() - t0

    log("== displacement along a beam in an infinite medium (exact)")
    guard("displacement", lambda: _displacement(log))
    log("\n== problems with anisotropic laws, vs OpenMC")
    for name, (prob, law, st) in CASES.items():
        guard(name, lambda: _compare(
            name, _build(prob, law, name, st), workdir, particles, batches,
            log))
    log("\n== the same without mesh tallies: full-size anisotropic balls "
        "(v2.1)")
    for name in FULL_BALLS:
        prob, law, st = CASES[name]
        guard(name + "_balls", lambda: _compare(
            name + "_balls", _no_meshes(_build(prob, law, name, st)),
            workdir, particles, batches, log, need_balls=True))
    guard("filters", lambda: _compare("filters", filters_model(), workdir,
                                      particles, batches, log, balls=False))
    guard("detectors", lambda: _detectors(workdir, particles, batches, log))
    passed = sum(bool(r.get("pass")) for r in rows.values())
    log(f"\nanisocheck: {passed} of {len(rows)} cases pass")
    Path(workdir, "anisocheck.json").write_text(json.dumps(
        rows, indent=1, default=_json))
    return rows


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
