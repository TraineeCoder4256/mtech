#!/usr/bin/env python3
"""Is the benchmark still the same everywhere?  Run: python check_benchmark.py

The lattice is defined twice, on purpose.  mc.lattice() is what every model
is scored on; openmc_lattice.py keeps its own copy of the numbers and
imports nothing from this project, so that OpenMC stays an INDEPENDENT
check.  The price of two copies is that they could drift apart without
anyone noticing.  This script is what notices.

It checks three things, and stops at the first failure:

  1. DEFINITIONS   the constants in openmc_lattice.py (read from its source
                   text, so OpenMC need not be installed) against
                   mc.lattice(): size, pitch, mesh, absorber list, cross
                   sections at every scale, source cell and source box,
                   boundary conditions.
  2. REFERENCE     every reference/lattice_x*.npz was built from the problem
                   mc.lattice() defines TODAY (a stored fingerprint, compared
                   on load), and its file hash matches reference/manifest.json.
  3. AGREEMENT     where a reference file carries an OpenMC field, mc.py and
                   OpenMC agree cell by cell to within their combined
                   statistics, and a deliberately wrong comparison (OpenMC's
                   field flipped upside down) is rejected, which shows the
                   test has the power to see a real error.

Nothing here runs a transport code; it takes a second.
"""
import ast
import json
import pathlib
import re
import sys

import numpy as np

import benchmark as B
import mc

OMC = pathlib.Path("openmc_lattice.py")
CHI2_MAX = 2.0          # chi^2 per cell above this = the codes disagree
CONTROL_MIN = 50.0      # the flipped control must be at least this bad


def fail(msg):
    sys.exit(f"FAIL  {msg}")


def omc_constants(scale):
    """The module-level constants of openmc_lattice.py, evaluated with
    SCALE = scale, without importing it."""
    tree = ast.parse(OMC.read_text())
    env = {"SCALE": float(scale)}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            name = node.targets[0].id
            if name in ("L", "PITCH", "MESH", "BG", "AB", "SOURCE_CELL", "ABSORBERS"):
                env[name] = eval(compile(ast.Expression(node.value), str(OMC), "eval"),
                                 {"__builtins__": {}}, env)
    return env


def check_definitions():
    src = OMC.read_text()
    for scale in B.SCALES:
        c, p = omc_constants(scale), B.problem(scale)
        per = p["per_cm"]
        pairs = [("L", c["L"], p["L"]), ("PITCH", c["PITCH"], p["pitch"]),
                 ("MESH", c["MESH"], p["n"]),
                 ("SOURCE_CELL", tuple(c["SOURCE_CELL"]), tuple(p["source_cell"])),
                 ("ABSORBERS", sorted(map(tuple, c["ABSORBERS"])), sorted(mc.ABSORBERS))]
        # cross sections: background and absorber, read off mc's arrays
        is_ab = p["sig_a"] > 0
        pairs += [("BG sig_s", c["BG"][0], np.unique(p["sig_s"][~is_ab]).item()),
                  ("BG sig_a", c["BG"][1], np.unique(p["sig_a"][~is_ab]).item()),
                  ("AB sig_s", c["AB"][0], np.unique(p["sig_s"][is_ab]).item()),
                  ("AB sig_a", c["AB"][1], np.unique(p["sig_a"][is_ab]).item())]
        # the absorber map mc.py actually builds, cell by cell
        cells = {(x, y) for y in range(7) for x in range(7)
                 if is_ab[y * per:(y + 1) * per, x * per:(x + 1) * per].all()}
        pairs.append(("absorber map", sorted(cells), sorted(mc.ABSORBERS)))
        # source box: OpenMC spans the SOURCE_CELL-th pitch in x and y
        sx, sy = c["SOURCE_CELL"]
        box = (sx * c["PITCH"], sy * c["PITCH"], (sx + 1) * c["PITCH"], (sy + 1) * c["PITCH"])
        pairs.append(("source box", box, tuple(p["source"])))
        for what, a, b in pairs:
            if a != b:
                fail(f"scale {scale:g}: {what} differs -- openmc_lattice.py {a!r}, "
                     f"mc.lattice() {b!r}")
    # things that are code, not constants: check the text says them
    for what, pat in (("source box built from SOURCE_CELL and PITCH",
                       r"Box\(\(sx \* PITCH, sy \* PITCH"),
                      ("vacuum on all four sides", None),
                      ("reflective top and bottom in z", None),
                      ("isotropic scattering", r"xs\.order = 0"),
                      ("isotropic source", r"Isotropic\(\)")):
        if what.startswith("vacuum"):
            ok = len(re.findall(r'[XY]Plane\([^)]*boundary_type="vacuum"', src)) == 4
        elif what.startswith("reflective"):
            ok = len(re.findall(r'ZPlane\([^)]*boundary_type="reflective"', src)) == 2
        else:
            ok = re.search(pat, src) is not None
        if not ok:
            fail(f"openmc_lattice.py no longer shows: {what}")
    print(f"ok    definitions: openmc_lattice.py == mc.lattice() at scales {B.SCALES}")


def check_references():
    man_path = B.REFERENCE / "manifest.json"
    if not man_path.exists():
        fail(f"{man_path} missing -- run make_reference.py")
    man = json.loads(man_path.read_text())
    out = {}
    for scale in B.SCALES:
        path = B.reference_path(scale)
        try:
            ref = B.load_reference(scale)          # checks the fingerprint
        except (FileNotFoundError, ValueError) as e:
            fail(str(e))
        entry = man["files"].get(path.name)
        if entry is None or entry["sha256"] != B.sha256(path):
            fail(f"{path} does not match reference/manifest.json (edited or rebuilt "
                 f"without the manifest)")
        out[scale] = ref
    print(f"ok    reference: {len(out)} files match the definition and the manifest")
    return out


def check_agreement(refs):
    for scale, r in refs.items():
        if "omc_cell" not in r:
            print(f"--    scale {scale:g}: no OpenMC field in the reference")
            continue
        a = B.agreement(r["mc_cell"], r["mc_cell_se"], r["omc_cell"], r["omc_cell_sd"])
        ctl = B.agreement(r["mc_cell"], r["mc_cell_se"],
                          r["omc_cell"][::-1], r["omc_cell_sd"][::-1])
        fine = B.agreement(r["mc_fine"], r["mc_fine_se"], r["omc_fine"], r["omc_fine_sd"],
                           min_rel=0.2)
        ab_z = ((float(r["mc_absorbed"]) - float(r["omc_absorbed"]))
                / np.hypot(float(r["mc_absorbed_se"]), float(r["omc_absorbed_sd"])))
        print(f"      scale {scale:>3g}: 7x7 chi2/cell {a['chi2_per_cell']:.2f}, "
              f"max|z| {a['max_abs_z']:.2f}, median diff {a['median_rel_diff']*100:.2f}%; "
              f"fine chi2 {fine['chi2_per_cell']:.2f}; absorption z {ab_z:+.2f}; "
              f"control chi2 {ctl['chi2_per_cell']:,.0f}")
        if a["chi2_per_cell"] > CHI2_MAX or fine["chi2_per_cell"] > CHI2_MAX or abs(ab_z) > 4:
            fail(f"scale {scale:g}: mc.py and OpenMC disagree beyond statistics")
        if ctl["chi2_per_cell"] < CONTROL_MIN:
            fail(f"scale {scale:g}: the flipped control passes, so this test "
                 f"cannot see an orientation error")
    print("ok    agreement: mc.py == OpenMC to within statistics wherever both exist")


if __name__ == "__main__":
    check_definitions()
    check_agreement(check_references())
    print("\nBENCHMARK CONSISTENT")
