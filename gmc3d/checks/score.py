"""One number out of 10 for how well gmc3d is checked (Phani, 30 Oct 2026).

Used by: run.py (`python run.py score`).

The score adds up five parts.  Each part is the share of its checks that
pass, times its points:

    part                     points  one check is
    unit tests                  2    one test (tests.xml, from pytest)
    against OpenMC              4    one tally (or the leakage) of one
                                     validation problem, for plain MC or
                                     for exact balls (validate.json)
    exact answers               1    one analytic check (validate.json)
    random models               2    one robustness model, passing or
                                     refused cleanly (robust.json)
    point detectors             1    one detector check (detcheck.json)

A tally counts as one check however many scores it has: its scores are
the same flux times different cross sections, so counting them separately
would let one tally outweigh others.  A part whose result file is missing
scores 0, and says so: the score never assumes a check passed.

What the score does NOT measure (printed with it, so nobody reads 10 as
"everything is fine"):
    * the trained NETWORK backend.  Every check here uses the exact walk
      ("exact balls").  The network has its own known error (the rare
      short-path tail, GAPS.md), and is checked by `run.py ballcheck`.
    * speed
    * physics beyond Stage 1 (one group, isotropic, fixed source)
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from . import validate

PARTS = (("unit tests", 2.0), ("against OpenMC", 4.0),
         ("exact answers", 1.0), ("random models", 2.0),
         ("point detectors", 1.0))
NOT_SCORED = ("the trained network backend (every check here uses the "
              "exact walk; the network is checked by run.py ballcheck)",
              "speed",
              "physics beyond Stage 1 (one group, isotropic scattering, "
              "fixed source)")


def _tests(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    ok, fails = 0, []
    for s in suites:
        for c in s.iter("testcase"):
            name = f"{c.get('classname')}::{c.get('name')}"
            bad = [e for e in c if e.tag in ("failure", "error")]
            skipped = any(e.tag == "skipped" for e in c)
            if bad or skipped:
                fails.append(name + (" (skipped)" if skipped else ""))
            else:
                ok += 1
    return ok, ok + len(fails), fails


def _openmc(val):
    ok, n, fails = 0, 0, []
    for prob, res in val["problems"].items():
        for mode in ("mc_vs_openmc", "balls_vs_openmc"):
            groups = {}
            for key, e in res[mode].items():
                groups.setdefault(key.split("/")[0], {})[key] = e
            for tally, entries in groups.items():
                passed, bad = validate.verdict(entries)
                n += 1
                if passed:
                    ok += 1
                else:
                    fails.append(f"{prob} {mode.split('_')[0]} {tally}: "
                                 + "; ".join(bad))
            lost = res["counters_mc" if mode.startswith("mc")
                       else "counters_balls"]["lost"]
            n += 1
            if lost == 0:
                ok += 1
            else:
                fails.append(f"{prob} {mode.split('_')[0]}: {lost} lost "
                             f"particles")
    return ok, n, fails


def _analytic(val):
    ok, fails = 0, []
    for name, v, exp, se in val["analytic"]:
        good = abs(v - exp) <= validate.Z1_MAX * se if se > 0 else \
            abs(v - exp) <= 1e-9 * max(abs(exp), 1e-300)
        ok += good
        if not good:
            fails.append(f"{name}: {v:.8g} vs {exp:.8g} (se {se:.2g})")
    return ok, len(val["analytic"]), fails


def _robust(rob):
    fails = [f"model {r['model']}: {r['status']} "
             f"({r.get('reason') or 'see robust.log'})"
             for r in rob["models"] if r["status"] not in ("PASS",
                                                           "REFUSED")]
    return rob["ok"], rob["total"], fails


def _detectors(det):
    fails = [c["name"] + ": " + c["detail"] for c in det["checks"]
             if not c["pass"]]
    return len(det["checks"]) - len(fails), len(det["checks"]), fails


def score(folder):
    """{part: (points earned, points, passed, checks, failures)} and the
    total, from the result files in `folder` (or the newest run folder
    holding each file, if `folder` lacks it)."""
    folder = Path(folder)
    runs = folder.parent

    def find(name):
        if (folder / name).exists():
            return folder / name
        hits = sorted(runs.rglob(name), key=lambda p: p.stat().st_mtime)
        return hits[-1] if hits else None

    files = {"unit tests": find("tests.xml"),
             "against OpenMC": find("validate.json"),
             "exact answers": find("validate.json"),
             "random models": find("robust.json"),
             "point detectors": find("detcheck.json")}
    readers = {"unit tests": _tests,
               "against OpenMC": lambda p: _openmc(json.loads(p.read_text())),
               "exact answers": lambda p: _analytic(
                   json.loads(p.read_text())),
               "random models": lambda p: _robust(json.loads(p.read_text())),
               "point detectors": lambda p: _detectors(
                   json.loads(p.read_text()))}
    out, total = {}, 0.0
    for part, pts in PARTS:
        path = files[part]
        if path is None:
            out[part] = dict(earned=0.0, points=pts, passed=0, checks=0,
                             failures=["no result file: not run"],
                             file=None)
            continue
        ok, n, fails = readers[part](path)
        earned = pts * ok / n if n else 0.0
        total += earned
        out[part] = dict(earned=earned, points=pts, passed=ok, checks=n,
                         failures=fails, file=str(path))
    return out, total


def report(parts, total):
    lines = ["| part | checks passed | points |", "|---|---|---|"]
    for part, p in parts.items():
        lines.append(f"| {part} | {p['passed']} / {p['checks']} | "
                     f"{p['earned']:.2f} / {p['points']:.0f} |")
    lines.append(f"| **total** | | **{total:.1f} / 10** |")
    fails = [(part, f) for part, p in parts.items() for f in p["failures"]]
    lines += ["", "Failed checks:" if fails else "Failed checks: none"]
    lines += [f"- {part}: {f}" for part, f in fails]
    lines += ["", "Not measured by this score:"]
    lines += [f"- {x}" for x in NOT_SCORED]
    lines += ["", "From: " + ", ".join(sorted({Path(p['file']).name + ' (' +
                                               Path(p['file']).parent.name +
                                               ')' for p in parts.values()
                                               if p['file']}))]
    return "\n".join(lines)
