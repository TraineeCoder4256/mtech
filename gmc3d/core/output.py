"""Write results to disk: JSON for numbers, VTK for 3D pictures.

Used by: run.py and checks/validate.py.
Reads: tallies.Results.

    save_json   every tally (mean and standard error per bin), the leakage,
                the work counters, timings and settings, in one file
    save_vtk    one mesh tally as a VTK file that ParaView opens directly
                (File > Open, then colour by flux, absorption or the
                relative error)  -- SRS S1-TAL-3

The VTK file is the old "legacy" text format: a header, the grid's corner,
spacing and size, then one value per voxel, x fastest -- the same order the
tallies use, so no reshuffling is needed.
"""

import json
from pathlib import Path

import numpy as np


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def save_json(results, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"leakage": results.leakage, "leakage_se": results.leakage_se,
            "leakage_se_batches": results.leakage_se_batches,
            "error_bar_groups": results.groups,
            "tallies": {}, "counters": results.counters,
            "timing": results.timing, "settings": results.settings}
    for name, t in results.tallies.items():
        data["tallies"][name] = {
            k: t[k] for k in ("id", "filter", "bin_ids", "mesh", "mean", "se")
            if k in t}
        data["tallies"][name]["columns"] = ["flux", "absorption"]
    if extra:
        data.update(extra)
    path.write_text(json.dumps(_plain(data), indent=1))


def save_vtk(tally, path, title="gmc3d mesh tally"):
    """One mesh tally (an entry of Results.tallies) -> a .vtk file."""
    ll, ur, dim = (np.asarray(a) for a in tally["mesh"])
    spacing = (ur - ll) / dim
    n = int(np.prod(dim))
    mean, se = tally["mean"], tally["se"]
    rel = np.where(mean[:, 0] > 0, se[:, 0] / np.where(mean[:, 0] > 0,
                                                        mean[:, 0], 1), 0.0)
    lines = ["# vtk DataFile Version 3.0", title, "ASCII",
             "DATASET STRUCTURED_POINTS",
             "DIMENSIONS {} {} {}".format(*(dim + 1)),
             "ORIGIN {} {} {}".format(*ll),
             "SPACING {} {} {}".format(*spacing),
             f"CELL_DATA {n}"]
    for name, col in (("flux", mean[:, 0]), ("absorption", mean[:, 1]),
                      ("flux_rel_error", rel)):
        lines.append(f"SCALARS {name} double 1")
        lines.append("LOOKUP_TABLE default")
        lines.append("\n".join(f"{v:.8e}" for v in col))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
