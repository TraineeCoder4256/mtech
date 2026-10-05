"""Run OpenMC on the same model and read its answer in our format.

Used by: run.py (--openmc), checks/validate.py.
Gives: a tallies.Results, so compare.py can line it up bin by bin with
ours.

The model object is the one problems/*.py built -- the very object
openmc_import.load() compiled for our solver -- so both codes solve the
same problem by construction (SRS S1-GEO-1).  OpenMC runs in multigroup
mode with the one-group library common.py wrote.

OpenMC reports tallies per source particle with the standard error of the
batch means, as tallies.py does, and the leakage as one of its "global
tallies".  Results are cached in the run folder (statepoint file), so a
rerun of the same model and settings is free.
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import openmc

from core.tallies import Results

OPENMC = os.path.join(sys.prefix, "bin", "openmc")


def run(model, workdir, particles=None, batches=None, seed=1, threads=None,
        reuse=True):
    """Run OpenMC (or reuse a finished run in `workdir`).  Returns Results."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    s = model.settings
    if particles:
        s.particles = int(particles)
    if batches:
        s.batches = int(batches)
    s.seed = int(seed)
    sp_path = workdir / f"statepoint.{s.batches}.h5"
    key = _key(model, workdir)
    keyfile = workdir / "run.key"
    wall = None
    if not (reuse and sp_path.exists() and keyfile.exists()
            and keyfile.read_text() == key):
        t = time.perf_counter()
        sp_path = model.run(cwd=str(workdir), threads=threads or os.cpu_count(),
                            output=False,
                            openmc_exec=OPENMC if os.path.exists(OPENMC)
                            else "openmc")
        wall = time.perf_counter() - t
        keyfile.write_text(key)
    return read(sp_path, wall)


def read(sp_path, wall=None):
    with openmc.StatePoint(str(sp_path)) as sp:
        g = {row[0].decode(): row for row in sp.global_tallies}
        leak, leak_se = float(g["leakage"][3]), float(g["leakage"][4])
        out = {}
        for tid, t in sp.tallies.items():
            col = [t.scores.index(k) for k in ("flux", "absorption")]
            mean = t.mean.reshape(t.mean.shape[0], -1)[:, col]
            se = t.std_dev.reshape(t.std_dev.shape[0], -1)[:, col]
            out[t.name or f"tally{tid}"] = {"id": tid, "mean": mean,
                                            "se": se}
        timing = {"openmc_transport_s": float(sp.runtime["transport"]),
                  "openmc_total_s": float(sp.runtime["total"])}
        if wall is not None:
            timing["wall_s"] = wall
        settings = {"particles": sp.n_particles, "batches": sp.n_batches,
                    "code": "openmc " + ".".join(map(str, sp.version))}
    return Results(leak, leak_se, out, {}, timing, settings)


def _key(model, workdir):
    """A fingerprint of the whole model (geometry, materials, settings,
    tallies): a cached run is reused only if this matches."""
    tmp = workdir / "fingerprint.xml"
    model.export_to_model_xml(str(tmp))
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.unlink()
    return digest
