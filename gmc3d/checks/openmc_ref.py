"""Run OpenMC on the same model and read its answer in our format.

Used by: run.py (--openmc), checks/validate.py.
Gives: a tallies.Results, so compare.py can line it up bin by bin with
ours.

The model object is the one problems/*.py built -- the very object
openmc_import.load() compiled for our solver -- so both codes solve the
same problem by construction (SRS S1-GEO-1).  OpenMC runs in multigroup
mode with the one-group library common.py wrote.

OpenMC reports tallies per source particle with the standard error of the
batch means, and the leakage as one of its "global tallies".  Results are
cached in the run folder (statepoint file), so a rerun of the same model and
settings is free.

HOW MANY BATCHES.  For the same number of histories the batch count
changes only the ERROR BAR, never the answer, and it costs no time.
Measured here on 6 October 2026, 4,000,000 histories per row
(`runs/2026-10-06/ref_batches.log`):

    problem   split              transport s   leakage                rel err
    sphere    40 x 100,000            13.02    0.0314987 +- 8.6e-05   0.2720%
    sphere    200 x 20,000            12.13    0.0314987 +- 8.7e-05   0.2760%
    sphere    1000 x 4,000            12.43    0.0314988 +- 8.9e-05   0.2835%
    slab      40 x 100,000            12.05    0.860771  +- 1.89e-04  0.0219%
    slab      200 x 20,000            12.94    0.860770  +- 1.76e-04  0.0205%
    slab      1000 x 4,000            13.12    0.860771  +- 1.80e-04  0.0209%
    cask      40 x 100,000            21.58    0.0023605 +- 2.3e-05   0.9633%
    cask      200 x 20,000            19.98    0.0023605 +- 2.4e-05   0.9955%
    cask      1000 x 4,000            22.68    0.0023605 +- 2.4e-05   1.0175%

The answers agree to six figures and the times to within this container's
own +-6% run-to-run spread, so splitting finely is free.

What it buys is a trustworthy error bar.  An error bar estimated from only
40 numbers is itself uncertain by about 11%, and a figure of merit, going as
one over its square, by about 24% -- and a single run cannot tell you
whether it drew a high one or a low one.  The slab shows both sides of
that: at the seed above its 40-batch error is 0.000189 against a converged
0.000180, but at another seed the same 40-batch estimate came out 0.000131,
25% low, which flatters OpenMC and makes our comparison look worse than it
is.

So a reference run that will be used for a figure of merit should be split
into MANY batches (`checks/race.py` uses 1,000).  core/tallies.py does the
same thing on our side by taking its groups from the thread chunks; see its
docstring and `checks/error_bars.py`.

THE TIME IS THE MACHINE'S, NOT THE CODE'S.  `openmc_transport_s` is
OpenMC's own internal transport timer out of the statepoint, not a wall
clock, so it is clean -- and it still moved a long way between containers.
The same four problems, the same 4M histories, the same OpenMC 0.16.0
(commit 617d35a5), analog mode:

    problem    5 Oct container    6 Oct container
    sphere          8.38 s         11.90 - 12.71 s
    slab            9.68 s         12.21 - 14.11 s
    cask           14.89 s         19.98 - 22.68 s
    curved         35.16 s         45.45 - 55.49 s

Our own rows did not move at all over the same rebuild (sphere plain MC
14.22 s then, 12.66 - 14.02 s now; slab table 9.40 s then, 9.10 - 10.45 s
now), so this is not a general slowdown: it is OpenMC's branch-heavy CSG
transport meeting a different host.  Nothing in the container recorded the
5 October CPU, so there is no way to say which machine is representative.

One suspect ruled out: conda-forge now prefers the DAGMC variant
(`openmc-0.16.0-dagmc_nompi_py312h012c719_200`), so the rebuilt environment
may not hold the same build as before.  Timed on the sphere's own input,
4 threads, the DAGMC build runs at 327-337k particles/s and the plain
`nodagmc_nompi_py312h0956b6e_0` build at 264-289k, so the build we have is
the FASTER of the two and neither reaches the 477k/s the 5 October machine
implies.  The variant is not the explanation.

The consequence for every multiplier in this project: compare only rows
measured in the SAME session.  The sphere lookup-table run is 3.20x OpenMC
analog on the 5 October machine and 4.6-5.8x on this one; both are honest,
and mixing them is not.
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
