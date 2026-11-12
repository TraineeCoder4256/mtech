"""12 Nov 2026 (Innovation, catch-up for Phani): every reference problem, OpenMC vs ours with
each trained model family, for accuracy (z vs OpenMC analog) and speed (FOM).

Run from gmc3d/:  python runs/2026-11-12/race-catchup/race_models.py SEED
Uses checks/race.race with an extended contender list; models are loaded
through run.backend (the registry), seed 0 of each family unless named.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import run as R                                           # noqa: E402
from checks import race, validate                        # noqa: E402
from core import output                                   # noqa: E402

# the four v1 models the 5 Nov race left out, plus anchors (plain MC, exact
# walk, nflow and cvae seed 0) re-run here, because the code has moved on
# since 5 Nov (v1.6.5 -> v2.1.7) and timings differ between sessions
CONTENDERS = (("ours, plain MC", None, None),
              ("ours + exact walk", "oracle", 2.0),
              ("ours + nflow", "nflow", 3.0),
              ("ours + cvae", "cvae", 3.0),
              ("ours + nflow seed 1", "nflow_seed1", 3.0),
              ("ours + cfm seed 1", "cfm_seed1", 3.0),
              ("ours + cgan", "cgan", 3.0),
              ("ours + cgan seed 1", "cgan_seed1", 3.0))

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
names = sys.argv[2:] or list(validate.VALIDATE_ORDER)
here = Path(__file__).resolve().parent
logf = open(here / f"race_seed{seed}.log", "w")


def log(s=""):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


bes = {b: R.backend(b) for b in {c[1] for c in CONTENDERS if c[1]}}
for b, be in bes.items():
    d = be.describe() if hasattr(be, "describe") else {}
    log(f"backend {b}: {json.dumps({k: d[k] for k in list(d)[:6]}, default=str)}")
res = race.race(names, bes, particles=100_000, batches=40, seed=seed,
                workdir=str(R.DATA / "openmc"), contenders=CONTENDERS, log=log)
(here / f"race_seed{seed}.json").write_text(
    json.dumps(output._plain(res), indent=1))
log("done")
