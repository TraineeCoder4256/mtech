"""m2.1: the model-side comparison of all six (whole balls, exact first
scatter), then Innovation's gate on each.

    python runs/2026-11-10/models-v2.1/judge.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from ball import models                                     # noqa: E402
from ball.models import compare_aniso, fairness             # noqa: E402

f = open(HERE / "compare.log", "w")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


iso = models.load("nflow", 0, name="network")
bes = {f"{fam}_v2_seed{s}": models.load(fam, s, interface=2, iso=iso)
       for s in fairness.SEEDS for fam in models.names()}
res = compare_aniso.run(bes, log=log)
(HERE / "compare.json").write_text(json.dumps(res, indent=1, default=str))
(HERE / "gate").mkdir(exist_ok=True)
for s in fairness.SEEDS:
    for fam in models.names():
        m = f"{fam}_v2_seed{s}"
        with open(HERE / "gate" / f"gate2_{m}.log", "w") as out:
            subprocess.run([sys.executable, "run.py", "gate2", m, "--out",
                            str(HERE / "gate")], cwd=ROOT, stdout=out,
                           stderr=subprocess.STDOUT)
log("FINISHED")
