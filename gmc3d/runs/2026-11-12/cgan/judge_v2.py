"""cGAN at interface v2: the model-side comparison (whole balls, exact
first scatter) beside the accepted cVAE and the flow, then Innovation's
gate2 on each cGAN seed.

    python runs/2026-11-12/cgan/judge_v2.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from ball import models                                     # noqa: E402
from ball.models import compare_aniso                       # noqa: E402

f = open(HERE / "compare_v2.log", "w")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


iso = models.load("nflow", 0, name="network")
bes = {f"{fam}_v2_seed{s}": models.load(fam, s, interface=2, iso=iso)
       for fam, s in (("cgan", 0), ("cgan", 1), ("cvae", 0), ("nflow", 0))}
res = compare_aniso.run(bes, log=log)
(HERE / "compare_v2.json").write_text(json.dumps(res, indent=1, default=str))
(HERE / "gate").mkdir(exist_ok=True)
for s in (0, 1):
    m = f"cgan_v2_seed{s}"
    with open(HERE / "gate" / f"gate2_{m}.log", "w") as out:
        subprocess.run([sys.executable, "run.py", "gate2", m, "--out",
                        str(HERE / "gate")], cwd=ROOT, stdout=out,
                       stderr=subprocess.STDOUT)
log("FINISHED")
