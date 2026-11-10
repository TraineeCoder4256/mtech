#!/bin/bash
# Second half of the m2.0 run: train_all.py hit the 2 h background limit
# while training cvae seed 1 (step ~2000), so: cvae seed 1, then the
# model-side comparison of all six, then Innovation's gate on each.
set -e
cd "$(dirname "$0")/../../.."
P=/opt/mm/root/envs/gmc/bin/python
H=runs/2026-11-10/models-v2
$P - <<'PY'
import sys; sys.path.insert(0, ".")
from ball import models
f = open("runs/2026-11-10/models-v2/train_cvae_v2_seed1.log", "w")
def log(*a):
    f.write(" ".join(map(str, a)) + "\n"); f.flush()
models.train("cvae", 1, interface=2, log=log)
PY
$P - <<'PY'
import sys, json; sys.path.insert(0, ".")
from pathlib import Path
from ball import models
from ball.models import compare_aniso, fairness
H = Path("runs/2026-11-10/models-v2")
f = open(H / "compare.log", "w")
def log(*a):
    s = " ".join(map(str, a)); print(s, flush=True); f.write(s + "\n"); f.flush()
iso = models.load("nflow", 0, name="network")
bes = {f"{fam}_v2_seed{s}": models.load(fam, s, interface=2, iso=iso)
       for s in fairness.SEEDS for fam in models.names()}
res = compare_aniso.run(bes, log=log)
(H / "compare.json").write_text(json.dumps(res, indent=1, default=str))
PY
for m in nflow cfm cvae; do for s in 0 1; do
  $P run.py gate2 ${m}_v2_seed$s --out $H/gate > $H/gate/gate2_${m}_v2_seed$s.log 2>&1 || echo "gate2 ${m}_v2_seed$s exit $?"
done; done
echo FINISHED
