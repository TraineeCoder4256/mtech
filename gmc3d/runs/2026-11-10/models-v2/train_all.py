"""Train every family at interface v2, seeds 0 and 1, in one process (the
41M-row training set is built once).  Then the model-side comparison.

    python runs/2026-11-10/models-v2/train_all.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from ball import models                                     # noqa: E402
from ball.models import compare_aniso, fairness, pipeline   # noqa: E402

V2 = 2


def logger(path):
    f = open(path, "w")

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


log = logger(HERE / "train_all.log")
pipeline.aniso_data(log)
for seed in fairness.SEEDS:
    for fam in models.names():
        log(f"--- {fam} seed {seed}")
        models.train(fam, seed, log=logger(HERE / f"train_{fam}_v2_seed{seed}.log"),
                     interface=V2)
pipeline._ANISO_CACHE.clear()

iso = models.load("nflow", 0, name="network")
bes = {f"{fam}_v2_seed{seed}": models.load(fam, seed, interface=V2, iso=iso)
       for seed in fairness.SEEDS for fam in models.names()}
res = compare_aniso.run(bes, log=logger(HERE / "compare.log"))
(HERE / "compare.json").write_text(json.dumps(res, indent=1, default=str))
