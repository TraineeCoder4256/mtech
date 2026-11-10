"""m2.1: train every family at interface v2 on the k >= 2 rows, one seed
per process (the background limit is 2 h; three models take ~1.2 h).

    python runs/2026-11-10/models-v2.1/train_seed.py <seed>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from ball import models                                     # noqa: E402
from ball.models import pipeline                            # noqa: E402

seed = int(sys.argv[1])


def logger(path):
    f = open(path, "w")

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


pipeline.aniso_data(logger(HERE / f"data_seed{seed}.log"))
for fam in models.names():
    models.train(fam, seed, interface=2,
                 log=logger(HERE / f"train_{fam}_v2_seed{seed}.log"))
