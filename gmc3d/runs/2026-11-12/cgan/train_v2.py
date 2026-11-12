"""cGAN at interface v2 (k >= 2 rows, 24,000 steps), one seed per
process (the background limit is 2 h).

    python runs/2026-11-12/cgan/train_v2.py <seed>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from ball import models                                     # noqa: E402

seed = int(sys.argv[1])
f = open(HERE / f"train_cgan_v2_seed{seed}.log", "w")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


models.train("cgan", seed, interface=2, log=log)
log("FINISHED")
