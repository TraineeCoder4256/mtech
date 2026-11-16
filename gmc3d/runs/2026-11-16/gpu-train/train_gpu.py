"""Train the v2 models (interface v2, anisotropic balls) on a GPU.

Since v2.2.0 this is the same as `python run.py train --interface 2
--family all --device cuda --seeds 0 1`; kept so the commands given on
16 Nov still work.  v2.1.10 patched a copy of fit_loop in here; m2.3 put a
`device` argument in ball/models/ itself, so the patch is gone.

    python runs/2026-11-16/gpu-train/train_gpu.py --seeds 0 1
    python runs/2026-11-16/gpu-train/train_gpu.py --seeds 0 --families cvae \
        --precision fp16 --compile
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import torch                                                # noqa: E402

from ball import models                                     # noqa: E402
from ball.models import pipeline                            # noqa: E402

DEVICE = "cuda"


def logger(path):
    f = open(path, "w")

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def main():
    global DEVICE
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--families", nargs="+", default=["nflow", "cfm", "cvae"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=None,
                   help="default: the shared budget (24,000)")
    p.add_argument("--precision", default="fp32",
                   choices=["fp32", "tf32", "fp16"])
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()
    DEVICE = a.device
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("torch sees no GPU: install a CUDA build of PyTorch")
    pipeline.aniso_data(logger(HERE / "data.log"))
    for s in a.seeds:
        for fam in a.families:
            models.train(fam, s, steps=a.steps, interface=2, device=DEVICE,
                         precision=a.precision, compile=a.compile,
                         log=logger(HERE / f"train_{fam}_v2_seed{s}.log"))


if __name__ == "__main__":
    main()
