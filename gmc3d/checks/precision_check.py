"""What lower precision costs in accuracy, ball by ball.

Used by: run by hand (PyTorch and NumPy only, no OpenMC).  From gmc3d/:

    python -m checks.precision_check
    python -m checks.precision_check --device cuda

Uses: checks/ball_metrics.py (the scoring), checks/net_variants.py (the
candidate samplers), ball/network.py and ball/walk.py.  Changes nothing.

checks/net_variants.py measures what float16 and bfloat16 BUY (1.5-2x on a
CPU, more on a GPU's tensor cores).  This file measures what they COST: the
same ball-by-ball comparison against the exact walk that
`python run.py ballcheck` runs, for the network in float32, float16 and
bfloat16, plus the compiled float32 version (which should be identical to
within rounding).

Read the weight-factor columns first.  They are E[exp(-kappa s)], the share
of a particle's weight that survives a ball, and they are where a small
error in the path distribution turns into a biased answer -- the same place
gmc2d's absorber tail went wrong.  A precision is only worth using if its
errors there are no larger than float32's.
"""

import argparse

import numpy as np
import torch

from ball import data as D
from ball import network as N
from . import ball_metrics
from . import net_variants as V


class Wrapped:
    """A backend that draws with a given sampler instead of net.sample."""

    def __init__(self, be, fn, name):
        self.be, self.fn, self.name = be, fn, name
        self.r_min, self.r_max = be.r_min, be.r_max

    @torch.no_grad()
    def sample(self, R, seeds=None, idx=None):
        R = np.asarray(R, np.float64)
        c = torch.from_numpy(self.be.cnorm(np.log(R)[:, None]).astype(
            np.float32)).to(self.be.device)
        z = torch.randn(R.size, 2, generator=self.be.gen,
                        device=self.be.device)
        y = self.fn(z, c).float().cpu().numpy().astype(np.float64)
        return D.decode(R, self.be.ynorm.undo(y))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--which", nargs="*",
                   default=["knots", "compiled", "half", "bfloat16"])
    a = p.parse_args()

    be, trained = V.load(a.device)
    if not trained:
        raise SystemExit("needs the trained network: python run.py train")
    name = torch.cuda.get_device_name(be.device) \
        if be.device.type == "cuda" else "CPU"
    print(f"device: {name}; torch {torch.__version__}; "
          f"{a.n:,} draws per radius\n")

    variants = V.build(be.net, be.device, a.which)
    for label, fn in variants.items():
        rows = ball_metrics.score(Wrapped(be, fn, label), n=a.n)
        print(ball_metrics.report(rows, f"network, {label}:"))
        print()


if __name__ == "__main__":
    main()
