#!/usr/bin/env python3
"""Which exit coordinate is the model actually worst at?

The aggregate tests (sliced Wasserstein, C2ST) say whether the 5-D joint is
right, not which part of it is wrong.  This breaks the error down per
coordinate, each as a 1-D Wasserstein distance against the MC-vs-MC floor at
matched sample size, so the numbers are ratios to sampling noise and are
comparable across coordinates with very different scales.

Run it with two checkpoints to see where an encoding change actually landed:

  python scripts/perdim_error.py --ckpts models/boundary_v1 models/boundary_v2

Reading it: a ratio of 1 is indistinguishable from noise.  The coordinate
with the largest ratio is the one to fix; if a change improves four
coordinates and wrecks the fifth, that shows up here and nowhere else.
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_sparam import held_out_conditions, clouds    # noqa: E402
from gmc_end_to_end import load_sampler                   # noqa: E402
from joint_whole import w1_1d                             # noqa: E402

LABELS = ["p/4W", "Omega_x", "Omega_y", "Omega_z", "log10 s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--ckpts", nargs="+",
                    default=["models/boundary_v1", "models/boundary_v2"])
    ap.add_argument("--data", default="data/lattice_singlecell.npz")
    ap.add_argument("--ode-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    sam = {}
    for c in args.ckpts:
        s = load_sampler(ROOT / c, ode_steps=args.ode_steps,
                         device=args.device)
        sam[s.s_param] = s
        print(f"{c}: s_param = {s.s_param}")
    d = np.load(ROOT / args.data)

    for w in args.w:
        conds = held_out_conditions(d, w)
        res = {sp: clouds(s, d, w, conds, args.seed)[:2]
               for sp, s in sam.items()}
        R = next(iter(res.values()))[0]
        rng = np.random.default_rng(0)
        h = rng.permutation(len(R))
        half = len(h) // 2
        A, B = h[:half], h[half:2 * half]

        print(f"\n=== W~ = {w:g} mfp, {len(R):,} matched samples ===")
        print(f"{'coordinate':>11} {'MC floor':>10} " +
              " ".join(f"{k:>10}" for k in sam) + "  |  " +
              " ".join(f"{k + ' /floor':>13}" for k in sam))
        for j, lab in enumerate(LABELS):
            sd = R[:, j].std() + 1e-12
            floor = w1_1d(R[A, j], R[B, j]) / sd
            vals = [w1_1d(R[A, j], res[sp][1][A, j]) / sd for sp in sam]
            print(f"{lab:>11} {floor:>10.4f} " +
                  " ".join(f"{v:>10.4f}" for v in vals) + "  |  " +
                  " ".join(f"{v / floor:>13.2f}" for v in vals))
        print("  (W1 normalised by the MC standard deviation of that "
              "coordinate; floor is MC vs MC at the same sample size)")


if __name__ == "__main__":
    main()
