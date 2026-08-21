#!/usr/bin/env python3
"""Generate the single-cell transmission dataset the boundary model trains on.

ONE physical experiment, repeated many times:

    a particle enters a square, purely scattering cell of optical width W
    through the left face, at height xi, travelling in direction Omega_in.
    Monte Carlo walks it until it leaves.  Record where it left (p), which
    way it was going (Omega_out) and how far it travelled (s).

That is all the boundary model ever needs to learn, and nothing about any
larger geometry enters here.  A cell is fully described by its optical
width W = pitch * sigma_s, so we sample W directly on a log grid rather
than deriving it from some particular problem's materials.

The conditioning space is 4-dimensional:

    W        optical width of the cell        log grid, --n-w values
    xi       entry height, 0 to 1             uniform
    Omega_in entry direction on the unit disk uniform, with Omega_x > 0

Sampling is UNIFORM over that space, not physical.  A generative model has
to be accurate everywhere it will be asked, and at solve time it gets asked
at whatever conditions the geometry produces -- so coverage matters more
than matching any one problem's natural distribution.

Each condition is sampled --per-cond times.  Coverage beats precision at
fixed cost: many conditions with few samples each trains better than few
conditions with many, because the flow learns the conditional map, not the
individual histograms.

Output: one npz with one row per sampled history.
    W, y0, oxi, oyi     the entry condition (y0 is xi)
    p, oxo, oyo, ozo, s the exit state
    k                   scattering count; k == 0 means the particle crossed
                        without colliding, which training drops (it is an
                        analytic Dirac component, see gmc/sampler.py)

Usage:
  python scripts/make_data.py                       # defaults, ~3.1M rows
  python scripts/make_data.py --n-cond 512 --per-cond 32 --out data/small.npz
"""
import argparse
import pathlib
import time

import numpy as np

import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import sample_single_cell            # noqa: E402


def sample_conditions(n_cond, rng):
    """Uniform entry conditions: height in [0,1], direction on the half-disk."""
    xi = rng.uniform(0.0, 1.0, n_cond)
    # uniform on the unit disk, then keep the Omega_x > 0 half
    r = np.sqrt(rng.uniform(0.0, 1.0, n_cond))
    th = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, n_cond)
    ox, oy = r * np.cos(th), r * np.sin(th)
    ox = np.maximum(ox, 1e-4)                  # strictly entering
    return xi, ox, oy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-min", type=float, default=0.025)
    ap.add_argument("--w-max", type=float, default=20.0)
    ap.add_argument("--n-w", type=int, default=16,
                    help="number of optical widths, log-spaced")
    ap.add_argument("--n-cond", type=int, default=4096,
                    help="entry conditions per optical width")
    ap.add_argument("--per-cond", type=int, default=48,
                    help="histories per entry condition")
    ap.add_argument("--seed", type=int, default=20250818)
    ap.add_argument("--out", default="data/singlecell.npz")
    args = ap.parse_args()

    Ws = np.geomspace(args.w_min, args.w_max, args.n_w)
    rng = np.random.default_rng(args.seed)
    total = args.n_w * args.n_cond * args.per_cond
    print(f"{args.n_w} optical widths from {args.w_min:g} to {args.w_max:g} "
          f"mfp\n{args.n_cond:,} conditions each x {args.per_cond} histories "
          f"= {total:,} rows")

    cols = {k: [] for k in ("W", "y0", "oxi", "oyi",
                            "p", "oxo", "oyo", "ozo", "s", "k")}
    t0 = time.time()
    for iw, W in enumerate(Ws):
        xi, ox, oy = sample_conditions(args.n_cond, rng)
        for j in range(args.n_cond):
            r = sample_single_cell(args.per_cond, W, W, mode="boundary",
                                   entry_pos=xi[j], entry_dir=(ox[j], oy[j]),
                                   seed=int(rng.integers(1, 2**31 - 1)),
                                   n_blocks=1)
            m = args.per_cond
            cols["W"].append(np.full(m, W))
            cols["y0"].append(np.full(m, xi[j]))
            cols["oxi"].append(np.full(m, ox[j]))
            cols["oyi"].append(np.full(m, oy[j]))
            cols["p"].append(r["p"])
            cols["oxo"].append(r["dir"][:, 0])
            cols["oyo"].append(r["dir"][:, 1])
            cols["ozo"].append(r["dir"][:, 2])
            cols["s"].append(r["s"])
            cols["k"].append(r["k"])
        el = time.time() - t0
        done = (iw + 1) / args.n_w
        print(f"  W = {W:7.3f} mfp   {done*100:5.1f}%   "
              f"{el:6.1f}s elapsed, {el/done - el:6.1f}s left", flush=True)

    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["W"] = out["W"].astype(np.float32)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)

    k = out["k"]
    print(f"\nwrote {path}  ({path.stat().st_size / 2**20:.1f} MiB)")
    print(f"  {len(k):,} rows, {int((k > 0).sum()):,} collided "
          f"({(k > 0).mean()*100:.1f}%) -- training uses the collided ones")
    print(f"  mean scatters per history {k.mean():.2f}")
    print(f"  total wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
