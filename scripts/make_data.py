#!/usr/bin/env python3
"""Generate the single-cell transmission dataset.  Run: python scripts/make_data.py

One experiment, repeated: a particle enters a square scattering cell of
optical width W through the left face at height xi in direction Omega_in;
Monte Carlo walks it until it leaves; record where it left (p), which way it
was going (Omega_out), how far it went (s), and how many times it scattered
(k).  No larger geometry is involved -- a cell is fully described by its
optical width, so W is sampled directly.

Conditions are sampled uniformly rather than physically, because the model
must be accurate everywhere the geometry might ask it, not just where a
particular problem happens to send particles.
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import sample_single_cell            # noqa: E402

# ---- settings ----------------------------------------------------------
W_MIN, W_MAX, N_W = 0.025, 20.0, 16     # optical widths, log-spaced
N_COND = 4096                           # entry conditions per width
PER_COND = 48                           # histories per condition
SEED = 20250818
OUT = ROOT / "data" / "singlecell.npz"

FIELDS = ("W", "y0", "oxi", "oyi", "p", "oxo", "oyo", "ozo", "s", "k")


def main():
    Ws = np.geomspace(W_MIN, W_MAX, N_W)
    rng = np.random.default_rng(SEED)
    print(f"{N_W} widths {W_MIN:g}-{W_MAX:g} mfp x {N_COND:,} conditions "
          f"x {PER_COND} histories = {N_W * N_COND * PER_COND:,} rows")

    cols = {k: [] for k in FIELDS}
    t0 = time.time()
    for iw, W in enumerate(Ws):
        # uniform entry height, and uniform on the half-disk of directions
        xi = rng.uniform(0.0, 1.0, N_COND)
        r = np.sqrt(rng.uniform(0.0, 1.0, N_COND))
        th = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, N_COND)
        ox = np.maximum(r * np.cos(th), 1e-4)     # strictly entering
        oy = r * np.sin(th)

        for j in range(N_COND):
            out = sample_single_cell(PER_COND, W, W, entry_pos=xi[j],
                                     entry_dir=(ox[j], oy[j]), n_blocks=1,
                                     seed=int(rng.integers(1, 2**31 - 1)))
            for name, v in (("W", W), ("y0", xi[j]),
                            ("oxi", ox[j]), ("oyi", oy[j])):
                cols[name].append(np.full(PER_COND, v))
            cols["p"].append(out["p"])
            cols["s"].append(out["s"])
            cols["k"].append(out["k"])
            for i, name in enumerate(("oxo", "oyo", "ozo")):
                cols[name].append(out["dir"][:, i])

        done = (iw + 1) / N_W
        el = time.time() - t0
        print(f"  W = {W:7.3f} mfp   {done*100:5.1f}%   "
              f"{el:5.1f}s elapsed, {el/done - el:5.1f}s left", flush=True)

    data = {k: np.concatenate(v) for k, v in cols.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **data)

    k = data["k"]
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 2**20:.1f} MiB)")
    print(f"  {len(k):,} rows, {int((k > 0).sum()):,} collided "
          f"({(k > 0).mean()*100:.1f}%) -- training uses those")
    print(f"  mean scatters per history {k.mean():.2f}")


if __name__ == "__main__":
    main()
