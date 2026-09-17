#!/usr/bin/env python3
"""Step 1: the training dataset.  Run: python make_data.py

One experiment, repeated: a particle enters a RECTANGULAR scattering cell
of optical size W x H through the left face at height xi, travelling in
direction Omega_in; Monte Carlo walks it until it leaves; record where it
left, which way it was going, and how far it went.

No geometry is involved -- a cell is fully described by its optical size,
so W and H are drawn directly.  Both vary independently, as the paper's
conditioning requires; square cells are just the case W = H, and the
lattice used for the end-to-end test is one instance of that.

W, H, xi and Omega_in are drawn CONTINUOUSLY and uniformly over their
ranges rather than from a grid.  The model must be accurate wherever a
geometry asks it, and continuous sampling also means evaluation conditions
are genuinely unseen.
"""
import pathlib
import time

import numpy as np

import mc

# ---- settings ----------------------------------------------------------
SIZE_MIN, SIZE_MAX = 0.05, 20.0    # optical size range for both W and H
ASPECT_MAX = 8.0                   # cap on W/H, so cells stay sane
N_COND = 60000                     # entry conditions
PER_COND = 48                      # histories per condition
SEED = 20250818
OUT = pathlib.Path("data/cells.npz")

FIELDS = ("W", "H", "xi", "ox_in", "oy_in", "p", "ox", "oy", "oz", "s", "k")


def draw_conditions(n, rng):
    """Log-uniform sizes with a bounded aspect ratio; uniform entry state."""
    W = np.exp(rng.uniform(np.log(SIZE_MIN), np.log(SIZE_MAX), n))
    H = np.exp(rng.uniform(np.log(SIZE_MIN), np.log(SIZE_MAX), n))
    ratio = np.clip(W / H, 1 / ASPECT_MAX, ASPECT_MAX)
    H = W / ratio
    xi = rng.uniform(0.0, 1.0, n)
    # uniform on the half-disk of directions entering the left face
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, n)
    return W, H, xi, np.maximum(r * np.cos(th), 1e-4), r * np.sin(th)


def main():
    rng = np.random.default_rng(SEED)
    W, H, xi, ox, oy = draw_conditions(N_COND, rng)
    print(f"{N_COND:,} cells, sizes {SIZE_MIN:g}-{SIZE_MAX:g} mfp, "
          f"aspect up to {ASPECT_MAX:g}:1")
    print(f"{PER_COND} histories each = {N_COND * PER_COND:,} rows")

    cols = {k: [] for k in FIELDS}
    t0 = time.time()
    for j in range(N_COND):
        out = mc.sample_cell(PER_COND, W[j], H[j], xi=xi[j],
                             direction=(ox[j], oy[j]), n_blocks=1,
                             seed=int(rng.integers(1, 2**31 - 1)))
        for name, v in (("W", W[j]), ("H", H[j]), ("xi", xi[j]),
                        ("ox_in", ox[j]), ("oy_in", oy[j])):
            cols[name].append(np.full(PER_COND, v))
        cols["p"].append(out["p"])
        cols["s"].append(out["s"])
        cols["k"].append(out["k"])
        for i, name in enumerate(("ox", "oy", "oz")):
            cols[name].append(out["dir"][:, i])
        if (j + 1) % 10000 == 0:
            el = time.time() - t0
            print(f"  {j+1:,}/{N_COND:,}   {el:5.1f}s elapsed, "
                  f"{el * N_COND / (j+1) - el:5.1f}s left", flush=True)

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
