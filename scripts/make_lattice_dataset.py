#!/usr/bin/env python3
"""Generate a GMC single-cell transmission dataset from lattice cells.

Each record is one transmission through a single lattice cell:

  conditioning : W (cell optical size, mfp), y0 (entry height fraction),
                 Omega_x_in, Omega_y_in  (entry point on the unit disk)
  target       : p_exit (perimeter coordinate), Omega_{x,y,z}_out,
                 s (path length, mfp), k (scattering events)
  provenance   : sig_s, sig_a, pitch, K of the lattice cell it came from

Why the axes collapse
---------------------
The single-cell walk is pure scattering with sigma_s = 1 in optical units,
so a lattice cell of physical size ``pitch`` and cross section ``sig_s``
enters only through the product

    W = pitch * sig_s        (cell optical size, mean free paths)

``pitch`` and ``sig_s`` are therefore *not* independent axes: sweeping either
one sweeps W.  We deduplicate on W so no compute is wasted regenerating the
same distribution twice, while still recording every (pitch, sig_s) pair that
produced it.

``sig_a`` does not enter the walk at all -- the framework recovers absorbing
problems by attenuating the returned path length, exp(-sig_a * s_phys) -- so
it is carried as provenance for the consumer to apply.

Similarly the lattice size K does not change any single-cell distribution: a
K=3 and a K=11 lattice with the same pitch and materials contain identical
cells.  K enters only through *composition* -- how many background and
absorber cells a lattice of that size holds -- which is written to the
companion ``*_composition.npz`` so records can be reweighted per lattice size.

Usage:
  python scripts/make_lattice_dataset.py [--k 3 5 7 9 11]
        [--pitch 0.25 0.5 1 2 4] [--materials paper scatter_heavy absorb_heavy]
        [--n-cond 4096] [--per-cond 48] [--out data/lattice_singlecell.npz]

Choosing --n-cond vs --per-cond
-------------------------------
Their product fixes the record count, but the split matters.  Few entry
states with many samples each resolves the conditional distribution
precisely at a handful of points; many entry states with fewer samples
covers the (y0, Omega_x, Omega_y) conditioning space densely.  A conditional
generative sampler needs the latter, so the defaults favour coverage.

The difference is measurable at equal cost: going from 192x1024 to 4096x48
raises the effective sample size of the cosine reweighting from ~67 to ~1058
and cuts the reweighted <s> error from 4.8%/14.2% (median/max) to 1.3%/3.0%.
It also matters for the uncollided population -- at small W most particles
cross without scattering, so their exit direction *is* their entry direction,
and few entry states turn that into a handful of delta spikes rather than a
usable distribution.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc2d import sample_single_cell  # noqa: E402
from mc2d.lattice import absorber_cells  # noqa: E402

# (sig_s, sig_a) for the background and absorber cells of a lattice.
MATERIAL_SETS = {
    "paper":         {"background": (1.0, 0.0), "absorber": (0.5, 9.5)},
    "scatter_heavy": {"background": (1.0, 0.0), "absorber": (5.0, 5.0)},
    "absorb_heavy":  {"background": (1.0, 0.0), "absorber": (0.1, 9.9)},
}


def cell_table(ks, pitches, material_sets):
    """Every (K, pitch, material, kind) lattice cell type, with its W."""
    rows = []
    for k in ks:
        n_abs = len(absorber_cells(k))
        counts = {"absorber": n_abs, "background": k * k - n_abs}
        for mname in material_sets:
            for kind, (ss, sa) in MATERIAL_SETS[mname].items():
                for pitch in pitches:
                    rows.append({"K": k, "pitch": pitch, "material": mname,
                                 "kind": kind, "sig_s": ss, "sig_a": sa,
                                 "W": pitch * ss, "count": counts[kind]})
    return rows


def sample_entry_states(n, rng):
    """Uniform over the incoming half-disk {Omega_x > 0, |Omega_xy| < 1}.

    Uniform coverage of the conditioning space beats physical (cosine)
    weighting for training a conditional sampler: the network must be
    accurate at grazing incidence too, which cosine weighting starves.

    Caveat for consumers: converting these samples back to a physical
    cosine-law ensemble needs the importance weight

        w = Omega_x / |Omega_z| = Omega_x / sqrt(1 - Omega_x^2 - Omega_y^2)

    which diverges as the entry point approaches the rim of the disk
    (grazing incidence, Omega_z -> 0).  The weights are heavy-tailed: at 192
    entry states per W the effective sample size is only ~30-80, so
    reweighted estimates carry ~10% scatter even though the underlying
    samples are exact.  Raise --n-cond if you intend to reweight rather than
    train conditionally.
    """
    y0 = rng.random(n)
    ox = np.empty(n)
    oy = np.empty(n)
    filled = 0
    while filled < n:
        a = rng.random(2 * (n - filled)) * 2.0 - 1.0
        cx, cy = a[::2], a[1::2]
        ok = (cx > 0.0) & (cx * cx + cy * cy < 1.0)
        take = min(int(ok.sum()), n - filled)
        ox[filled:filled + take] = cx[ok][:take]
        oy[filled:filled + take] = cy[ok][:take]
        filled += take
    return y0, ox, oy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    ap.add_argument("--pitch", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--materials", nargs="+", default=list(MATERIAL_SETS),
                    choices=list(MATERIAL_SETS))
    ap.add_argument("--n-cond", type=int, default=4096,
                    help="entry states sampled per cell optical size")
    ap.add_argument("--per-cond", type=int, default=48,
                    help="transmissions per entry state")
    ap.add_argument("--seed", type=int, default=20250818)
    ap.add_argument("--out", default="data/lattice_singlecell.npz")
    args = ap.parse_args()

    rows = cell_table(args.k, args.pitch, args.materials)

    # Deduplicate on W (rounded to kill float noise); keep provenance.
    by_w = {}
    for r in rows:
        by_w.setdefault(round(r["W"], 12), []).append(r)
    ws = sorted(by_w)
    print(f"{len(rows)} lattice cell types -> {len(ws)} distinct optical "
          f"sizes W in [{ws[0]:g}, {ws[-1]:g}] mfp")

    rng = np.random.default_rng(args.seed)
    cols = {n: [] for n in ("W", "y0", "oxi", "oyi", "p", "oxo", "oyo",
                            "ozo", "s", "k", "sig_s", "sig_a", "pitch", "K")}
    t0 = time.time()

    for wi, w in enumerate(ws):
        y0s, oxs, oys = sample_entry_states(args.n_cond, rng)
        # One representative (sig_s, sig_a, pitch, K) per W for provenance;
        # the full mapping lives in the composition file.
        rep = by_w[w][0]
        for c in range(args.n_cond):
            r = sample_single_cell(
                args.per_cond, w, w, mode="boundary",
                entry_pos=float(y0s[c]),
                entry_dir=(float(oxs[c]), float(oys[c])),
                seed=int(rng.integers(1, 2**31 - 1)), n_blocks=8)
            m = args.per_cond
            cols["W"].append(np.full(m, w))
            cols["y0"].append(np.full(m, y0s[c]))
            cols["oxi"].append(np.full(m, oxs[c]))
            cols["oyi"].append(np.full(m, oys[c]))
            cols["p"].append(r["p"])
            cols["oxo"].append(r["dir"][:, 0])
            cols["oyo"].append(r["dir"][:, 1])
            cols["ozo"].append(r["dir"][:, 2])
            cols["s"].append(r["s"])
            cols["k"].append(r["k"])
            cols["sig_s"].append(np.full(m, rep["sig_s"]))
            cols["sig_a"].append(np.full(m, rep["sig_a"]))
            cols["pitch"].append(np.full(m, rep["pitch"]))
            cols["K"].append(np.full(m, rep["K"]))
        done = (wi + 1) * args.n_cond * args.per_cond
        print(f"  W={w:9.4g} mfp  {done:>9d} records  "
              f"{time.time()-t0:6.1f} s", flush=True)

    out = {}
    for name, chunks in cols.items():
        a = np.concatenate(chunks)
        out[name] = a.astype(np.int32) if name in ("k", "K") \
            else a.astype(np.float32)

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **out)
    n = len(out["W"])
    print(f"wrote {outp}  ({n} records, {outp.stat().st_size/1e6:.1f} MB)")

    # Composition table: how many cells of each W a lattice of size K holds.
    comp = np.array([[r["K"], r["pitch"], r["sig_s"], r["sig_a"], r["W"],
                      r["count"]] for r in rows], dtype=np.float64)
    compp = outp.with_name(outp.stem + "_composition.npz")
    np.savez_compressed(
        compp, table=comp,
        columns=np.array(["K", "pitch", "sig_s", "sig_a", "W", "count"]),
        materials=np.array(args.materials))
    print(f"wrote {compp}  ({len(comp)} cell types)")


if __name__ == "__main__":
    main()
