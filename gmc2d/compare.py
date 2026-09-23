#!/usr/bin/env python3
"""Put every evaluated model side by side.  Run: python compare.py [model ...]

Reads only results/<model>/<timestamp>/metrics.json and provenance.json,
which run.py writes, and takes the LATEST run of each model.  It never
re-runs anything, so it is instant and it compares exactly the numbers in
each model's own report.

Accuracy numbers are comparable across any runs: every model was scored
against the same frozen reference (the reference manifest hash is checked).
Speed numbers are only comparable between runs on the same hardware, so
the table says which machine each speed came from and warns if they differ.

Writes results/comparison.txt and results/comparison.pdf.
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = pathlib.Path("results")


def latest_runs(names=None):
    runs = {}
    for m in sorted(RESULTS.glob("*/*/metrics.json")):   # timestamps sort in time order
        name = m.parent.parent.name
        if names and name not in names:
            continue
        runs[name] = m.parent                              # later ones overwrite
    return runs


def load(run):
    met = json.loads((run / "metrics.json").read_text())
    prov = json.loads((run / "provenance.json").read_text())
    return met, prov


def machine(prov):
    h = prov["hardware"]
    return f"{h['cpu']} x{h['cores']} {h['device']}" + (f" {h['gpu']}" if h.get("gpu") else "")


def main():
    runs = latest_runs(set(sys.argv[1:]) or None)
    if not runs:
        sys.exit("no results/<model>/<timestamp>/metrics.json yet -- run: python run.py <model>")
    data = {n: load(r) for n, r in runs.items()}
    # the reference cheapest to explain first, then models by accuracy
    order = sorted(data, key=lambda n: (n not in ("oracle", "free"), n != "oracle",
                                        data[n][0]["lattice"]["1"]["ratio"]))

    scales = list(data[order[0]][0]["lattice"])
    refs = {p["reference_manifest_sha256"] for _, p in data.values()}
    machines = {n: machine(p) for n, (_, p) in data.items()}

    head = (f"{'model':<10} {'nfe':>4} {'params':>9} "
            + " ".join(f"{'x' + s:>7}" for s in scales)
            + f" {'bias%':>7} {'chi2':>7} {'sliced':>7} {'C2ST%':>6} {'clamp%':>7}"
            + f" {'us/cross':>9} {'speedup':>9} {'b-even':>8}")
    lines = []
    for n in order:
        met, prov = data[n]
        lat, cost = met["lattice"], met["cost"]
        cells = met["cells"]
        sliced = np.mean([c["sliced_w1"][0] for c in cells])
        c2st = np.mean([c["c2st_model"][0] for c in cells]) * 100
        params = met["model"].get("params")
        lines.append(
            f"{n:<10} {met['model']['nfe']:>4} {params if params else '-':>9} "
            + " ".join(f"{lat[s]['ratio']:>6.2f}x" for s in scales)
            + f" {lat['1']['bias']*100:>7.3f} {lat['1']['cell_chi2']:>7.2f}"
            + f" {sliced:>7.2f} {c2st:>6.1f} {cost['clamped_frac']*100:>7.2f}"
            + f" {cost['t_per_crossing_gmc']*1e6:>9.2f} {cost['speedup']:>8.4g}x"
            + f" {cost['breakeven_scatters']:>8,.0f}")
    floor_c2st = np.mean([c["c2st_floor"][0] for c in data[order[0]][0]["cells"]]) * 100

    warn = []
    if len(refs) > 1:
        warn.append("! runs were scored against DIFFERENT reference manifests; "
                    "accuracy columns are not comparable. Re-run the older ones.")
    if len(set(machines.values())) > 1:
        warn.append("! speed columns come from different machines:")
        warn += [f"    {n:<10} {m}" for n, m in machines.items()]
    if "free" in data:
        warn.append("  free's accuracy columns are meaningless by design (it draws noise); "
                    "read only its cost columns")

    text = "\n".join([
        "MODEL COMPARISON  (latest run of each; accuracy vs the frozen reference)",
        "=" * len(head),
        head, "-" * len(head), *lines, "-" * len(head),
        f"x<scale>  lattice error as a multiple of the Monte Carlo noise floor "
        f"(1.00 = as good as MC)",
        "bias%    systematic error at scale 1 with the noise averaged out; "
        "chi2 = per-cell bias test (~1 is clean)",
        "sliced   joint exit-state distance, x the MC floor, mean over the cell shapes",
        f"C2ST%    classifier accuracy telling model from MC (50 = cannot; MC vs MC "
        f"scores {floor_c2st:.1f})",
        "us/cross time per macro-cell crossing at scale 1; speedup = MC time / GMC "
        "time at scale 1",
        "b-even   scattering events one crossing must replace before GMC wins",
        *warn,
        "",
        "runs:", *[f"  {n:<10} {runs[n]}  ({machines[n]})" for n in order],
    ])
    (RESULTS / "comparison.txt").write_text(text + "\n")
    print(text)

    # accuracy against cost: where each model sits, and where the goalposts are
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for n in order:
        met = data[n][0]
        x = met["cost"]["t_per_crossing_gmc"] * 1e6
        y = met["lattice"]["1"]["ratio"]
        ax.scatter(x, y, s=40, zorder=3, marker="x" if n == "free" else "o")
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 1.03, " Monte Carlo noise floor", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time per macro-cell crossing (us), scale 1")
    ax.set_ylabel("lattice error / MC noise floor, scale 1")
    ax.set_title("accuracy against cost: down and left is better", fontsize=10)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.pdf")
    plt.close(fig)
    print(f"\nwrote {RESULTS}/comparison.txt and comparison.pdf")


if __name__ == "__main__":
    main()
