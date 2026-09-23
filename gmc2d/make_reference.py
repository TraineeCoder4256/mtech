#!/usr/bin/env python3
"""Build the frozen reference every model is scored against.  Run: python make_reference.py

For each scale in benchmark.SCALES this writes reference/lattice_x<scale>.npz:

  mc_*        a high-statistics mc.py solution -- the reference itself.
              Mean and standard error from independent runs (not from one
              run's internal spread), on the fine mesh and the 7x7 macro
              cells.  mc.py is the reference rather than OpenMC because it
              is available at every scale, needs no conda install, and has
              been shown to agree with OpenMC (check_benchmark.py).
  floor_cells K independent solves at exactly benchmark.N particles.  They
              give the NOISE FLOOR: how far an exact method with N particles
              lands from the reference purely by chance.  A model is scored
              as a ratio to this, so 1.0 means "as good as Monte Carlo can
              be at this particle count".
  omc_*       OpenMC's solution, copied in when openmc_lattice.py has been run
              at that scale.  Its parameters are checked against mc.lattice()
              before it is accepted.

and reference/manifest.json: what was run, with which seeds, on what
machine, which commit, and a hash of every file.

Takes about twenty minutes on 4 cores, mostly the thickest scale.  The
reference is committed; nobody needs to rerun this unless the benchmark
changes.
"""
import json
import os
import platform
import subprocess
import time

import numpy as np
import numba

import benchmark as B
import mc

# (independent runs, histories per run) per scale.  40M everywhere: with
# numba on 4 cores that is ~1 min at scale 1 and ~10 min at scale 20, and it
# puts the reference noise ~45x under the noise floor it is used to measure,
# so a model's score is its own error, not the reference's.
RUNS = {1: (100, 400_000), 4: (100, 400_000), 10: (100, 400_000), 20: (100, 400_000)}
K_FLOOR = 6                  # N-particle solves for the noise floor (15 pairs)


def mc_reference(prob, runs, per_run, scale):
    per_cm, d = prob["per_cm"], prob["L"] / prob["n"]
    fine, cell, absorbed, scat = [], [], [], 0.0
    t0 = time.time()
    for r in range(runs):
        phi, st = mc.solve(prob, per_run, seed=1_000_000 * int(scale) + 7919 * r)
        fine.append(phi)
        cell.append(B.coarsen(phi, per_cm))
        absorbed.append((prob["sig_a"] * phi).sum() * d * d)
        scat += st["scatters"]
    ms = lambda x: (np.mean(x, 0), np.std(x, 0, ddof=1) / np.sqrt(len(x)))
    (f, fse), (c, cse), (a, ase) = ms(fine), ms(cell), ms(absorbed)
    return {"mc_fine": f, "mc_fine_se": fse, "mc_cell": c, "mc_cell_se": cse,
            "mc_absorbed": a, "mc_absorbed_se": ase,
            "mc_histories": runs * per_run, "mc_runs": runs,
            "mc_scatters_per_particle": scat / (runs * per_run),
            "mc_wall_seconds": time.time() - t0}


def noise_floor(prob, scale):
    cells = np.array([B.coarsen(mc.solve(prob, B.N, seed=900_000_000 + 1000 * k
                                         + int(scale))[0], prob["per_cm"])
                      for k in range(K_FLOOR)])
    return {"floor_cells": cells, "floor_n": B.N}


def openmc_part(prob, scale):
    tag = "" if scale == 1 else f"_x{scale:g}"
    path = f"results/openmc_reference{tag}.npz"
    if not os.path.exists(path):
        return {}, f"no {path} (run: python openmc_lattice.py {scale:g})"
    o = np.load(path)
    bg = (prob["sig_s"][0, 0], prob["sig_a"][0, 0])          # a corner cell
    x, y = mc.ABSORBERS[0]
    k = prob["per_cm"]
    ab = (prob["sig_s"][y * k, x * k], prob["sig_a"][y * k, x * k])
    checks = {"scale": np.isclose(o["scale"], scale), "L": o["L"] == prob["L"],
              "pitch": o["pitch"] == prob["pitch"], "mesh": o["mesh"] == prob["n"],
              "background": np.allclose(o["bg"], bg), "absorber": np.allclose(o["ab"], ab),
              "layout": sorted(map(tuple, o["absorbers"].tolist())) == sorted(mc.ABSORBERS),
              "source": tuple(o["source_cell"].tolist()) == tuple(prob["source_cell"])}
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise ValueError(f"{path} does not match mc.lattice({scale}): {bad}")
    part = {f"omc_{k}": o[k] for k in ("fine", "fine_sd", "cell", "cell_sd",
                                       "absorbed", "absorbed_sd")}
    part.update(omc_histories=int(o["particles"]) * int(o["batches"]),
                omc_seed=int(o["seed"]), omc_version=str(o["openmc_version"]))
    return part, f"OpenMC {str(o['openmc_version'])}, {part['omc_histories']:,} histories"


def provenance():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                if l.startswith("model name")), platform.processor()) \
        if os.path.exists("/proc/cpuinfo") else platform.processor()
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "numba": numba.__version__}


def main():
    B.REFERENCE.mkdir(exist_ok=True)
    manifest = {"provenance": provenance(), "protocol": {
        "scales": B.SCALES, "N": B.N, "k_floor": K_FLOOR,
        "reference_seed": "1_000_000 * scale + 7919 * run",
        "floor_seed": "900_000_000 + 1000 * k + scale"}, "files": {}}

    for scale in B.SCALES:
        prob = B.problem(scale)
        runs, per_run = RUNS[scale]
        print(f"scale {scale:>3}: mc.py {runs} x {per_run:,} histories ...", flush=True)
        ref = mc_reference(prob, runs, per_run, scale)
        print(f"           {ref['mc_wall_seconds']:.0f} s, "
              f"{ref['mc_scatters_per_particle']:.1f} scatters per particle", flush=True)
        ref.update(noise_floor(prob, scale))
        omc, omc_note = openmc_part(prob, scale)
        ref.update(omc)
        print(f"           {omc_note}")

        f = ref["floor_cells"]
        rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        info = {"mc_histories": ref["mc_histories"],
                "mc_wall_seconds": round(ref["mc_wall_seconds"], 1),
                "noise_floor_vs_reference": [rel(x, ref["mc_cell"]) for x in f],
                "noise_floor_pairwise": [rel(f[i], f[j]) for i in range(len(f))
                                         for j in range(i)],
                "openmc": omc_note}
        if omc:
            info["mc_vs_openmc_cells"] = B.agreement(ref["mc_cell"], ref["mc_cell_se"],
                                                     ref["omc_cell"], ref["omc_cell_sd"])
            info["mc_vs_openmc_fine"] = B.agreement(ref["mc_fine"], ref["mc_fine_se"],
                                                    ref["omc_fine"], ref["omc_fine_sd"], 0.2)
            a = info["mc_vs_openmc_cells"]
            print(f"           vs OpenMC, 7x7: chi2/cell {a['chi2_per_cell']:.2f}, "
                  f"max |z| {a['max_abs_z']:.2f}, median diff {a['median_rel_diff']*100:.2f}%")
        print(f"           noise floor at N={B.N:,}: "
              f"{np.mean(info['noise_floor_vs_reference'])*100:.3f}% vs reference")

        path = B.reference_path(scale)
        np.savez_compressed(path, definition=json.dumps(B.definition(scale)), **ref)
        info["sha256"] = B.sha256(path)
        info["bytes"] = path.stat().st_size
        manifest["files"][path.name] = info

    (B.REFERENCE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(v["bytes"] for v in manifest["files"].values())
    print(f"\nwrote {len(manifest['files'])} reference files ({total / 2**20:.2f} MiB) "
          f"and {B.REFERENCE}/manifest.json")


if __name__ == "__main__":
    main()
