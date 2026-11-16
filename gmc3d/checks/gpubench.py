"""Everything the GPU changes, measured in one go (v2.2.1).

Used by: run.py (`python run.py gpubench --device cuda`).  Run it on the
GPU machine, from gmc3d/, with the accepted weights in data/models/
(nflow_seed0 and cvae_v2_seed0) and OpenMC installed.

Five parts, each written to its own file in the output folder:

 1. env.json        the machine: GPU, CUDA, torch, CPU cores.
 2. tests.log       the GPU unit tests (tests/test_models_m23.py, m24):
                    training and inference on cuda, which no container of
                    ours could run.
 3. models/         Development's model bench (python -m ball.models.bench):
                    cost per draw of every family, v1 and v2, at 1k to 1M
                    draws per call, for every option (compile, CUDA graphs,
                    fp16), and how far fp16 moves the draws.
 4. gate2_*.log     the acceptance gate (343 checks) for the default v2
                    model with the fastest options: fp16 and graphs may only
                    be used if it still passes, 0 of 343.
 5. race.log/json   whole problems against OpenMC, same session, `seeds`
                    seeds, median FOM.  Contenders:
                      OpenMC analog, OpenMC implicit
                      ours plain MC, ours + exact walk (CPU)
                      network on the GPU:
                        fp32                      (what v2.1.9 did)
                        fp32, 8 batches in flight (bigger calls)
                        fp32, 8 in flight + overlap
                        graphs + fp16, 8 in flight + overlap  (everything)
                    each at R* = 2 (a draw on a GPU costs about a flight).

The isotropic race problems use the v1 network (nflow seed 0) for their
balls; the v2 model is checked by the gate.  Quoting rules as ever: same
session, median of the seeds, about 15% on any speed ratio, say which
OpenMC mode.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from core import output
from . import race

PROBLEMS = ("sphere", "slab", "cask", "curved", "tilted")
FLY = 8


def _env(device):
    import torch
    e = {"python": sys.version.split()[0], "torch": torch.__version__,
         "cuda_available": torch.cuda.is_available(),
         "cpu": platform.processor() or platform.machine(),
         "cpu_cores": __import__("os").cpu_count()}
    if torch.cuda.is_available():
        e.update(gpu=torch.cuda.get_device_name(0),
                 cuda=torch.version.cuda,
                 gpu_memory_gb=round(torch.cuda.get_device_properties(0)
                                     .total_memory / 2 ** 30, 1))
    try:
        import openmc
        e["openmc"] = openmc.__version__
    except ImportError:
        e["openmc"] = None
    return e


def _sub(cmd, path, log):
    t = time.perf_counter()
    with open(path, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    tail = Path(path).read_text().strip().splitlines()[-1:] or [""]
    log(f"  {' '.join(cmd[1:])}: exit {rc}, {time.perf_counter() - t:.0f} s"
        f"  [{tail[0][:100]}]")
    return rc


def _median_rows(per_seed):
    """{problem: [rows]} per seed -> the same with the median of each
    row's seconds and FOMs (answers from the first seed)."""
    out = {}
    for name in per_seed[0]:
        rows = []
        for i, r0 in enumerate(per_seed[0][name]):
            rs = [ps[name][i] for ps in per_seed]
            r = dict(r0)
            r["seconds"] = float(np.median([x["seconds"] for x in rs]))
            r["fom"] = {k: float(np.median([x["fom"][k] for x in rs]))
                        for k in r0["fom"]}
            r["seconds_all"] = [x["seconds"] for x in rs]
            rows.append(r)
        out[name] = rows
    return out


def run(out, device="cuda", problems=PROBLEMS, seeds=(11, 12, 13),
        particles=100_000, batches=40, parts=(1, 2, 3, 4, 5), load=None,
        log=print):
    """load(name, device, **opts) -> a backend (run.py's, with options)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    res = {}
    if 1 in parts:
        res["env"] = _env(device)
        (out / "env.json").write_text(json.dumps(res["env"], indent=1))
        log(f"1. machine: {res['env']}")
    if 2 in parts:
        log("2. GPU unit tests")
        res["tests_rc"] = _sub([py, "-m", "pytest", "-q",
                                "tests/test_models_m23.py",
                                "tests/test_models_m24.py"],
                               out / "tests.log", log)
    if 3 in parts:
        log("3. model bench (every family and option)")
        res["bench_rc"] = _sub([py, "-m", "ball.models.bench", "--device",
                                device, "--out", str(out / "models")],
                               out / "models_bench.log", log)
    if 4 in parts:
        log("4. gate2 for the default v2 model, plain and fastest")
        for tag, extra in (("fp32", []),
                           ("graphs_fp16", ["--precision", "fp16",
                                            "--graphs"])):
            _sub([py, "run.py", "gate2", "cvae_v2_seed0", "--device", device,
                  "--out", str(out / f"gate2_{tag}")] + extra,
                 out / f"gate2_{tag}.log", log)
    if 5 in parts:
        log(f"5. race, {len(seeds)} seeds, {particles:,} x {batches} "
            f"histories, problems {', '.join(problems)}")
        backends = {"gpu fp32": load("network", device),
                    "oracle": load("oracle", device)}
        if device != "cpu":             # fp16 and graphs need a GPU
            backends["gpu graphs+fp16"] = load("network", device,
                                               precision="fp16", graphs=True)
        fly = {"in_flight": FLY}
        both = {"in_flight": FLY, "overlap": True}
        contenders = (
            ("ours, plain MC", None, None),
            ("ours + exact walk (CPU)", "oracle", 2.0),
            ("GPU fp32", "gpu fp32", 2.0, {"in_flight": 1}),
            (f"GPU fp32, {FLY} in flight", "gpu fp32", 2.0, fly),
            (f"GPU fp32, {FLY} in flight, overlap", "gpu fp32", 2.0, both),
            (f"GPU graphs+fp16, {FLY} in flight, overlap",
             "gpu graphs+fp16", 2.0, both))
        per_seed = []
        for s in seeds:
            log(f"\n-- seed {s}")
            per_seed.append(race.race(
                list(problems), backends, particles, batches, seed=s,
                workdir=str(out / "openmc"), contenders=contenders, log=log))
            (out / "race_per_seed.json").write_text(
                json.dumps(output._plain(per_seed), indent=1))
        med = _median_rows(per_seed)
        log(f"\n== MEDIAN of {len(seeds)} seeds (FOM against OpenMC analog)")
        for name, rows in med.items():
            log(f"\n{name}")
            race._report(rows, log)
        (out / "race.json").write_text(json.dumps(output._plain(med),
                                                  indent=1))
        res["race"] = med
    return res
