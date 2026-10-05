"""Command line for gmc3d.  Run from this folder (gmc3d/).

    python run.py solve slab                       plain Monte Carlo
    python run.py solve slab --backend table --r-star 3 --openmc
    python run.py validate                         Stage 1 suite vs OpenMC
    python run.py table                            build the lookup table
    python run.py train                            train the network
    python run.py ballcheck                        backends vs exact walk
    python run.py measure                          speed study, all problems
    python run.py race                             OpenMC vs ours, head to head

Backends: mc (no balls), oracle (exact walk), table, network.
Every command writes its output into runs/<today>/ (text log + JSON, and
VTK files for mesh tallies from `solve`).  Generated data (cross-section
libraries, walk data, the table, trained networks, OpenMC run folders) goes
in data/, which git ignores.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ball import network, table                              # noqa: E402
from ball.oracle import Oracle                                # noqa: E402
from checks import ball_metrics, compare, measure, openmc_ref  # noqa: E402
from checks import race, validate                             # noqa: E402
from core import openmc_import, output, transport             # noqa: E402
from problems import PROBLEMS                                 # noqa: E402

DATA = HERE / "data"
TABLE_FILE = DATA / "ball" / "table.npz"
NETWORK_DIR = network.MODEL_DIR / "flow_seed0"


def backend(name):
    if name == "mc":
        return None
    if name == "oracle":
        return Oracle()
    if name == "table":
        if not TABLE_FILE.exists():
            print("building the table (python run.py table) ...")
            table.Table.build().save(TABLE_FILE)
        return table.Table.load(TABLE_FILE)
    if name == "network":
        if not (NETWORK_DIR / "model.pt").exists():
            sys.exit("no trained network: run  python run.py train  first")
        return network.Network.load(NETWORK_DIR)
    sys.exit(f"unknown backend {name!r}")


class Log:
    """print() that also keeps a copy for the run folder."""

    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(self.lines) + "\n")


def out_dir(args):
    d = Path(args.out) if args.out else \
        HERE / "runs" / datetime.date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_solve(args, log):
    model = PROBLEMS[args.problem]()
    pr = openmc_import.load(model)
    be = backend(args.backend)
    r = transport.run(pr, args.particles, args.batches, seed=args.seed,
                      backend=be, r_star=args.r_star,
                      mesh_rule=args.mesh_rule, verbose=True)
    log(f"{args.problem} with {args.backend}: leakage {r.leakage:.6g} +- "
        f"{r.leakage_se:.2g}")
    for name, t in r.tallies.items():
        if t["mean"].shape[0] == 1:
            log(f"  {name}: flux {t['mean'][0, 0]:.6g} +- {t['se'][0, 0]:.2g}"
                f", absorption {t['mean'][0, 1]:.6g} +- {t['se'][0, 1]:.2g}")
    log(f"  counters {r.counters}")
    log(f"  timing {r.timing}")
    d = out_dir(args)
    stem = f"{args.problem}_{args.backend}"
    extra = {}
    if args.openmc:
        ref = openmc_ref.run(model, DATA / "openmc" / args.problem,
                             args.particles, args.batches,
                             seed=args.seed + 100)
        cmp = compare.compare(r, ref)
        log(compare.report(cmp, "  vs OpenMC"))
        extra["vs_openmc"] = cmp
    output.save_json(r, d / f"{stem}.json", extra)
    for name, t in r.tallies.items():
        if t["filter"] == "mesh":
            output.save_vtk(t, d / f"{stem}_{name}.vtk")
    log(f"  written to {d}/{stem}.*")
    return stem


def cmd_validate(args, log):
    res = validate.suite(DATA / "openmc", args.particles, args.batches,
                         seed=args.seed, names=args.problems or None, log=log)
    (out_dir(args) / "validate.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "validate"


def cmd_table(args, log):
    t = table.Table.build()
    t.save(TABLE_FILE)
    log(f"table: {t.mu_tab.shape[0]} radii from {t.r_min:g} to {t.r_max:g}"
        f", {t.mu_tab.shape[1]} walks each, saved to {TABLE_FILE}")
    return None


def cmd_train(args, log):
    net = network.train(NETWORK_DIR if args.seed == 0 else None,
                        steps=args.steps, seed=args.seed, log=log)
    log(f"saved; {net.describe()}")
    return f"train_seed{args.seed}"


def cmd_ballcheck(args, log):
    rows = {}
    for b in args.backends:
        rows[b] = ball_metrics.score(backend(b))
        log(ball_metrics.report(rows[b], f"\n{b} vs the exact walk"))
    (out_dir(args) / "ballcheck.json").write_text(
        json.dumps(output._plain(rows), indent=1))
    return "ballcheck"


def cmd_measure(args, log):
    bes = {b: backend(b) for b in args.backends}
    res = measure.study(args.problems or list(validate.PROBLEM_ORDER), bes,
                        r_stars=args.r_stars, particles=args.particles,
                        batches=args.batches, seed=args.seed, log=log)
    (out_dir(args) / "measure.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "measure"


def cmd_race(args, log):
    bes = {b: backend(b) for b in args.backends}
    res = race.race(args.problems or list(validate.PROBLEM_ORDER) + ["cask"],
                    bes, particles=args.particles, batches=args.batches,
                    seed=args.seed, workdir=DATA / "openmc", log=log)
    (out_dir(args) / "race.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "race"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, particles=100_000, batches=20):
        sp.add_argument("--particles", type=int, default=particles)
        sp.add_argument("--batches", type=int, default=batches)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--out", default=None, help="output folder")

    s = sub.add_parser("solve")
    s.add_argument("problem", choices=sorted(PROBLEMS))
    s.add_argument("--backend", default="mc",
                   choices=["mc", "oracle", "table", "network"])
    s.add_argument("--r-star", type=float, default=3.0)
    s.add_argument("--mesh-rule", default="cap", choices=["cap", "centre"])
    s.add_argument("--openmc", action="store_true")
    common(s)

    s = sub.add_parser("validate")
    s.add_argument("problems", nargs="*")
    common(s)

    s = sub.add_parser("table")
    common(s)

    s = sub.add_parser("train")
    s.add_argument("--steps", type=int, default=network.STEPS)
    common(s)
    s.set_defaults(seed=0)

    s = sub.add_parser("ballcheck")
    s.add_argument("--backends", nargs="+", default=["table", "network"])
    common(s)

    s = sub.add_parser("measure")
    s.add_argument("problems", nargs="*")
    s.add_argument("--backends", nargs="+",
                   default=["oracle", "table", "network"])
    s.add_argument("--r-stars", nargs="+", type=float,
                   default=[2.0, 3.0, 5.0, 10.0])
    common(s, batches=10)

    s = sub.add_parser("race")
    s.add_argument("problems", nargs="*")
    s.add_argument("--backends", nargs="+",
                   default=["oracle", "network", "table"])
    common(s, batches=40)

    args = p.parse_args()
    log = Log()
    stem = globals()[f"cmd_{args.cmd}"](args, log)
    if stem:
        log.save(out_dir(args) / f"{stem}.log")


if __name__ == "__main__":
    main()
