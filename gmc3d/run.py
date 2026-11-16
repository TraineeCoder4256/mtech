"""Command line for gmc3d.  Run from this folder (gmc3d/).

    python run.py solve slab                       plain Monte Carlo
    python run.py solve slab --backend table --r-star 3 --openmc
    python run.py validate                         Stage 1 suite vs OpenMC
    python run.py table                            build the lookup table
    python run.py train                            train the network (the flow)
    python run.py train --family cfm --seed 1      train any model family
    python run.py compare                          every family vs the exact
                                                   walk, and cost per draw
    python run.py ballcheck                        backends vs exact walk
    python run.py measure                          speed study, all problems
    python run.py race                             OpenMC vs ours, head to head
    python run.py meshcheck                        mesh profile vs exact walks
    python run.py maps                             flux maps vs OpenMC, as PNG
    python run.py robust                           random models vs OpenMC
    python run.py detcheck                         point detectors: exact,
                                                   balls vs plain, vs OpenMC
    python run.py gapcheck                         the v1.6.1 features (albedo,
                                                   new filters, sources,
                                                   triggers, ...) vs OpenMC
    python run.py anisocheck                       anisotropic scattering
                                                   (v2.0): every OpenMC law
                                                   vs OpenMC, exact beam
                                                   displacement, detectors
    python run.py gpubench --device cuda           everything a GPU changes,
                                                   measured (checks/gpubench)
    python run.py gate2 cvae_v2                    the acceptance gate for an
                                                   interface v2 model (v2.1.2)
    python run.py score                            one number out of 10 from
                                                   the latest test, validate,
                                                   robust and detector runs
    python run.py check model.py                   what gmc3d will use, refuse
                                                   and warn about, without
                                                   running (also a problem
                                                   name, model.xml or folder)

Backends: mc (no balls), oracle (exact walk), table, network (the default
model: the flow, seed 0), or any model family by name (nflow, cfm ...;
`cfm_seed1` for another training seed).  The families live in ball/models/
(the model side); adding one is a file plus a registry line there.
--device cuda runs the network on a GPU (the transport stays on the CPU);
checks/gpu_bench.py measures that without OpenMC.
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

from ball import models, table                               # noqa: E402
from ball.models import compare as model_compare             # noqa: E402
from ball.models import store as model_store                 # noqa: E402
from ball.oracle import Oracle                                # noqa: E402
from checks import ball_metrics, compare, measure, openmc_ref  # noqa: E402
from checks import flux_maps, mesh_profile, race, validate    # noqa: E402
from checks import source_check                               # noqa: E402
from core import openmc_import, output, transport             # noqa: E402
from problems import PROBLEMS                                 # noqa: E402
from core.tallies import SCORES                               # noqa: E402

DATA = HERE / "data"
TABLE_FILE = DATA / "ball" / "table.npz"
DEFAULT_MODEL = ("nflow", 0)           # what `network` means; changes only
                                       # when Innovation accepts a model
DEFAULT_ANISO_MODEL = ("cvae", 0)      # `network`'s anisotropic balls
                                       # (interface v2, m2.1; accepted
                                       # v2.1.6, 11 Nov 2026)


# how a model runs, from --precision / --compile / --graphs (m2.4): fp32
# with no compile unless asked, so CPU runs stay bit-identical
MODEL_OPTS = {}


def backend(name, device="cpu", **opts):
    """A ball backend by name; opts (precision, compile, graphs) go to every
    network it loads, on top of MODEL_OPTS."""
    o = {**MODEL_OPTS, **opts}
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
        # the accepted models: v1 for isotropic balls, v2 for anisotropic
        # ones (isotropic balls go to v1 unchanged, so isotropic problems
        # give the same numbers as with v1 alone)
        return _warm(_load(*DEFAULT_ANISO_MODEL, device=device,
                           name="network", interface=2,
                           iso=_v1_default(device, **o), **o), o)
    family, seed = _family(name)
    if family is not None and "_v2" in name:
        # interface v2 (anisotropic balls): the v2 model, with isotropic
        # balls going to the accepted v1 model (DEFAULT_MODEL)
        return _warm(_load(family, seed, device=device, name=name,
                           interface=2, iso=_v1_default(device, **o), **o),
                     o)
    if family is None:
        sys.exit(f"unknown backend {name!r}: mc, oracle, table, network or "
                 f"a model family ({', '.join(models.names())})")
    return _warm(_load(family, seed, device=device, name=name, **o), o)


def _load(*a, **kw):
    try:
        return models.load(*a, **kw)
    except (FileNotFoundError, model_store.NotSupported, ValueError) as e:
        sys.exit(str(e))


def _warm(be, o):
    """Compile / record CUDA graphs now, not inside the first timed call."""
    if (o.get("compile") or o.get("graphs")) and hasattr(be, "warmup"):
        be.warmup()
    return be


def _v1_default(device="cpu", **o):
    """The accepted interface v1 model (DEFAULT_MODEL), named `network`."""
    return _load(*DEFAULT_MODEL, device=device, name="network", **o)


def _family(name):
    """'cfm' -> ('cfm', 0); 'cfm_seed1' -> ('cfm', 1); else (None, 0)."""
    base, _, seed = name.partition("_seed")
    if base.endswith("_v2"):
        base = base[:-3]
    if base in models.REGISTRY and (not seed or seed.isdigit()):
        return base, int(seed or 0)
    return None, 0


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
    for line in pr.warnings:
        log(f"  WARNING {line}")
    be = backend(args.backend, args.device)
    r = transport.run(pr, args.particles, args.batches, seed=args.seed,
                      backend=be, r_star=args.r_star,
                      mesh_rule=args.mesh_rule, verbose=True)
    log(f"{args.problem} with {args.backend}: leakage {r.leakage:.6g} +- "
        f"{r.leakage_se:.2g}")
    for name, t in r.tallies.items():
        if t["filter"] == "surface":
            for sid, m_, e_ in zip(t["bin_ids"], t["mean"][:, 0],
                                   t["se"][:, 0]):
                log(f"  {name}: surface {sid} current {m_:.6g} +- {e_:.2g}")
        elif t["mean"].shape[0] == 1:
            log(f"  {name}: " + ", ".join(
                f"{sc} {t['mean'][0, k]:.6g} +- {t['se'][0, k]:.2g}"
                for k, sc in enumerate(SCORES) if sc in t["scores"]))
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
        if t["filter"] == "mesh" and t.get("mesh_type", "rect") == "rect":
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
    fams = models.names() if args.family == "all" else [args.family]
    seeds = args.seeds or [args.seed]
    if args.interface == 2 and "cgan" in fams and args.family == "all":
        fams = [f for f in fams if f != "cgan"]     # not accepted (v2.1.7)
    if args.interface == 2:
        from ball.models import pipeline
        pipeline.aniso_data(log)                    # built once, CPU
    for s in seeds:
        for fam in fams:
            be = models.train(fam, seed=s, steps=args.steps, log=log,
                              interface=args.interface, device=args.device,
                              precision=args.precision,
                              compile=args.compile)
            d = {k: v for k, v in be.describe().items() if k != "train"}
            log(f"saved to data/models/"
                f"{model_store.folder(fam, s, args.interface)}"
                f" (and the shared copy if there is one); {d}")
    tag = args.family + ("_v2" if args.interface == 2 else "")
    return f"train_{tag}_seed{'-'.join(map(str, seeds))}"


def cmd_compare(args, log):
    res = model_compare.run({b: backend(b, args.device)
                             for b in args.backends}, log=log)
    (out_dir(args) / "compare.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "compare"


def cmd_ballcheck(args, log):
    rows = {}
    for b in args.backends:
        rows[b] = ball_metrics.score(backend(b, args.device))
        log(ball_metrics.report(rows[b], f"\n{b} vs the exact walk"))
    (out_dir(args) / "ballcheck.json").write_text(
        json.dumps(output._plain(rows), indent=1))
    return "ballcheck"


def cmd_measure(args, log):
    bes = {b: backend(b, args.device) for b in args.backends}
    res = measure.study(args.problems or list(validate.PROBLEM_ORDER), bes,
                        r_stars=args.r_stars, particles=args.particles,
                        batches=args.batches, seed=args.seed, log=log)
    (out_dir(args) / "measure.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "measure"


def cmd_race(args, log):
    bes = {b: backend(b, args.device) for b in args.backends}
    res = race.race(args.problems or list(validate.PROBLEM_ORDER) + ["cask"],
                    bes, particles=args.particles, batches=args.batches,
                    seed=args.seed, workdir=DATA / "openmc", log=log)
    (out_dir(args) / "race.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "race"


def cmd_meshcheck(args, log):
    rows = mesh_profile.run(log=log)
    (out_dir(args) / "meshcheck.json").write_text(
        json.dumps(output._plain(rows), indent=1))
    return "meshcheck"


def cmd_maps(args, log):
    res = flux_maps.run(args.problems or list(validate.VALIDATE_ORDER),
                        backend(args.backend, args.device),
                        out_dir(args) / "maps", DATA / "openmc",
                        args.particles, args.batches, seed=args.seed,
                        r_star=args.r_star, log=log)
    (out_dir(args) / "maps.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "maps"


def cmd_sourcecheck(args, log):
    res = source_check.run(DATA / "openmc", args.particles, args.batches,
                           seed=args.seed, names=args.cases or None, log=log)
    (out_dir(args) / "sourcecheck.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "sourcecheck"


def cmd_check(args, log):
    from core import model_check
    rep = model_check.check(model_check.load_model(args.model))
    for line in rep.lines():
        log(line)
    return None


def cmd_robust(args, log):
    from checks import robust
    res = robust.suite(DATA / "openmc", args.models, log=log)
    (out_dir(args) / "robust.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "robust"


def cmd_detcheck(args, log):
    from checks import detcheck
    res = detcheck.run(DATA / "openmc", args.particles, args.batches,
                       seed=args.seed, log=log)
    (out_dir(args) / "detcheck.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "detcheck"


def cmd_gapcheck(args, log):
    from checks import gapcheck
    res = gapcheck.suite(DATA / "openmc", args.particles, args.batches,
                         names=args.cases or None, log=log)
    (out_dir(args) / "gapcheck.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return "gapcheck"


def cmd_anisocheck(args, log):
    from checks import aniso
    be = None if args.backend == "oracle" else backend(args.backend,
                                                         args.device)
    if be is not None:
        log(f"balls walked by {args.backend}: {be.describe()}")
    res = aniso.suite(DATA / "openmc", args.particles, args.batches,
                      names=args.cases or None, log=log, ball_backend=be)
    tag = "anisocheck" + ("" if be is None else f"_{args.backend}")
    (out_dir(args) / f"{tag}.json").write_text(
        json.dumps(output._plain(res), indent=1))
    return tag


def cmd_gate2(args, log):
    from checks import gate_v2
    be = backend(args.model, args.device)
    res = gate_v2.run(be, n=args.particles, log=log)
    gate_v2.save(res, out_dir(args))
    return f"gate2_{args.model}"


def cmd_gpubench(args, log):
    from checks import gpubench
    gpubench.run(out_dir(args), device=args.device,
                 problems=args.problems or gpubench.PROBLEMS,
                 seeds=args.seeds, particles=args.particles,
                 batches=args.batches, parts=args.parts,
                 load=lambda name, device, **o: backend(name, device, **o),
                 log=log)
    return "gpubench"


def cmd_score(args, log):
    from checks import score
    parts, total = score.score(out_dir(args))
    text = score.report(parts, total)
    log(text)
    (out_dir(args) / "SCORE.md").write_text(
        f"# gmc3d check score: {total:.1f} / 10\n\n{text}\n")
    return "score"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, particles=100_000, batches=20, inference=True):
        sp.add_argument("--particles", type=int, default=particles)
        sp.add_argument("--batches", type=int, default=batches)
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--out", default=None, help="output folder")
        sp.add_argument("--device", default="cpu",
                        help="where the network runs: cpu or cuda")
        sp.add_argument("--in-flight", type=int, default=1,
                        help="statistical batches run at once (same "
                             "answer; bigger network calls, for a GPU)")
        sp.add_argument("--overlap", action="store_true",
                        help="the CPU tracks while the GPU draws (needs a "
                             "backend with submit/collect)")
        if inference:
            sp.add_argument("--precision", default="fp32",
                            choices=["fp32", "tf32", "fp16"],
                            help="the network's arithmetic (fp16: GPU)")
            sp.add_argument("--compile", action="store_true",
                            help="torch.compile the network")
            sp.add_argument("--graphs", action="store_true",
                            help="CUDA graphs (GPU; implies --compile)")

    s = sub.add_parser("solve")
    s.add_argument("problem", choices=sorted(PROBLEMS))
    s.add_argument("--backend", default="mc")
    s.add_argument("--r-star", type=float, default=3.0)
    s.add_argument("--mesh-rule", default="profile",
                   choices=["profile", "cap"])
    s.add_argument("--openmc", action="store_true")
    common(s)

    s = sub.add_parser("validate")
    s.add_argument("problems", nargs="*")
    common(s)

    s = sub.add_parser("table")
    common(s)

    s = sub.add_parser("train")
    s.add_argument("--family", default="nflow",
                   choices=models.names() + ["all"],
                   help="all: every family (at --interface 2, all but cgan)")
    s.add_argument("--interface", type=int, default=1, choices=[1, 2],
                   help="1: isotropic balls (R -> mu, s); 2: anisotropic")
    s.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="several seeds in one go (default: --seed)")
    s.add_argument("--steps", type=int, default=None,
                   help="default: the shared step budget (fairness.py)")
    s.add_argument("--precision", default="fp32",
                   choices=["fp32", "tf32", "fp16"],
                   help="fp16: mixed precision on a GPU (not cgan)")
    s.add_argument("--compile", action="store_true",
                   help="torch.compile the training step")
    common(s, inference=False)
    s.set_defaults(seed=0)

    s = sub.add_parser("compare")
    s.add_argument("--backends", nargs="+", default=models.names(),
                   help="families (cfm, cfm_seed1 ...), oracle, table")
    common(s)

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

    s = sub.add_parser("meshcheck")
    common(s)

    s = sub.add_parser("sourcecheck")
    s.add_argument("cases", nargs="*", help=" ".join(source_check.CASES))
    common(s)
    s.set_defaults(seed=5)

    s = sub.add_parser("maps")
    s.add_argument("problems", nargs="*")
    s.add_argument("--backend", default="network",
                   help="oracle, table, network or a model family")
    s.add_argument("--r-star", type=float, default=3.0)
    common(s)
    s.set_defaults(seed=5)

    s = sub.add_parser("robust")
    s.add_argument("--models", type=int, default=30)
    common(s)

    s = sub.add_parser("detcheck")
    common(s)
    s.set_defaults(seed=7)

    s = sub.add_parser("gapcheck")
    s.add_argument("cases", nargs="*")
    common(s, particles=20_000)

    s = sub.add_parser("anisocheck")
    s.add_argument("cases", nargs="*")
    s.add_argument("--backend", default="oracle",
                   help="what walks the balls: oracle (exact) or a v2 "
                        "model, e.g. nflow_v2_seed0")
    common(s, particles=50_000)

    s = sub.add_parser("gate2")
    s.add_argument("model", help="a v2 model, e.g. cvae_v2 or nflow_v2_seed1")
    common(s, particles=100_000)

    s = sub.add_parser("gpubench")
    s.add_argument("problems", nargs="*")
    s.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    s.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="1 machine, 2 GPU tests, 3 model bench, 4 gate2, "
                        "5 race")
    common(s, batches=40)
    s.set_defaults(device="cuda")

    s = sub.add_parser("score")
    common(s)

    s = sub.add_parser("check")
    s.add_argument("model", help="problem name, .py file, model.xml or "
                   "folder of OpenMC xml files")

    args = p.parse_args()
    transport.DEFAULTS.update(in_flight=getattr(args, "in_flight", 1),
                              overlap=getattr(args, "overlap", False))
    if args.cmd != "train":
        MODEL_OPTS.update({k: v for k, v in (
            ("precision", getattr(args, "precision", "fp32")),
            ("compile", getattr(args, "compile", False)),
            ("graphs", getattr(args, "graphs", False))) if v not in
            ("fp32", False)})
    log = Log()
    stem = globals()[f"cmd_{args.cmd}"](args, log)
    if stem:
        log.save(out_dir(args) / f"{stem}.log")


if __name__ == "__main__":
    main()
