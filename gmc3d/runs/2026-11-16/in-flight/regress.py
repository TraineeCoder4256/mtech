"""Bit-for-bit regression for the v2.2 transport changes (several batches
in flight, overlapped backend calls).  The exact-walk backend (Oracle) uses
each particle's own random stream, so every arrangement must give the SAME
numbers as one batch at a time.

    python runs/2026-11-16/in-flight/regress.py save   [before the change]
    python runs/2026-11-16/in-flight/regress.py check  [after: K = 1, 3, 4
                                                        and overlap]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np                                          # noqa: E402

from ball.oracle import Oracle                              # noqa: E402
from checks import aniso, detcheck                          # noqa: E402
from core import openmc_import, transport                   # noqa: E402
from problems import PROBLEMS                               # noqa: E402

N, B = 3000, 6          # 3000 is not a multiple of 32 chunks: uneven split


def cases():
    yield "sphere", openmc_import.load(PROBLEMS["sphere"]()), {}
    yield "cask", openmc_import.load(PROBLEMS["cask"]()), {}
    yield "slab", openmc_import.load(PROBLEMS["slab"]()), {}
    yield "tilted", openmc_import.load(PROBLEMS["tilted"]()), {}
    m = aniso._no_meshes(aniso._build("sphere", aniso.P3, "p3", {}))
    yield "sphere_p3", openmc_import.load(m), {}
    yield "detectors", openmc_import.load(detcheck.scatter_model()), {}


def flat(r):
    out = {"leakage": np.array([r.leakage, r.leakage_se,
                                r.leakage_se_batches])}
    for k, t in r.tallies.items():
        out[f"{k}/mean"] = np.asarray(t["mean"])
        out[f"{k}/se"] = np.asarray(t["se"])
    c = dict(r.counters)
    c.pop("backend_calls", None)
    out["counters"] = np.array([c[k] for k in sorted(c)], float)
    return out


class Deferred(Oracle):
    """The exact walk behind submit/collect (the work is done at submit),
    to drive transport's overlap path without a GPU."""

    def submit(self, R, seeds, idx):
        return self.sample(R, seeds, idx)

    def submit_aniso(self, R, mats, ang, seeds, idx):
        return self.sample_aniso(R, mats, ang, seeds, idx)

    def collect(self, ticket):
        return ticket


def run_all(**kw):
    res = {}
    for name, pr, extra in cases():
        for be in (None, Deferred() if kw.get("overlap") else Oracle()):
            r = transport.run(pr, N, B, seed=3, backend=be, r_star=2.0,
                              **extra, **kw)
            tag = f"{name}/{'oracle' if be else 'mc'}"
            for k, v in flat(r).items():
                res[f"{tag}/{k}"] = v
            print(tag, r.leakage, r.counters.get("backend_calls"),
                  flush=True)
    return res


if sys.argv[1] == "save":
    np.savez(HERE / "baseline.npz", **run_all())
else:
    base = np.load(HERE / "baseline.npz")
    opts = [dict(in_flight=1), dict(in_flight=3), dict(in_flight=4),
            dict(in_flight=4, overlap=True)]
    bad = 0
    for o in opts:
        res = run_all(**o)
        diff = [k for k in base.files if k not in res or
                not np.array_equal(base[k], res[k], equal_nan=True)]
        print(o, "IDENTICAL" if not diff else f"DIFFER: {diff[:6]}",
              flush=True)
        bad += bool(diff)
    sys.exit(bad)
