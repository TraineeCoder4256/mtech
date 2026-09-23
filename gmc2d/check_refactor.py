#!/usr/bin/env python3
"""Proof that the pipeline refactor changed nothing.  Run: python check_refactor.py

On 23 Sept 2026 the flow-matching ODE solve and training loop moved out of
sampler.py and train.py into generators/cfm.py, behind the Generator
interface.  That is only safe if every number the old code produced, the new
code produces too -- not approximately, BIT FOR BIT.  This script runs the
old code (read straight out of git at commit 81d8340, so it cannot drift)
side by side with the new code on identical inputs and compares with
np.array_equal, not a tolerance.

No trained model is needed.  The question is "does the refactor change
anything", not "is the model good", and a randomly initialised network
answers it just as well as a trained one.

Three checks, from the smallest unit up:

  1. SAMPLING     one batch of exit states, for each ODE solver
  2. TRANSPORT    a whole lattice solve through gmc_solve
  3. TRAINING     a short training run: the saved weights and normalisers
                  (needs data/cells.npz from make_data.py; skipped otherwise)
"""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

import mc
from data import Normalizer
from model import VelocityField
from sampler import Sampler, CellSampler
from generators.cfm import CFM
from solve import gmc_solve

BASE = "81d8340"              # the last commit before the split
UNCHANGED = ["mc.py", "data.py", "model.py", "solve.py"]


def git_show(path):
    return subprocess.run(["git", "show", f"{BASE}:gmc2d/{path}"], check=True,
                          capture_output=True, text=True).stdout


def load_legacy(name, tmp):
    """Import the file as it was at BASE, under a name that cannot collide
    with the current module."""
    src = pathlib.Path(tmp) / f"legacy_{name}.py"
    src.write_text(git_show(f"{name}.py"))
    spec = importlib.util.spec_from_file_location(f"legacy_{name}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def conditions(n, rng):
    W = np.exp(rng.uniform(np.log(0.05), np.log(20.0), n))
    H = W * np.exp(rng.uniform(-1.0, 1.0, n))
    r, th = np.sqrt(rng.uniform(0, 1, n)), rng.uniform(-1.5, 1.5, n)
    return W, H, rng.uniform(0, 1, n), np.maximum(r * np.cos(th), 1e-4), r * np.sin(th)


def main():
    ok = True
    diff = subprocess.run(["git", "diff", "--quiet", BASE, "--"] + UNCHANGED)
    print(f"{', '.join(UNCHANGED)} unchanged since {BASE}: "
          f"{'yes' if diff.returncode == 0 else 'NO'}")
    ok &= diff.returncode == 0

    torch.manual_seed(1)
    net = VelocityField(6, 5, 64, 2)          # small and random: fast, and enough
    rng = np.random.default_rng(3)
    yn = Normalizer().fit(rng.normal(size=(1000, 6)).astype(np.float32) * 2 + 0.5)
    cn = Normalizer().fit(rng.normal(size=(1000, 5)).astype(np.float32))

    with tempfile.TemporaryDirectory() as tmp:
        old = load_legacy("sampler", tmp)

        print("\n1. SAMPLING  (20,000 entry states, every output array compared)")
        W, H, xi, ox, oy = conditions(20000, np.random.default_rng(5))
        for solver, steps in (("euler", 4), ("heun", 5), ("rk4", 2)):
            a = old.Sampler(net, yn, cn, "cpu", steps, solver).sample(W, H, xi, ox, oy, seed=9)
            news = {"Sampler()": Sampler(net, yn, cn, "cpu", steps, solver),
                    "CellSampler(CFM)": CellSampler(CFM(net, steps, solver), yn, cn)}
            for label, smp in news.items():
                b = smp.sample(W, H, xi, ox, oy, seed=9)
                good = all(same(a[k], b[k]) for k in a) and set(a) == set(b)
                ok &= good
                print(f"   {solver:<5} x{steps}  {label:<17} "
                      f"{'bit-identical' if good else 'DIFFERENT'}")

        print("\n2. TRANSPORT  (gmc_solve on the lattice, 3,000 particles)")
        prob = mc.lattice()
        old_smp = old.Sampler(net, yn, cn, "cpu", 5, "heun")
        new_smp = Sampler(net, yn, cn, "cpu", 5, "heun")
        fa, sa = gmc_solve(prob, 3000, old_smp, seed=11)
        fb, sb = gmc_solve(prob, 3000, new_smp, seed=11)
        keys = [k for k in sa if not k.startswith("wall")]
        good = same(fa, fb) and all(sa[k] == sb[k] for k in keys)
        ok &= good
        print(f"   flux field and {len(keys)} counters  "
              f"{'bit-identical' if good else 'DIFFERENT'}")

        print("\n3. TRAINING  (60 steps at batch 4096, weights and normalisers)")
        if not pathlib.Path("data/cells.npz").exists():
            print("   skipped: data/cells.npz not found -- run make_data.py")
        else:
            old_train = load_legacy("train", tmp)
            old_train.STEPS, old_train.VAL_EVERY = 60, 20
            old_train.OUT = pathlib.Path(tmp) / "old"
            old_train.main()

            import generators.cfm as cfm
            import train as new_train
            cfm.STEPS, cfm.VAL_EVERY = 60, 20
            new_train.OUT = pathlib.Path(tmp) / "new"
            new_train.main()

            A = torch.load(old_train.OUT / "model.pt", weights_only=True)
            B = torch.load(new_train.OUT / "model.pt", weights_only=True)
            w_same = (A["config"] == B["config"] and A["ema"].keys() == B["ema"].keys()
                      and all(torch.equal(A["ema"][k], B["ema"][k]) for k in A["ema"]))
            n_same = (json.loads((old_train.OUT / "norm.json").read_text()) ==
                      json.loads((new_train.OUT / "norm.json").read_text()))
            strip = lambda f: [ln.rsplit(None, 2)[0] for ln in f.read_text().splitlines()]
            l_same = strip(old_train.OUT / "log.txt") == strip(new_train.OUT / "log.txt")
            ok &= w_same and n_same
            print(f"   {len(A['ema'])} weight tensors  {'bit-identical' if w_same else 'DIFFERENT'}")
            print(f"   normalisers        {'identical' if n_same else 'DIFFERENT'}")
            print(f"   loss log           {'identical' if l_same else 'differs'} "
                  f"(throughput column excluded: it is wall-clock)")

    print("\n" + ("REFACTOR CHANGES NOTHING" if ok else "*** REFACTOR CHANGED SOMETHING ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
