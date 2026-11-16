"""Check: train_gpu.fit_loop_device on the CPU gives the same losses as
training.fit_loop (40 steps per family).  Run from gmc3d/ with PYTHONPATH=."""
import sys, importlib.util
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("tg", "runs/2026-11-16/gpu-train/train_gpu.py")
tg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)
from ball.models import pipeline, base, training, fairness, encoding
import numpy as np
d = pipeline.aniso_data()
# subsample to keep it fast but keep dtype
sub = {k: v for k, v in d.items()}
from ball import models
for fam in ["nflow", "cfm", "cvae"]:
    out = []
    for fl in (training.fit_loop, tg.fit_loop_device):
        tg.DEVICE = "cpu"
        base.fit_loop = fl
        F = models.get(fam)
        net, cfg, facts = F.fit(d, seed=0, steps=40, interface=2, log=lambda *a: None)
        out.append(facts["history"][-1])
    print(fam, out, "IDENTICAL" if out[0] == out[1] else "DIFFER", flush=True)
