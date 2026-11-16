"""v2.1.10 record: train_gpu.fit_loop_device (that version of the script)
on the CPU gave the same losses as training.fit_loop (40 steps per family;
check_cpu_equal.log).  Since v2.2.0 the script calls models.train(device=)
from m2.3 instead, so this no longer runs as is."""
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
