"""Short tuning trial of the normalizing flow: python trial.py NAME KEY=VAL ..."""
import sys, json, time, pathlib
sys.path.insert(0, "/home/claude/mtech/gmc2d")
import torch
import generators.nflow as NF
from data import load_dataset
name = sys.argv[1]
for kv in sys.argv[2:]:
    k, v = kv.split("="); setattr(NF, k, type(getattr(NF, k))(v))
NF.STEPS = int(dict(a.split("=") for a in sys.argv[2:]).get("STEPS", 3000))
ds = load_dataset("/home/claude/mtech/gmc2d/data/cells.npz", 0.05, 0)
out = pathlib.Path("/tmp/claude-0/-home-claude-mtech/b1739738-3639-5e2b-8029-9af75a40cf7b/scratchpad/trials") / name
g = NF.NFlow(seed=0)
t = time.time(); facts = g.fit(ds, out, 3 * 3600)
facts["settings"] = {k: getattr(NF, k) for k in ("LAYERS", "WIDTH", "BINS", "LR", "JITTER", "STEPS")}
print("RESULT", name, json.dumps(facts))
