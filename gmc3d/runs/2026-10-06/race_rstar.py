"""Race with R* = 1.5 as well as 2, on this container, fresh OpenMC."""
import json, sys
from pathlib import Path
HERE = Path("/home/claude/mtech/gmc3d"); sys.path.insert(0, str(HERE))
from ball import network, table
from ball.oracle import Oracle
from checks import race
from core import output

TABLE = HERE / "data" / "ball" / "table.npz"
bes = {"table": table.Table.load(TABLE),
       "network": network.Network.load(network.MODEL_DIR / "flow_seed0")}
CONT = (("ours, plain MC", None, None),
        ("ours + table", "table", 1.5),
        ("ours + table", "table", 2.0),
        ("ours + network", "network", 2.0))
log_lines = []
def log(*p):
    s = " ".join(str(x) for x in p); print(s, flush=True); log_lines.append(s)
res = race.race(["sphere", "slab", "cask", "curved"], bes,
                particles=100_000, batches=40, seed=11,
                workdir=str(HERE / "data" / "openmc"), contenders=CONT,
                log=log)
out = HERE / "runs" / "2026-10-06"
(out / "race_fixed.json").write_text(json.dumps(output._plain(res), indent=1))
(out / "race_fixed.log").write_text("\n".join(log_lines) + "\n")
print("written")
