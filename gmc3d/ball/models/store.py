"""Where trained models live, and the manifest that says what each one is.

Used by: pipeline.py (saves), backend.py and ball/network.py (load).
Uses: ball/interface.py (the version a model must match).

One folder per trained model, named <family>_seed<k>:

    gmc3d/data/models/<family>_seed<k>/      the local copy (git ignores
        model.pt                             data/), the one that is loaded
        manifest.json
    /mnt/project-files/models/<family>_seed<k>/   the shared copy

Why two.  Each working session runs in its own container, and containers
are wiped, so a model trained in one would never reach another.  The
project-files folder is shared by every session and survives them, so
saving writes both, and loading copies the shared one in when the local one
is missing.  Weights stay out of git.

manifest.json answers "what is this model?" without loading it: family,
seed, the interface version it was trained for, its settings and weight
count, the steps it reached, a hash of the walks it saw, and the git
commit of the code that trained it.  A model trained for another interface
version is refused at load time ("not yet supported" until retrained).

model.pt keeps the format ball/network.py has always written (ema, info,
cnorm, ynorm), plus family/config/interface, so older flow checkpoints
still load: one without a family is a flow, one without an interface
version was trained for interface 1.
"""

import datetime
import json
import shutil
import subprocess
from pathlib import Path

import torch

from .. import interface as I
from .encoding import Norm

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
SHARED_DIR = Path("/mnt/project-files/models")


class NotSupported(RuntimeError):
    """A model trained for a different interface version."""


def folder(family, seed):
    return f"{family}_seed{int(seed)}"


def _commit():
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=here, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."],
                               cwd=here.parent.parent, capture_output=True,
                               text=True).stdout.strip() != ""
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save(out_dir, family, config, net, data, facts, shared=True):
    """Write model.pt and manifest.json; copy both to the shared home."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {**facts, "r_min": data["r_min"], "r_max": data["r_max"],
            "n_train": int(data["y_train"].shape[0]),
            **({k: config[k] for k in config})}
    torch.save({"family": family.name, "config": dict(config),
                "interface": I.INTERFACE_VERSION,
                "ema": {k: v.cpu() for k, v in net.state_dict().items()},
                "info": info,
                "cnorm": (data["cnorm"].mean, data["cnorm"].std),
                "ynorm": (data["ynorm"].mean, data["ynorm"].std)},
               out_dir / "model.pt")
    manifest = {
        "family": family.name, "seed": facts["seed"],
        "interface": I.INTERFACE_VERSION, "nfe": family.nfe,
        "config": dict(config),
        "params": facts["params"], "steps": facts["steps"],
        "steps_done": facts["steps_done"],
        "train_seconds": round(facts["train_seconds"], 1),
        "train": facts.get("train"),
        "final_val_loss": facts["history"][-1][2] if facts["history"]
        else None,
        "data": {"n_train": int(data["y_train"].shape[0]),
                 "n_val": int(data["y_val"].shape[0]),
                 "r_min": data["r_min"], "r_max": data["r_max"],
                 "seed": data.get("seed"), "sha256": data.get("sha256")},
        "commit": _commit(),
        "created": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if shared and SHARED_DIR.parent.exists():
        dest = SHARED_DIR / out_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("model.pt", "manifest.json"):
            shutil.copy2(out_dir / f, dest / f)
    return manifest


def locate(family, seed):
    """The local folder for <family>_seed<k>, copied in from the shared
    home if only that exists.  None if neither has it."""
    local = MODEL_DIR / folder(family, seed)
    if (local / "model.pt").exists():
        return local
    shared = SHARED_DIR / folder(family, seed)
    if (shared / "model.pt").exists():
        local.mkdir(parents=True, exist_ok=True)
        for f in shared.iterdir():
            shutil.copy2(f, local / f.name)
        return local
    return None


def read(out_dir):
    """model.pt -> (family name, config, state dict, info, cnorm, ynorm).
    Raises NotSupported for a model trained for another interface."""
    st = torch.load(Path(out_dir) / "model.pt", map_location="cpu",
                    weights_only=False)
    version = st.get("interface", 1)
    if version != I.INTERFACE_VERSION:
        raise NotSupported(
            f"{out_dir} was trained for interface v{version}; the current "
            f"interface is v{I.INTERFACE_VERSION}.  Not yet supported: "
            f"retrain it.")
    info = st["info"]
    family = st.get("family", "nflow")
    config = st.get("config") or {k: info[k] for k in
                                  ("layers", "width", "bins", "emb")}
    return (family, config, st["ema"], info, Norm(*st["cnorm"]),
            Norm(*st["ynorm"]))
