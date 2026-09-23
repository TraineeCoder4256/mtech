#!/usr/bin/env python3
"""Step 2: train the boundary model.  Run: python train.py

Conditional flow matching.  The training loop now lives in generators/cfm.py
behind the Generator interface that every model family implements; this
script keeps the old command working and writes exactly what it always did.
run.py is the general entry point (`python run.py cfm`).

Writes models/model.pt, models/norm.json, models/loss.png, models/log.txt.
"""
import pathlib

import torch

from data import load_dataset, save_norm
from generators.cfm import CFM, SEED

DATA = pathlib.Path("data/cells.npz")
OUT = pathlib.Path("models")
VAL_FRAC = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    print(f"device {DEVICE}")

    ds = load_dataset(DATA, VAL_FRAC, SEED)
    i = ds["info"]
    print(f"data   {i['rows']:,} rows -> {i['collided']:,} collided, "
          f"{i['conditions']:,} conditions")
    print(f"split  {i['train']:,} train / {i['val']:,} val "
          f"({i['val_conditions']:,} held-out conditions)")

    gen = CFM(device=DEVICE)
    gen.fit(ds, OUT)
    gen.save(OUT)
    save_norm(OUT / "norm.json", ds["ynorm"], ds["cnorm"], gen.config)
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
