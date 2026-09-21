# One full run, 21 September 2026

Committed so the numbers in [`../../SPEED.md`](../../SPEED.md) can be checked
without re-running the pipeline. `results/`, `figures/` and `models/` are
gitignored as run artefacts, which is right for a working tree and wrong for
a claim someone has to trust, so this one run is kept.

**Hardware.** Intel Xeon at 2.8 GHz, 4 cores, no GPU. torch 2.14.0 on 4
threads, numba 0.67.0 on 4 threads. Every wall time here is CPU-only on both
sides. A large dense matmul reaches 521 GFLOP/s on this machine, which is the
ceiling the arithmetic claims are measured against.

| file | produced by | what it is |
|---|---|---|
| `validate.txt` | `validate.py` | the MC baseline against three exact identities; all twelve cases pass |
| `train_log.txt` | `train.py` | 20,000 steps at batch 4,096, about 50 minutes here |
| `loss.png` | `train.py` | training and validation CFM loss |
| `evaluation.txt` | `evaluate.py` | accuracy and cost against the baseline |
| `accuracy.pdf` | `evaluate.py` | flux fields and their difference |
| `speed.pdf` | `evaluate.py` | cost against optical thickness |
| `speed_profile.{txt,json}` | `speed_profile.py` | where GMC's time goes, and the crossover projection |
| `speed_prototype.{txt,json}` | `speed_prototype.py` | whether the candidate fixes recover it |

**The headline.** On the lattice at the published cross sections, 20,000
particles: MC 0.029 s, GMC 9.8 s. The sampler is 337x slower and its flux is
0.94% from the MC field, against a 1.45% MC-vs-MC noise floor — accurate, and
not yet fast. `SPEED.md` says why and what to do about it.

**Reproducing.** `python validate.py && python make_data.py && python
train.py && python evaluate.py && python speed_profile.py && python
speed_prototype.py`. The training seed is fixed, so the model is the same;
wall times will follow whatever machine you run on, and the profile prints
its own.
