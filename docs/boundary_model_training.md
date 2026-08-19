# Boundary GMC model — training notes

Working reference for turning `data/lattice_singlecell*.npz` into a trained
boundary sampler. Written before the architecture exists; revisit each item
when wiring up the training loop.

Source: arXiv:2512.13965v1. Architecture and hyperparameters are in
supplementary material S.1–S.3, which is **not public** — anything below not
marked "paper" is our own reconstruction.

---

## 1. What the paper specifies

**Conditioning (Eq. 10):**

```
c = (W̃, H̃, ξ_in, Ω_in)
```

| symbol | meaning |
|---|---|
| `W̃`, `H̃` | cell optical dimensions, `W·σ_s` and `H·σ_s` (mfp) |
| `ξ_in` | entry position along the face, normalised to [0,1] |
| `Ω_in` | entry direction (paper uses polar coordinates) |

**Target (Eq. 11):** `y = (p_exit, Ω_exit, s)`

**Loss:** conditional flow matching,
`L(θ) = E‖v_θ(x_t, t) − ẋ_t‖²`, with `x_t` interpolating noise→data under
optimal-transport scheduling. Inference: draw `z ~ N(0,I)`, integrate
`dx/dt = v_θ(x, t, c)` from t=0 to t=1.

**Frame:** particles always enter the **left** face. Other faces are rotated
into this canonical frame before inference and rotated back after.

**Absorption is not an input.** The walk is pure-scattering; absorbing
problems are recovered afterwards by attenuating weight `exp(−σ_a·s)` and
tallying `Δw = w(1 − e^{−σ_a·s})`.

---

## 2. Preprocessing — REQUIRED

Training will not work sensibly without these. All are derivable from the
existing columns; **no regeneration needed**.

### 2.1 Encode `p_exit`

Two independent problems with the raw column:

* its range depends on `W` (runs 0 → 4W), so the same number means different
  things at different cell sizes → divide by `4W` first;
* once normalised it lives on a **circle** — `0.999` and `0.001` are
  physically adjacent but numerically far apart.

```
p̂ = p / (4W)                      # in [0,1)
encode as (cos 2π p̂, sin 2π p̂)    # 2 numbers, no seam
```

### 2.2 Rescale `s`

Raw `s` spans ~`3.5e-7` to `687` and correlates strongly with `W`. Since
`<s> = W` (Dirac invariance, verified), the ratio is O(1):

```
target = log(s / W)
```

Prefer this over `log s`, which stays badly conditioned across cell sizes.

### 2.3 Log-scale `W`

`W` spans 0.025 → 20. Linear scaling makes the network nearly blind to thin
cells. Use `log W`, then standardise.

### 2.4 Split train/val by ENTRY CONDITION, not by row

**This one fails silently.** Each entry state contributes `--per-cond` rows
(48 by default) with identical conditioning. A random row-wise split puts the
same condition in both train and val: the val loss looks good and the model
has partly memorised.

Split on the 4096 distinct entry conditions per `W` instead.

### 2.5 Leave alone

`y0` already [0,1]. `(oxi, oyi)` already on the unit disk in [-1,1].
`(oxo, oyo, ozo)` are unit-norm — feed all three and renormalise the model
output at sampling time. Avoids the wrapping problems a polar
representation introduces.

---

## 3. Preprocessing — STRONGLY RECOMMENDED

### 3.1 Split off the uncollided branch

At small `W` most particles cross without scattering. For them
`Ω_exit = Ω_in` **exactly** and `s` is the deterministic geometric chord —
a Dirac component that a smooth velocity field cannot represent. Training on
it smears the majority of thin-cell crossings.

`k == 0` identifies the branch exactly, so no regeneration is needed.

* train the network on `k > 0` only;
* at sampling time flip a coin with the analytic probability
  `exp(−d_chord)`, where `d_chord` is the entry-to-boundary distance along
  `Ω_in` in mfp, and return the analytic answer on success.

Measured uncollided fractions:

| W (mfp) | uncollided | usable (collided) rows of 196,608 |
|---|---|---|
| 0.025 | 96.7% | ~6,500 |
| 0.1 | 87.9% | ~23,800 |
| 1.0 | 38.7% | ~120,500 |
| 5.0 | 8.0% | ~180,900 |
| 20.0 | 2.3% | ~192,100 |

### 3.2 Decide per-`W` weighting

Generation is flat at 196,608 rows per `W`, but the table above leaves a 30×
imbalance after the split, worst exactly where the physics is most singular.
Rebalance according to which optical sizes actually matter — for the lattice
at pitch 1 only `W = 0.5` and `W = 1.0` occur.

---

## 4. Decisions to make

**Square cells only.** Our data has `H = W` throughout. If the architecture
takes `W̃` and `H̃` as separate inputs, either feed `W` into both slots (and
accept no generalisation to rectangles) or regenerate with independent `H`.
`sample_single_cell` already accepts `W` and `H` separately, so this is a
generator change, not a physics change. Square-only is sufficient to
reproduce the lattice benchmark.

**Discrete `W`.** The paper draws cell optical dimensions from a
*distribution*; we have 16 discrete values. A model conditioned on continuous
`W̃` but trained at 16 spikes has no guarantee of interpolating between them.
Fine for a proof of concept, not for a faithful reproduction — regenerate
with continuous `W` for anything written up.

---

## 5. Verification before training

Cheap checks that are painful to debug after a training run rather than
before:

* **round-trip the encodings** — decode `(cos, sin)` back to `p`, confirm the
  recovered point lies on the cell boundary and matches the original;
* **confirm the split leaks nothing** — no entry condition appears in both
  train and val;
* **sanity-check the uncollided branch** — for `k == 0` rows, verify
  `Ω_exit == Ω_in` to floating-point tolerance, and that `s` equals the
  geometric chord.

---

## 6. Dataset reference

`data/lattice_singlecell_coverage.npz` — 3,145,728 records, 16 optical sizes
in [0.025, 20] mfp, 4096 entry states per `W` × 48 transmissions each.
Entry states uniform on the half-disk `{Ω_x > 0, |Ω_xy| < 1}`, `y0` uniform.

Fields: `W, y0, oxi, oyi` (conditioning) and `p, oxo, oyo, ozo, s, k`
(targets). No per-record provenance — join `*_composition.npz` on `W` for
`(K, pitch, σ_s, σ_a, count)`, including the `σ_a` needed for attenuation.

Regenerate bit-identically with
`python scripts/make_lattice_dataset.py --n-cond 4096 --per-cond 48 --seed 777`
(~6 s).

**Caveat if you reweight to the physical cosine law:** the importance weight
`Ω_x/|Ω_z|` diverges at grazing incidence. ESS ≈ 1058 of 4096 — fine for
conditional training, noisy for reweighted estimates.

---

## 7. Not covered by this dataset

* **Internal (volumetric) births.** Boundary entry only. The hohlraum needs
  zero of these; the lattice needs them for one cell in `K²`. Simplest
  route is analog MC for the source cell rather than a second model — see
  the depth-conditioning idea if a unified model becomes interesting.
* **Rectangular cells** (§4).
* **Continuous `W`** (§4).
