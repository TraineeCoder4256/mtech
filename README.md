# Generative Monte Carlo for 2-D particle transport

A reproduction of **arXiv:2512.13965v1**, *Generative Monte Carlo Sampling for
Constant-Cost Particle Transport* (Farmer, Murray, Krotz, McClarren), built on
top of a 2-D monoenergetic Monte Carlo transport solver.

The idea under test: inside an optically thick, materially uniform cell, replace
the whole in-cell scattering random walk with **one draw from a conditional
generative model**. Monte Carlo cost grows with optical thickness (more
scatters per crossing); the generative sampler's cost is a fixed number of
network evaluations, whatever the thickness. Above some crossover thickness the
sampler wins.

```
mc2d/      the Monte Carlo baseline (numba, CPU) + the lattice geometry
gmc/       the generative boundary model: data encoding, network, CFM, sampler,
           and the transport driver that chains it across a real mesh
scripts/   every runnable step: dataset -> train -> evaluate -> joint tests ->
           end-to-end -> benchmarks
docs/      the write-up (markdown, HTML report, PDF)
data/      generated datasets (.npz)
models/    trained checkpoints
figures/   every figure produced by the scripts
```

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy numba matplotlib
```

Then install PyTorch **for your GPU**, which is the only dependency that differs
between a CPU box and a GPU box. Pick the wheel matching your driver from
<https://pytorch.org/get-started/locally/>; for a recent CUDA 12.x driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check the install before running anything else — this is the single most common
reason a "GPU run" silently executes on the CPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want:  2.x.y+cu124 True
# a version string ending in '+cpu' is a CPU-only wheel; reinstall.
```

Every script that touches the network takes `--device auto|cpu|cuda|mps`, and
`auto` (the default) picks CUDA if it is there. Each one prints the device and
the hardware it resolved to as its first line of output, so you can always see
what actually ran.

---

## 2. Reproducing everything, in order

Runtimes below are **measured on the 4-core CPU container this was developed
in**. The GPU column is what the work *should* cost given where the time goes;
`scripts/bench_gpu.py` (§3) measures it on your machine rather than guessing.
Steps 1 and 2 are pure numba/numpy and get no benefit from a GPU at all.

| # | Step | Command | CPU (4 cores) | GPU |
|---|------|---------|---------------|-----|
| 1 | Sanity-check the MC baseline and the lattice | `python scripts/lattice_explore.py` | ~2 min | same (no GPU work) |
| 2 | Generate the training dataset | `python scripts/make_lattice_dataset.py --out data/lattice_singlecell_coverage.npz` | ~25 min | same (no GPU work) |
| 3 | Inspect the dataset | `python scripts/inspect_lattice_dataset.py` | seconds | — |
| 4 | Train the boundary model | `python scripts/train_boundary_model.py --s-param detour --out models/boundary_v2 --device auto` | ~40 min | minutes, see §3 |
| 5 | Marginal evaluation | `python scripts/eval_boundary_model.py --ckpt models/boundary_v2 --device auto` | ~3 min | seconds |
| 6 | Joint: corner plot | `python scripts/joint_corner.py --ckpt models/boundary_v2 --ode-steps 5 --device auto` | ~4 min | seconds |
| 7 | Joint: conditional PDFs | `python scripts/joint_conditionals.py --ckpt models/boundary_v2 --ode-steps 5 --device auto` | ~4 min | seconds |
| 8 | Joint: whole-5-D tests | `python scripts/joint_whole.py --ckpt models/boundary_v2 --ode-steps 5 --device auto` | ~6 min | ~1 min (the C2ST classifier dominates) |
| 9 | Path-length encoding comparison | `python scripts/compare_sparam.py --device auto` | ~10 min | ~2 min |
| 10 | End-to-end lattice solve + speed sweep | `python scripts/gmc_end_to_end.py --ckpt models/boundary_v2 --ode-steps 5 --device auto --n 20000` | ~30 min | see the caveat in §4 |
| 11 | Hardware benchmark | `python scripts/bench_gpu.py --device auto --compile` | ~5 min | ~3 min |
| 12 | Rebuild the report/PDF | `python scripts/build_paper.py` | ~1 min | — |

Steps 4–11 all need step 2's dataset; steps 5–10 all need step 4's checkpoint.
Nothing else is ordered, so 5–9 can run concurrently if you have the memory.

### Reproducing the *paper's* baseline figures (no generative model involved)

```bash
python scripts/fig2b_singlecell.py     # single-cell exit distributions from MC
python scripts/fig3_benchmarks.py      # lattice and hohlraum flux fields
python scripts/fig4a_convergence.py    # MC convergence
python scripts/fig4b_scaling.py        # cost vs optical thickness
```

`fig3_benchmarks.py` will overlay digitized reference curves if you drop them at
`reference/paper_fig3_lineouts.npz`; without that file it plots our results
alone. Those reference curves are not in the repo because we did not digitize
them.

### Two checkpoints

`models/boundary_v1` and `models/boundary_v2` differ only in how the path length
`s` is encoded as the sixth output:

* **v1, `--s-param logW`** — `u = log(s / W̃)`. The physical constraint
  `s ≥ s_min(p)` (the path cannot be shorter than the straight line from the
  entry point to the exit point the model just generated) has to be *learned*,
  and where it fails the sampler clamps. That clamp fired on 4.22% of samples
  and put a spike in the joint distribution that the classifier test in step 8
  detects even though every marginal looks correct.
* **v2, `--s-param detour`** (default) — `u = log(s / s_min(p))`, so the decoder
  reconstructs `s = s_min(p)·exp(u)` and the constraint holds by construction.

Checkpoints written before v2 carry no `s_param` key and load as `logW`
automatically, so v1 still reproduces exactly. Step 9 puts the two side by side.

---

## 3. Does this code use a GPU efficiently?

Read this section before quoting any speed number.

### What was actually broken, and is now fixed

`GMCBoundarySampler` defaulted to `device="cpu"` and **no caller ever overrode
it**. On a GPU box, step 4 would have trained on the GPU while steps 5–10 —
including the entire speed comparison — silently ran on the CPU. Every entry
point now takes `--device` and prints what it resolved to. If you are picking
this repo up from an older copy, this is the change that matters most.

Two smaller things went with it: the ODE solver was allocating a fresh time
tensor at every stage of every step (a kernel launch per stage on a GPU) and now
builds them once; and the host→device copy of the conditioning vector is now
non-blocking.

### What I can and cannot tell you

**I could not measure any of this.** The container this was developed in has
`torch 2.13.0+cpu` and four CPU cores — there is no GPU here and I did not have
one at any point. Everything below is either an audit of the code or a
prediction from the arithmetic, and `scripts/bench_gpu.py` exists precisely so
you replace it with measurement:

```bash
python scripts/bench_gpu.py --device cuda --compile
```

It reports four things, and the numbers you get are the real answer:

1. **NFE/s versus batch size.** NFE — network function evaluations per second —
   is the whole cost model; everything else is this number divided by the
   evaluations one sample needs. The table shows where the curve flattens.
2. **How much is launch overhead.** The same FLOPs run as one dense matmul
   chain, as an achievable ceiling. A ratio near 1 means arithmetic-bound; well
   above 1 means the small kernels dominate and `--compile` should help.
3. **Solver cost at matched NFE** for Euler/Heun/RK4.
4. **Transport occupancy** — how fast the live particle batch decays during the
   end-to-end solve, which is where GPU utilisation is worst.

### My prediction, and the reasoning, so you can check it against the tables

The network is **1.68 M parameters**, so one evaluation is roughly
`2 × 1.68 M ≈ 3.4 MFLOP` per sample. That is *small*. The consequences differ
per workload:

**Training (step 4) — will use the GPU, but will not saturate a big one.**
At batch 4096 one forward is ~14 GFLOP; forward+backward ~41 GFLOP. A modern
datacentre GPU does that in single-digit milliseconds, which is the same order
as the ~50 kernel launches per forward pass. Expect to be launch-latency bound
and to see something like 10–30× over four CPU cores rather than 100×. **If a
run finishes suspiciously fast per step but the GPU sits at low utilisation,
raise `--batch` to 16384 or 32768** (and scale `--lr` up modestly); the step
count can stay the same because the dataset has 1.55 M training rows and 20 k
steps at batch 4096 is only ~53 nominal epochs.

**Bulk sampling (steps 5–9) — should use the GPU well.** These call the sampler
with tens of thousands of conditions at once, which is squarely in the saturated
part of the batch curve. Expect the largest speedups here, and expect them to be
uninteresting, because these steps are not the bottleneck.

**End-to-end transport (step 10) — this is where efficiency is genuinely poor,
and it is structural.** Three reasons, all visible in section 4 of the
benchmark:

* The live particle set shrinks at every cell crossing as particles leak out or
  fall below the weight cutoff. Early calls are large, late calls are tiny and
  land back in the launch-bound regime.
* There is a full host↔device synchronisation *per crossing*: the sampler
  returns to NumPy so the driver can do the geometry, rotations and tallies on
  the CPU.
* That inter-crossing geometry work is single-threaded NumPy and does not move
  to the GPU at all, so Amdahl's law caps the achievable gain.

The practical mitigation is to run step 10 with a large `--n` (100 k or more);
that raises the whole occupancy curve without changing anything else. The real
fix — keeping particle state resident on the device and doing the geometry in
Torch — is not implemented.

### The caveat that matters more than any of the above

**The Monte Carlo baseline is numba on the CPU and has no GPU path.** Moving GMC
to a GPU and leaving MC on the CPU does not measure the algorithm; it measures
the hardware gap between an A100 and four cores, and it will move the crossover
thickness dramatically in GMC's favour for reasons that have nothing to do with
generative modelling.

So when you run step 10 on a GPU, report **both**:

* **GMC-on-CPU vs MC-on-CPU** — the honest algorithmic comparison, and the one
  that belongs in any claim about where the crossover is.
* **GMC-on-GPU vs MC-on-CPU** — the deployment comparison. Legitimate to
  report, but only with the hardware on both sides stated in the same sentence.

The reference paper has the same asymmetry, which is part of why its wall-clock
numbers and ours differ; see `docs/gmc_report.html` for that decomposition.

---

## 4. Reading the outputs

| Output | What good looks like |
|--------|----------------------|
| `models/*/loss_curve.png` | train and val curves overlapping; a val curve above train means the condition-wise split is catching real overfitting |
| `figures/gmc_boundary_eval.png` | model marginals on top of MC marginals at several `W̃` |
| `figures/joint_corner_W1.png` | lower triangle: MC filled, GMC contours on top. Upper triangle is GMC − MC and should be structureless noise |
| `figures/joint_conditionals_W1.png` | every conditional slice overlays; TV distance ~0.02–0.05 is sampling noise at these counts |
| `figures/joint_whole_W1.png` | sliced-Wasserstein ratio near 1, and a **C2ST accuracy near 50%** — a classifier trained specifically to separate the two 5-D clouds cannot |
| `figures/sparam_comparison.png` | the clamp spike at `log10(s/s_min) = 0` present for v1 and gone for v2 |
| `figures/gmc_end_to_end.png` | flux field agreement against the MC-vs-MC statistical floor, plus the speed sweep and its crossover |
| `scripts/bench_gpu.py` stdout | §3 above |

The C2ST is the strictest test here and the one to trust: marginals can all
match while the joint is wrong, and a sliced test can miss a defect that a
classifier finds. It is what exposed the v1 clamp spike.

---

## 5. Known limitations

* `W̃` takes 16 discrete values in the dataset and is **not held out** in
  evaluation — held-out entry conditions are, but not unseen optical sizes.
  Generalisation across `W̃` is therefore untested.
* Cells are square (`H = W`) throughout. The conditioning vector has a separate
  `H̃` slot and the code paths are there, but no rectangular data exists.
* Only the **boundary** model is implemented. Internal births are handled by
  falling back to analog MC in the birth cell.
* The uncollided component is a Dirac delta that a smooth flow cannot represent;
  it is sampled analytically and the network never sees it.
* No distillation. The reference work's 1–2 NFE regime is not reproduced.
