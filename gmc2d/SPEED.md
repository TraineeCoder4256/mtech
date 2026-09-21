# Why it is slower, and what would make it faster

`evaluate.py` measures whether the sampler beats Monte Carlo, and on this
problem it does not. This file says why, from `results/speed_profile.txt` and
`results/speed_prototype.txt`, and what each candidate fix is actually worth.

Every number below was measured on one machine on 21 September 2026. Run the
two scripts to reproduce them; they write both a report and the raw JSON.

## The machine, stated first because it changes the answer

Intel Xeon at 2.8 GHz, 4 cores, **no GPU**. Both sides run on the CPU, so the
comparison is not measuring hardware against hardware — but it is also not a
per-core comparison: `mc.py` is numba-compiled across all four threads, the
GMC host loop is single-threaded NumPy, and only the network's matrix
multiplies are threaded. A large dense matmul reaches 521 GFLOP/s here, which
is the ceiling any arithmetic claim below is measured against.

## The result

The 7×7 lattice at the published cross sections, 20,000 particles per solve:

| | |
|---|---|
| Monte Carlo | 0.029 s |
| GMC | 9.8 s |
| ratio | **337× slower** |

Across repeated runs the ratio lands between 300× and 390×; the spread is
timer and thread noise, not a trend.

**Accuracy is not what is wrong.** GMC's relative L2 against the MC field is
0.94%, which is *below* the 1.45% that two MC runs differ by at the same
particle count. Systematic bias, with the noise averaged out over several
fields, is 0.69%. Total flux is conserved to 1.0003. Whatever is traded away
for speed, it is not being traded against a model that was already marginal.

## Where the time goes

```
network (ODE solve)             9.657 s   98.6 %
analog MC in the birth cell     0.001 s    0.0 %
geometry / tallies (NumPy)      0.139 s    1.4 %
```

So it is the model, not the Python. The `gmc2d/README.md` posed this as a
fork in the road — network-dominated means the model is too expensive,
host-dominated means no model work will help — and the answer is
unambiguous.

But the second half of that dichotomy is not quite right either. Replace the
velocity field with a function that returns zeros, keeping every shape, batch,
rotation and tally identical, and the same solve still takes **0.338 s, nine
times Monte Carlo's**. The host loop is not free. It is merely small next to a
network that is 30 times larger than it.

## The four bottlenecks, in order of size

### 1. The model costs 33 MFLOP per cell crossing, and the crossing is worth one scattering event

1,676,678 parameters, 3.33 MFLOP per network evaluation, 10 evaluations per
sample: **33.3 MFLOP per cell crossing**. At the published scale a crossing
contains 0.9 scattering events, and Monte Carlo spends 254 ns on one of those.
GMC spends 68 µs on the crossing.

The raw arithmetic ratio is about a million to one, and that number is
misleading in the sampler's favour — both sides run at wildly different
efficiency:

| | throughput |
|---|---|
| GMC network, batch 4,096 | 289 GFLOP/s |
| large dense matmul, same machine | 521 GFLOP/s |
| MC scattering loop | 0.12 GFLOP/s |

The network is at 55% of what this CPU can do on a perfect matmul, which is
ordinary for 256-wide layers on 6-wide data. The Monte Carlo walk is not doing
arithmetic at all: a random number, a logarithm, two divisions and a branch,
with the cost set by the dependent chain and the RNG. It leaves about 2,400×
of the machine's arithmetic capability unused — and that waste is the only
reason the crossover is a few hundred mean free paths away instead of a
million.

**What this rules out.** There is no implementation fix of any size here.
`torch.compile` at a fixed batch shape is worth 1.34×, `inference_mode`
nothing (the sampler already runs under `no_grad`), and bfloat16 is 2.9×
*slower* on this CPU because there are no bf16 kernels for it. The lever is
the flop count itself: fewer evaluations, or a smaller field.

### 2. At this scale there is nothing for a sampler to save

Lattice cells are 0.5–1.0 mean free paths. Half of all crossings are
uncollided and already handled analytically for free. The ones that do scatter
average under two events. The sampler is replacing one scattering event with
ten network evaluations.

The sharpest version of this: run the **identical host loop with the in-cell
walk done exactly** instead of generated — same rotations, same cell-averaged
tallies, same everything but the sampler — and it solves the problem in
**0.10 s against the sampler's 10.4 s**. It is 100× faster than the model it
is standing in for, at every thickness tested up to 20 mfp:

| cells | analog in the loop | GMC | fine-mesh MC |
|---|---|---|---|
| 1 mfp | 0.10 s | 10.4 s | 0.029 s |
| 4 mfp | 0.09 s | 13.5 s | 0.031 s |
| 10 mfp | 0.15 s | 20.5 s | 0.067 s |
| 20 mfp | 0.28 s | 29.9 s | 0.173 s |

Note also that the macro-cell driver *with an exact walk* is only 1.6–3.5×
slower than the fine-mesh solver, and the gap narrows as cells thicken,
because it skips the fine mesh crossings.

### 3. The call shape wastes about a quarter of the network time

Per-sample network cost against batch size, 4 threads:

| batch | 1 | 8 | 64 | 512 | 4,096 | 20,000 | 100,000 |
|---|---|---|---|---|---|---|---|
| µs/sample | 612 | 121 | 45 | 12.2 | **11.5** | 14.5 | 22.2 |

Both ends are bad, and the solve visits both. The crossing loop makes 400
sampler calls; 384 of them carry fewer than 512 particles. Those calls are
1.8% of the crossings and **12.9% of the network time**. At the other end, a
200,000-particle solve pushes batches past 4,096, where activations stop
fitting in cache, and the per-particle cost rises from 551 µs to 853 µs —
so simply running more particles makes it *worse*, not better.

Holding every crossing at the batch-4,096 cost would be worth about **1.24×**
on its own: tile large batches at a few thousand rows, and hand the long thin
tail of stragglers to analog MC rather than calling a network with one
particle in it.

### 4. The driver's own floor, which becomes the wall

The free-model run costs 16.9 µs per particle at the published scale — about
2 µs per cell crossing, roughly 10 µs of torch dispatch for 4,000 module calls
and 7 µs of NumPy geometry and tallies. That floor does not shrink when the
model does. It is why the projection below stops improving at 55 mfp however
cheap the network gets.

## What each change is worth

Measured, except where the last column says otherwise.

| change | network speedup | accuracy cost | needs retraining |
|---|---|---|---|
| ODE: heun 5 → euler 4 | 2.2× | L2 1.11% → 2.88% | no |
| ODE: heun 5 → euler 2 | 4.1× | L2 → 4.97% | no |
| ODE: heun 5 → euler 1 | 6.1× | L2 → 7.26% | no |
| `torch.compile`, fixed batch shapes | 1.34× | none | no |
| batch tiling + tail cutover | ~1.24× | none | no |
| thin-cell threshold | up to 100× | none — it improves | no |
| width 128, depth 3 | 4.3× | **unmeasured** | yes |
| width 64, depth 3 | 8.9× | **unmeasured** | yes |
| distillation to 1 NFE | 10× | **unmeasured** | yes |
| a GPU | not measurable here | none | no |

Two of these deserve a note.

**The ODE ladder is not monotonic.** rk4 with 5 steps costs 20 evaluations,
twice heun-5, and its lattice L2 is 2.00% against heun-5's 1.11% — worse, and
both are inside the noise band. Past about 4 evaluations the discretisation
has stopped being the limiting error; more steps buy nothing. Single-cell
Wasserstein distances tell the same story: 0.04 at 10 evaluations, 0.02 at 20.

**The thin-cell threshold costs no accuracy because it is not an
approximation.** Cells below a chosen optical width go back to the exact
walk, which is the reference the whole project is scored against. It is the
one change that makes the code both faster and more correct.

## The one number that decides everything

Monte Carlo's work in a cell is the number of scattering events, and that is
fixed by a theorem `validate.py` already checks: the mean path of a particle
through a convex region is `4V/S` **whatever the scattering does**. In optical
units that path *is* the collision count. So one network call replaces about
one optical mean chord's worth of scattering events — for a `W x H` cell,
`2WH/(W+H)`, and for a square, `W`.

Measured with exact walks through the solver's own loop:

| cells (mfp) | 10 | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|---|
| scatters per crossing | 9.3 | 19.9 | 41.3 | 85.8 | 176.5 | 354.6 |
| ratio to the chord | 0.93 | 0.99 | 1.03 | 1.07 | 1.10 | 1.11 |

Now write the speedup out. Both methods make the same cell crossings, so

```
    MC time     crossings x chord x t_scatter     chord x t_scatter
  --------- = ------------------------------- = -------------------
   GMC time      crossings x cost_per_crossing    cost_per_crossing
```

**The crossing count cancels exactly.** Geometry, mesh, how many times a
particle bounces between neighbours — none of it survives. What is left is
one comparison: the optical mean chord of a cell, times Monte Carlo's cost per
scattering event, against GMC's cost per crossing.

That last one is worth knowing on its own, because it is easy to get wrong.
Crossings per particle are *not* constant in optical thickness — measured with
exact walks they go 8.0, 12.0, 20.0, 35.9, 67.6, 124.7 at 10 to 320 mfp, as
W^0.80, because a thick scattering cell usually returns a particle through the
face it entered. The cost per *history* therefore grows. It just cancels
against Monte Carlo's work growing the same way.

## Where the crossover is

`cost_per_crossing` is 124 µs measured, of which 2 µs is driver. `t_scatter`
falls from 254 ns at 1 mfp to 23 ns at 160 and flattens there. So break-even
needs a cell whose optical mean chord is:

| configuration | cost per crossing | chord needed |
|---|---|---|
| as it stands (heun 5, 1.7M params) | 124 µs | ~5,000 mfp |
| euler 4 steps | 57 µs | ~2,300 mfp |
| + width 128, depth 3 | 15 µs | ~590 mfp |
| distilled to 1 NFE + width 64, depth 3 | 3.4 µs | ~135 mfp |
| an instantaneous model — the driver alone | 2 µs | **~80 mfp** |

Checked directly: measured speedups on the lattice are 0.0033, 0.0058, 0.0104
and 0.0200 at 10, 20, 40 and 80 mfp, which extrapolate to unity at about
5,900 mfp — the same place the chord law puts it.

**So the lattice is not badly shaped, it is thin.** A 1 mfp cell is worth one
scattering event no matter how the mesh is drawn. The way to a large chord is
a physically large uniform region or a very large cross section, since the
chord is `(2/3)L x sigma` for a cube of side `L`. Merging four cells into one
doubles the chord and halves the crossings, which is a real 2x — but the
lattice alternates material every centimetre, so there is nothing to merge.
That is why the problem, not the mesh, is what has to change.

## The accuracy budget, so a tradeoff can be priced

Run the host loop with exact walks and whatever error survives is the
driver's, not the model's:

| cells | driver alone | GMC | MC-vs-MC floor |
|---|---|---|---|
| 1 mfp | 1.51 % | 0.98 % | 1.47 % |
| 4 mfp | 1.14 % | 1.20 % | 0.94 % |
| 10 mfp | 1.42 % | 1.61 % | 0.94 % |
| 20 mfp | 2.94 % | 2.39 % | 1.62 % |

At every thickness the model's field is indistinguishable from an exact walk
through the same loop, and both sit at or slightly above the statistical
floor. **The model contributes no measurable error of its own.** That is the
budget available to spend on speed — and it says a cheaper model is worth
trying before a more accurate one.

It also locates the residual: the driver carries about 1–3%, growing with
thickness. Candidates are the cell-averaged tally, grazing entry directions
clipped to `Omega_x > 1e-6` in `solve.py`, and the 400-crossing cap. None of
them is a model problem.

## Suggested order of work

1. **Stop calling the network where it cannot win.** A threshold on optical
   width, with thin cells walked. Costs nothing in accuracy, needs no
   retraining, and at the published scale it is the whole 100×.
2. **Fix the call shape.** Tile batches at a few thousand rows; cut the tail
   over to analog MC instead of calling the network with one particle.
   Worth ~1.24×, free.
3. **Drop to 4 network evaluations** if 2.9% against a 1.5% floor is
   acceptable for the study. Worth 2.2×, free, reversible.
4. **Then shrink the model** — this is the large one, 4–9×, and the only
   question is accuracy, which the budget above says there is room for.
   Belongs to whoever owns `model.py`.
5. **Then distil to 1–2 evaluations**, which is the regime the paper actually
   reports and this reproduction does not.
6. **Add a test problem the method can win on** — a uniform region whose
   optical mean chord is in the hundreds of mean free paths. This matters as
   much as any of the above: no amount of model work makes a 1 mfp cell worth
   generating, because it is worth one scattering event. Note that `make_data.py`
   stops at `SIZE_MAX = 20.0`, so at 40 mfp and beyond the model is already
   extrapolating, which is the likeliest reason accuracy degrades there.
   Training data for thick cells costs exactly what the method is trying to
   avoid — it has to be budgeted, though it is paid once.
7. **Only then** is the driver worth rewriting, and by then it will be the
   whole cost.

## What is not a fix

- **bfloat16 on this CPU.** 2.9× slower; no kernels for it here. On a GPU, or
  a CPU with AMX, a different answer.
- **More particles.** Past batch 4,096 the per-particle cost rises.
- **More ODE steps.** rk4 costs twice heun-5 and is no more accurate.
- **A slower baseline.** `mc.py` is numba-compiled and multi-threaded, which
  is what makes the target honest.
