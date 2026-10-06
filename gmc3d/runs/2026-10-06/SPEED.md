# Where gmc3d's time goes, and what a GPU can do about it

6 October 2026. Everything below was measured on this container: 4 Intel Xeon
cores at 2.1 GHz, 15 GB of RAM, **no GPU**, torch 2.13.0, numba 0.68.0, NumPy
2.5.3, OpenMC 0.16.0 in multigroup mode. The network was retrained here from
scratch (8,000 steps, 12.7 min, validation NLL 2.1831, matching the 2.18 of
5 October), and the lookup table rebuilt, so every timing is on the real
thing rather than an untrained stand-in.

Two roofs of this machine, measured first so the rest can be read against
them: dense single-precision matrix multiply peaks at **516 GFLOP/s**
(2048 x 2048), and main memory moves **47-51 GB/s** (a PyTorch add and a copy
over 160 MB arrays).

The raw logs and JSON are beside this file. New analysis scripts live in
`checks/` and change nothing in `core/` or `ball/`:

| script | question it answers |
|---|---|
| `checks/net_profile.py` | what one draw is made of: weights, flops, operations, bytes, and the call split stage by stage |
| `checks/net_variants.py` | candidate rewrites of the sampling path, each checked against the original before it is timed |
| `checks/precision_check.py` | what float16 and bfloat16 cost in accuracy, ball by ball |
| `checks/loop_profile.py` | how big the backend's batches are, what sets that, and what a free backend would cost |
| `checks/gpu_project.py` | what a measured cost per draw does to a whole run |
| `checks/error_bars.py` | whether a reported error bar is the real spread of repeated runs |

**Read section 9 first if you are about to quote a number from this file.**
Sections 1-8 were written before the error-bar estimator was fixed, and
their "x OpenMC" figures divide our 6 October timings by 5 October OpenMC
timings taken on a different machine. Section 9 has the corrected,
same-session figures and the evidence.

## 1. The result in one paragraph

With the network's cost per draw at the 300 ns that Phani measured on a
Colab T4 (compiled, 100,000 requests per call), gmc3d would reach OpenMC's
leakage error bar **2.3x faster on the sphere, 2.7x on curved, 1.5x on the
slab and 1.2x on the cask** -- against today's measured 0.46x, 0.99x, 1.05x
and 0.39x. Raising the particles in flight tenfold, which costs nothing and
changes no answer, lifts the slab to 2.1x and the cask to 1.4x. That is
**within 8-31% of what a completely free backend would achieve**. After
that the limit is not the network at all: it is the Monte Carlo transport
loop running on four CPU cores.

So GPU work is worth doing, it roughly doubles the headline numbers, and it
runs out of room quickly. The list in section 6 is ordered accordingly.

## 2. Where a ball run's time goes, and the ceiling

`checks/loop_profile.py`, 2,000,000 histories per problem, R* = 2, the
lookup table as the backend. The table costs about 10 ns per draw, so a
table run **is** what a free backend would do, with the right physics.

| problem | plain MC | with balls (free backend) | speed-up | balls per history | per-draw budget to stay within 10% of the ceiling |
|---|---|---|---|---|---|
| sphere | 6.98 s | 1.24 s | 5.62x | 1.00 | 62 ns |
| slab | 9.61 s | 5.06 s | 1.90x | 1.07 | 236 ns |
| cask | 15.42 s | 7.99 s | 1.93x | 2.18 | 183 ns |
| curved | 35.01 s | 25.19 s | 1.39x | 3.92 | 321 ns |
| nested | 23.95 s | 25.91 s | 0.92x | 0.27 | -- |
| lattice3d | -- | no ball of R >= 2 fits | 1.00x | 0 | -- |

The budget column is the number a GPU has to hit. It is **62 to 321 ns per
draw**, and today's CPU network is 3,400-8,600 ns.

Speed is not the whole verdict, because our solver and OpenMC have different
noise per history. The figure of merit, FOM = 1 / (relative error^2 x
seconds), is how many times faster a contender reaches the same error bar,
and it does not depend on how many histories were run. From the 5 October
race (4M histories, 40 batches, so the error bars are trustworthy), as a
multiple of OpenMC analog's FOM:

| problem | OpenMC analog | OpenMC implicit | our plain MC | free backend (table) |
|---|---|---|---|---|
| sphere | 1.00x | 1.33x | 1.23x | **3.31x** |
| curved | 1.00x | 2.12x | 2.01x | **3.13x** |
| slab | 1.00x | 1.43x | 1.18x | **2.53x** |
| cask | 1.00x | 0.58x | 0.79x | **1.67x** |
| nested | 1.00x | 0.73x | 0.66x | 0.77x |
| lattice3d | 1.00x | 0.99x | 0.58x | 0.54x |

Note what the FOM reveals that a wall-clock ratio hides: the ball step
replaces many flights with one jump, so it **adds variance per history**. On
the slab the free backend is 1.90x faster in time but its leakage error bar
is 14% wider, so the FOM gain is 2.14x rather than 2.53/1.18. On the sphere,
where one ball swallows a whole history, the error bar is 42% wider. The gain
is real, but it is smaller than the clock says.

## 3. What one draw is made of

`checks/net_profile.py`, trained network, batch 100,000:

```
weights          146,554  (32 Linear layers)
multiply-adds    142,368 per draw = 285 kFLOP
torch operations     772 per call
tensor bytes written  40,124 per draw  (reads are at least as much again)
```

Stage by stage, nanoseconds per draw:

| stage | ns | share |
|---|---|---|
| host encode (NumPy log and standardise) | 2.2 | 0.0% |
| copy the radii to the device | 0.0 | 0.0% |
| draw the noise | 6.6 | 0.1% |
| **the flow** | **6,533** | **~100%** |
| copy (mu, s) back | 0.8 | 0.0% |
| host decode (NumPy exp) | 16.2 | 0.3% |

**The host side is 18 ns per draw.** That is important: it means no amount of
GPU speed runs into a Python or NumPy wall. A 20 ns backend is still 90%
GPU. Everything that matters is inside the flow.

And the flow is not doing arithmetic efficiently. It runs at **21.8 GMAC/s =
43.6 GFLOP/s, which is 8.4% of this machine's 516 GFLOP/s matrix-multiply
peak**, while writing 6.1 GB/s of tensors (so roughly 12-18 GB/s read and
written together, a quarter to a third of the 47-51 GB/s roof). It is at
neither roof. The profiler says why:

| operation | count | ms at batch 100k | share |
|---|---|---|---|
| `aten::copy_` | 248 | 155.9 | 25.5% |
| `aten::addmm` (the matrix multiplies) | 32 | 134.1 | 21.9% |
| `aten::add` | 90 | 85.1 | 13.9% |
| `aten::silu` | 8 | 68.7 | 11.2% |
| `aten::native_layer_norm` | 12 | 52.3 | 8.5% |
| `aten::mul` | 180 | 44.7 | 7.3% |
| `aten::_softmax` | 12 | 14.9 | 2.4% |
| `aten::gather` | 36 | 8.7 | 1.4% |
| everything else | | 46.9 | 7.9% |

**The arithmetic the method needs is 22% of the cost.** The other 78% is the
elementwise and copy traffic around 32 small matrix multiplies, spread over
772 operations per call. That is the shape of the problem on a CPU and it is
the same shape on a GPU, where it shows up as kernel launches instead.

Two things this rules out. The spline arithmetic is **not** the problem:
replacing every spline with a plain affine map, keeping all the matrix
multiplies, saved only 12.5% (5,308 ns against 6,065). And the cost per draw
has a **minimum at 10,000 requests** on a CPU and rises again:

| requests per call | 1,000 | 3,000 | 10,000 | 30,000 | 100,000 | 300,000 | 1,000,000 |
|---|---|---|---|---|---|---|---|
| ns per draw | 7,709 | 5,084 | **3,413** | 4,283 | 6,318 | 7,469 | 8,618 |

Below 10,000 it is per-operation dispatch; above it the activations stop
fitting in cache. A GPU has the first half of that curve and not the second,
which is exactly why batch size is the first item in section 6.

## 4. What sets the backend's batch size

`core/transport.py` runs one statistical batch at a time. Inside a batch it
advances every live particle until each dies or wants a ball, then sends all
the waiting radii in **one** call -- a wave. So

    requests per call  =  balls per history x particles per batch / waves

and `waves` is how many ball steps a history takes one after another, which
is a property of the problem. Raising the particles per batch therefore
raises the backend's batch in proportion. Measured, 2,000,000 histories split
three ways (`checks/loop_profile.py`):

| problem | 20 x 100,000 | 5 x 400,000 | 2 x 1,000,000 | waves per batch | leakage |
|---|---|---|---|---|---|
| slab | 2,733 per call | 9,740 | 22,320 | 39 -> 48 | 0.86015 every time |
| sphere | 99,996 | 399,982 | 999,956 | 1.0 | 0.03176 every time |
| cask | 6,907 | 25,231 | 58,986 | 32 -> 37 | 0.00239 every time |
| curved | 12,853 | 47,806 | 112,004 | 31 -> 35 | 0.58049 every time |
| nested | 3,170 | 11,533 | 28,528 | 9 | 0.20364 every time |

Three things to read off it. The scaling is nearly proportional (it falls a
little short because thicker in-flight populations take a few more waves).
**The answer is identical to five figures**, so this costs nothing in
accuracy. And transport time itself drops 4-8% with bigger batches, because
there are fewer wave loops and the Numba threads stay fuller.

The one real caveat: the error *bar* is estimated from the scatter between
batches, so 2 batches give a useless estimate of it (the error on an error
bar is about sqrt(2/N)). The fix is to raise the total histories rather than
cut the batch count, or to interleave the batches so all of them are in
flight at once -- see item 1 of section 6.

The number of calls is also the number of GPU synchronisations: 784 for the
slab at 20 x 100,000, 96 at 2 x 1,000,000.

## 5. What the T4 numbers mean

Phani's measurement on a Colab T4 (6 October, untrained network of the same
size, so the same cost): compiled, **3,600 ns** per draw at 1,000 requests,
**520 ns** at 10,000, **300 ns** at 100,000 and above; eager 16,000 / 2,300 /
1,040. `checks/gpu_project.py` puts that curve onto the 5 October race
(FOM as a multiple of OpenMC analog's; "x N flight" means N times the race's
100,000 particles per batch):

| problem | our plain MC | T4 today (x1) | x4 | x10 | x40 | free backend |
|---|---|---|---|---|---|---|
| sphere | 1.23x | 2.27x (300 ns) | 2.27x | 2.27x | 2.27x | 3.31x |
| curved | 2.01x | 2.72x (492 ns) | 2.83x | 2.87x | 2.87x | 3.13x |
| slab | 1.18x | 1.45x (1,577 ns) | 2.03x | 2.11x | 2.21x | 2.53x |
| cask | 0.79x | 1.17x (729 ns) | 1.35x | 1.40x | 1.42x | 1.67x |
| nested | 0.66x | 0.75x | 0.76x | 0.76x | 0.76x | 0.77x |

With the network eager instead of compiled every one of those falls below
1.7x, and the slab below 1.0x, so compiling is not optional.

And here is how much is left to win by making the draw cheaper still, at
large batch where the cost is flat:

| problem | 300 ns | 150 ns | 100 ns | 50 ns | 20 ns | free |
|---|---|---|---|---|---|---|
| sphere | 2.27x | 2.70x | 2.87x | 3.08x | 3.21x | 3.31x |
| curved | 2.87x | 2.99x | 3.04x | 3.08x | 3.11x | 3.13x |
| slab | 2.21x | 2.36x | 2.41x | 2.47x | 2.50x | 2.53x |
| cask | 1.42x | 1.54x | 1.58x | 1.62x | 1.65x | 1.67x |
| nested | 0.76x | 0.77x | 0.77x | 0.77x | 0.77x | 0.77x |

Halving the cost from 300 to 150 ns is worth 5-19%. Making it free is worth
at most 46% (the sphere) and under 12% everywhere else. **The prize is
getting to 300 ns at the batch sizes the problems actually produce, not
getting below it.**

## 6. The GPU optimisations, in the order they are worth doing

Each one says what it buys, what it costs, and whether it has been measured
or only reasoned about. None of them has been landed.

**1. Keep more particles in flight.** Measured here, free, no accuracy cost:
requests per call grow 8-10x, the answers are identical to five figures, and
transport gets 4-8% faster as well. Worth slab 1.45x -> 2.11x and cask
1.17x -> 1.40x on the T4's own curve. The clean form is to interleave the
statistical batches instead of running them one after another, so the error
bars keep their 20-40 batches while every history is in flight at once; that
is a change to the loop in `core/transport.py` (allocate `particles x
batches` and carry a batch index per particle). The cheap form needs no code
change at all: `--particles 1000000 --batches 20`, i.e. more histories.
Memory for the clean form is about 140 bytes per particle, so 4M in flight
is roughly 550 MB.

**2. Capture the call as a CUDA graph.** The T4 costs 1,577 ns per draw at
the slab's 2,733 requests and 300 ns at 100,000. That 5x is launch overhead
over 772 operations, not work. One graph replay removes nearly all of it.
CUDA graphs need a fixed shape, so the request array is padded up to one of
a few fixed sizes. Implemented and verified in
`checks/net_variants.py --which graphs`: `torch.compile(mode=
"reduce-overhead")` plus sqrt(2) buckets, which waste at most 6% of the
batch (powers of two waste up to 41%). Numerically it agrees with the
original to 5e-6, pure floating-point reassociation. **Not measured on a
GPU** -- on this CPU there are no graphs, and `reduce-overhead` is simply
slower there (5,892 ns against 4,323 for plain compile). This is the item to
test first on Colab. If it reaches 300-400 ns at 2,700 requests, it does for
the slab what item 1 does, without needing more particles.

**3. float16.** Measured on CPU: **1.45-2.19x** faster (3,140 ns against
6,866 at 12,853 requests). Measured for accuracy with
`checks/precision_check.py`, 200,000 draws at each of six radii: its errors
are indistinguishable from float32's -- mean exit cosine within 0.15%,
mean path within 0.7%, largest CDF gaps 0.0025-0.0063 against a noise level
of 0.0043, and surviving-weight errors no larger than float32's at every
absorption ratio tested. On a T4 it also halves the traffic and uses the
tensor cores, so expect at least the CPU's 1.5-2x. Worth 300 ns -> ~150 ns:
sphere 2.27x -> 2.70x, slab 2.21x -> 2.36x.

**4. Compile the flow by default.** Measured on CPU: 1.25-2.30x, best at
small batches (1.95x at 2,733 requests). On the T4, Phani measured 3.5-4.4x.
`checks/gpu_bench.py` already has a `--compile` flag; the network backend
itself does not, so a real run cannot use it. Making it the default for
`Network` (with a flag to turn it off) is a few lines in `ball/network.py`.

**5. Run the network at R* = 2.** Not a GPU change, a flag, but it only
makes sense once the draw is cheap. A ball pays when the draw costs less
than the flights it saves; a flight is 175-306 ns, so at 300 ns per draw the
break-even is about 1.3 scatters inside the ball, which is R of about 1.5.
The race ran the network at R* = 3 and R* = 10 because on a CPU it had to.
Every projection above already assumes R* = 2.

**6. Overlap the draw with transport.** `core/transport.py` advances, then
collects, then draws, then waits -- 600-800 times per run with nothing
overlapping. If the CPU advanced one half of the particles while the GPU
drew for the other, the run would cost max(transport, draws) instead of the
sum: sphere 1.84 s -> 1.24 s (33%), cask 10.78 s -> 7.99 s (26%), slab
5.70 s -> 5.06 s (11%). Reasoned from the measured split, not measured. It is
a genuine change to the solver's loop and the most likely of these to
introduce a bug, so it belongs after 1-4.

**7. Decode on the device.** Keep log(R) and the standardisation on the GPU
and send back (mu, s) rather than the raw targets. Measured ceiling: the host
work is 18 ns per draw, so this is worth at most 6% at 300 ns -- and 47% at
20 ns. Only worth doing if 1-4 work out.

**8. A smaller flow.** Out of scope here (it is a model change, and the
Development and Innovation threads own the model), but the cost scales with
it almost exactly. From 5 October, untrained, batch 100,000 on CPU: 4 layers
of width 32 (25k weights) 1.5 us, 4 x 16 (8.7k) 0.9 us, 2 x 16 (4.5k)
0.39 us, against 6.3 us for the 146k-weight flow. Accuracy at those sizes is
unknown -- none of them has been trained.

## 7. What is not a fix, with the measurement that says so

- **bfloat16.** Fast (1.75-2.01x on this CPU, which has AMX and
  avx512_bf16) but **wrong**: largest CDF gaps 0.007-0.020 against a 0.0043
  noise level, mean exit cosine off by +0.415% (z = +4.4) at R = 29, and
  NaN in the mean exit cosine at R = 1.5 and R = 3. Rejected on accuracy.
  (A T4 has no bfloat16 anyway; this matters for Ampere and later.)
- **Removing the copies from `Coupling.forward`.** `aten::copy_` is the
  single biggest line in the profiler at 25.5%, and `out = x.clone();
  out[:, bend] = yb` looks like the culprit. Rewriting the sampling path to
  carry the two targets as separate 1-D tensors and stack them once is
  **bit-for-bit identical** (difference 0.00e+00) and measured
  **0.96-1.17x** -- that is, nothing, inside the +-8% run-to-run spread. The
  copies move into the stack and the column gathers rather than disappearing.
  Kept in `checks/net_variants.py` as the `columns` variant because it is
  the base the compiled and float16 variants build on, but it buys nothing
  on its own.
- **Rebuilding the spline knots with one cumulative sum** instead of a pad
  and two subtractions: 0.97-1.21x, inside the same spread.
- **Bigger batches on a CPU.** The cost per draw bottoms out at 10,000
  requests and is 2.5x worse at 1,000,000. Item 1 of section 6 is a GPU
  optimisation specifically; on a CPU-only machine it is close to neutral
  (it still buys the 4-8% in transport).
- **Any amount of GPU work on `nested` or `lattice3d`.** With a completely
  free backend they are 0.77x and 0.54x of OpenMC. Almost no ball of
  R >= 2 fits, so there is nothing to replace. These are the negative
  controls and they behave as designed.

## 8. Limits that stopped this here

- **No GPU in this container.** `nvidia-smi` is absent and
  `torch.cuda.is_available()` is False, so every GPU number in sections 5
  and 6 is either Phani's Colab T4 measurement or a projection from it. The
  CUDA-graph item in particular is **unmeasured**: on a CPU
  `reduce-overhead` has nothing to capture. `checks/net_variants.py
  --device cuda` and `checks/precision_check.py --device cuda` run on Colab
  with no OpenMC needed.
- **The multipliers in sections 1, 2 and 5 mix two machines, and should
  not be quoted.** They divide 6 October measurements of ours by 5 October
  OpenMC rows, and OpenMC turns out to run 32-50% slower on this container
  than on that one. Section 9 has the evidence and the corrected,
  same-session figures. The ranking of the GPU options is unaffected, since
  it depends only on our own side.
- **Error bars from 20 or 40 groups are noisy.** This was fixed on
  6 October, after the rest of this file was written; section 9 describes
  the fix and what it was worth. The error-bar column of the three-way
  split in section 4 predates it and should not be read as a measurement.
- **The top of the network's trained range is unreliable, in any
  precision.** At R = 29 (trained to 30), the surviving weight is +1.4% high
  at absorption 0.01 per mean free path and +74% high at 0.1, in float32;
  float16 makes the latter +136%. The absolute numbers there are tiny, so it
  did not show up in any whole-problem answer, but a run that leans on big
  balls in a strongly absorbing material would need the range extended or
  `r_max` lowered. This is a training question, not a speed one.
- **This container's CPU is shared.** Repeated timings of the same thing
  spread by about 8%, so none of the 1.0x-1.2x differences above are
  meaningful on their own.

## 9. The error bars under all of this, and what two machines did to them

Written after sections 1-8, and it corrects them. Everything here is in
`runs/2026-10-06/error_bars.{log,json}`, `ref_batches.log`,
`race_repeat.{log,json}` and `race_fixed.{log,json}`, with the scripts that
produced them beside those files.

### The defect

A figure of merit is `1 / (relative error^2 x seconds)`, so it depends on
the SQUARE of an error bar. An error bar estimated from G independent
groups is itself uncertain by about `1 / sqrt(2 (G - 1))`, and a FOM by
`sqrt(2 / G)`. `core/tallies.py` estimated ours from the 40 statistical
batch means: 11% on the error bar, 24% on the FOM. The reference side was
worse, because `checks/race.py` ran OpenMC at the same 40 batches.

That is not a rounding detail. The slab's free-backend FOM read 2.53x on
5 October and 1.29x on the first 6 October race, and the whole difference
was OpenMC's own reported error bar moving from 0.0219% to 0.0153% on the
same problem with the same number of histories.

### The fix

Our solver already splits every batch across 32 thread chunks, each of
which is an independent sample of the same distribution. Taking the groups
from the chunks instead of the batches gives 32 x 40 = 1,280 groups instead
of 40, for free, from numbers the run already had. Unequal chunks are
handled with the weighted form, `sigma^2 = sum n_g (x_g - X)^2 / (G - 1)`
and `SE = sqrt(sigma^2 / N)`, accumulated from running sums so nothing is
stored per group. The MEAN is untouched, and
`tests/test_error_bars.py::test_mean_does_not_depend_on_the_grouping`
pins that. The old estimate is still computed and reported as
`leakage_se_batches` so the two can always be compared.

Validated by brute force in `checks/error_bars.py`: R independent runs,
seeds 1,000 apart, comparing each estimator against the actual spread of
the R run means.

| problem | truth (SD of run means) | new estimate | old | new stability | old stability |
|---|---|---|---|---|---|
| slab | 0.000305 | 0.000419 | 0.000415 | **2.8%** | 13.9% |
| sphere | 0.000149 | 0.000193 | 0.000185 | **3.2%** | 17.7% |
| cask | 6.72e-05 | 5.69e-05 | 5.71e-05 | **3.1%** | 15.8% |
| curved | 0.000411 | 0.000421 | 0.000425 | **2.8%** | 14.7% |

Stability is the run-to-run spread of the error bar itself; theory says 2.8%
from 1,280 groups and 16.2% from 40, and both columns land on it. The
apparent 1.37x bias on the slab at 30 runs was itself noise: at 150 runs it
is 0.92x with the truth known to only +-6%, and coverage of the 95%
interval is 94% against 91% for the old estimator.

### The reference side needed the same thing, and it is free

Splitting OpenMC's 4M histories into 1,000 batches instead of 40 changes its
answer in the sixth figure and its time not at all: sphere 13.02 / 12.13 /
12.43 s at 40 / 200 / 1,000 batches, slab 12.05 / 12.94 / 13.12, cask
21.58 / 19.98 / 22.68. `checks/race.py` now sets `REF_BATCHES = 1000`.

### What is left, and it is the timing

With both sides fixed, the same race was run at three seeds. Each seed is an
independent realisation of both codes and an independent timing.

| problem | contender | FOM x OpenMC analog, 3 seeds | spread | rel err, 3 seeds | spread |
|---|---|---|---|---|---|
| sphere | OpenMC implicit | 0.93 / 0.99 / 1.03 | 9.9% | 0.2013 / 0.2073 / 0.2054% | 3.0% |
| sphere | ours, plain MC | 1.74 / 1.97 / 1.84 | 12.5% | 0.1937 / 0.1959 / 0.1855% | 5.4% |
| sphere | ours + table, R* = 2 | 5.26 / 5.79 / 4.64 | 22.0% | 0.2581 / 0.2700 / 0.2726% | 5.5% |
| sphere | ours + network, R* = 2 | 0.40 / 0.45 / 0.37 | 18.4% | 0.2683 / 0.2677 / 0.2742% | 2.4% |
| slab | OpenMC implicit | 1.20 / 1.27 / 1.13 | 12.3% | 0.0156 / 0.0151 / 0.0158% | 4.5% |
| slab | ours, plain MC | 1.29 / 1.23 / 1.27 | 5.0% | 0.0157 / 0.0154 / 0.0149% | 5.2% |
| slab | ours + table, R* = 2 | 2.43 / 2.40 / 2.07 | 15.5% | 0.0156 / 0.0156 / 0.0157% | 0.6% |
| slab | ours + network, R* = 2 | 0.60 / 0.58 / 0.57 | 6.1% | 0.0160 / 0.0156 / 0.0154% | 3.9% |

The relative errors now repeat to 0.6-5.5%, which is the fix working. The
FOMs still move 5-22%, and the reason is the other factor: the seconds
spread 3.8-18.7% on this shared container. **Statistics is no longer the
limit on a multiplier; wall-clock noise is.** A published multiplier
therefore needs the median of at least three repeats, and 15% should be
quoted on it.

### The thing that invalidates every cross-day number

`openmc_transport_s` is OpenMC's own internal transport timer out of the
statepoint, not a wall clock. By that timer, on the same 4M histories, the
same OpenMC 0.16.0 (commit 617d35a5) and the same models:

| problem | 5 Oct container | 6 Oct container |
|---|---|---|
| sphere | 8.38 s | 11.90 - 12.71 s |
| slab | 9.68 s | 12.21 - 14.11 s |
| cask | 14.89 s | 19.98 - 22.68 s |
| curved | 35.16 s | 45.45 - 55.49 s |

Our own rows did not move over the same rebuild: sphere plain MC 14.22 s
then and 12.66-14.02 s now, slab table 9.40 s then and 9.10-10.45 s now. So
this is not the container being generally slower; it is OpenMC's
branch-heavy CSG transport meeting a different host, while our tight numba
kernel does not care.

One suspect ruled out: conda-forge now prefers the DAGMC variant, so the
rebuilt environment may not hold the build the 5 October runs used. Timed
on the sphere's own input at 4 threads, the DAGMC build runs at 327-337k
particles/s and the plain `nodagmc` build at 264-289k, so what we have is
the faster of the two, and neither reaches the 477k/s the 5 October timing
implies. The variant is not the explanation, and nothing in the container
recorded the 5 October CPU, so there is no way to say which machine is
representative.

**So compare only rows measured in the same session.** The sphere
lookup-table run is 3.20x OpenMC analog on the 5 October machine and
4.6-5.8x on this one. Both are honest; mixing them is not. Corrected
same-session figures, median of three seeds, FOM against OpenMC analog:

| problem | OpenMC implicit | ours plain MC | ours + table (free backend) | ours + network (CPU) |
|---|---|---|---|---|
| sphere | 0.99x | 1.84x | **5.26x** | 0.40x |
| slab | 1.20x | 1.27x | **2.40x** | 0.58x |

Against OpenMC implicit, the like-for-like algorithm, the free backend is
5.3x on the sphere and 2.0x on the slab.

## 10. Reproduce

From `gmc3d/`, with `/opt/mm/root/envs/gmc/bin/python` (the environment is
rebuilt with `micromamba create -p /opt/mm/root/envs/gmc -c conda-forge
python=3.12 openmc numba pytorch-cpu numpy`):

    python run.py table                                  # 3 s
    python run.py train                                  # 12.7 min here
    python -m checks.net_profile --device cpu --json runs/2026-10-06/net_profile_cpu.json
    python -m checks.net_variants --device cpu --batches 2733 12853 --buckets fine
    python -m checks.precision_check --device cpu --n 200000
    python -m checks.loop_profile --r-star 2 --histories 2000000 \
        --json runs/2026-10-06/loop_profile.json       # about 7 min
    python -m checks.gpu_project --costs runs/2026-10-06/t4_compiled.json --from-race
    python -m checks.error_bars --runs 30                # about 12 min
    python runs/2026-10-06/ref_batches.py                # about 3 min
    python runs/2026-10-06/race_rstar.py                 # the fixed race
    python runs/2026-10-06/race_repeat.py                # 3 seeds, 10 min
    python -m pytest tests -q                            # 41 tests

On a machine with a GPU, the three that matter and need no OpenMC:

    python -m checks.net_profile  --device cuda --compile
    python -m checks.net_variants --device cuda --batches 2733 6907 12853 100000 --buckets fine
    python -m checks.precision_check --device cuda
