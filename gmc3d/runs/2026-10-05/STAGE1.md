# Stage 1 results, 5 Oct 2026

Stage 1 of the SRS covers any CSG geometry, one energy group, isotropic
scattering and a fixed source. Everything below ran on this container: 4 CPU
cores, no GPU, OpenMC 0.16.0 in multigroup mode as the reference. The raw logs
and JSON sit beside this file.

## 1. What was built

The file map is in `gmc3d/README.md`. In one line each:

- **Geometry.** OpenMC's CSG (planes, spheres, cylinders, cones; nested
  universes; translations and rotations; rectangular lattices) is read straight
  from an `openmc.Model`, with a distance field that never overestimates.
- **Plain Monte Carlo.** Flights are drawn from the scattering cross section,
  absorption is applied as a continuous weight decay (as in gmc2d), and Russian
  roulette ends low-weight particles. Every particle has its own random stream,
  so a run repeats bit for bit.
- **Ball step.** After a scatter, if the biggest one-material ball has radius
  R >= R* (in mean free paths), one backend call moves the particle to the
  ball's edge. The backend gives the exit cosine mu and the path length s; the
  driver handles everything else.
- **Three backends.**
  - the exact walk (`oracle`)
  - a lookup table: 96 radii, 16,384 walks each
  - a normalizing flow: 146,554 weights, trained in 6.5 min on 4.5M exact
    walks
- **Checks.** A comparison with OpenMC, problems with exact answers, a
  ball-by-ball backend score, and a speed study.

## 2. Is it right? Yes

### Problems with exact answers (`validate.log`)

| check | plain MC | with exact balls (R* = 1) | exact |
|---|---|---|---|
| pure absorber leakage | 0.36787944 (error 1e-14) | same | e^-1 |
| infinite medium flux | 9.9988 +- 0.0072 | 9.9980 +- 0.0089 | 10 |
| infinite medium leakage | 0 | 0 | 0 |

### Against OpenMC, 2M histories each code (`validate.log`)

Leakage is shown with z = difference / combined error.

| problem | ours (plain) | OpenMC | z | ours (exact balls) | z |
|---|---|---|---|---|---|
| sphere | 0.031676 | 0.031436 | +1.5 | 0.031859 | +2.6 |
| slab | 0.86058 | 0.86078 | -0.6 | 0.86038 | -1.4 |
| curved | 0.58039 | 0.58051 | -0.4 | 0.58086 | +1.0 |
| nested | 0.20305 | 0.20416 | **-3.9** | 0.20308 | -3.5 |
| lattice3d | 0.033318 | 0.033363 | -0.2 | (no balls fit) | |

Three of these were looked at more closely:

- **Nested, -3.9.** Rechecked at 10M histories each: ours 0.20351 +- 0.00008
  against OpenMC 0.20356 +- 0.00015, so z = -0.3. Ten more seeds of ours
  scatter by 0.00032 against a reported error of 0.00029, so our error bars
  are honest. OpenMC's three nested runs (0.20327, 0.20416, 0.20356) show the
  2M-history reference run was itself about 2.7 sigma high.
- **Sphere and slab at 20M histories.** Ours and OpenMC agree: sphere leakage
  0.031744 vs 0.031762 (z -0.4); slab 0.860439 vs 0.860321 (z +1.2).
- **Lost particles:** 0 in every run. The geometry checks also passed: cell
  location matches OpenMC on 5 problems, and the distance field crossed no
  boundary in 10^6 test balls.

### Totals, cell tallies and material tallies

These are exact with balls on (the ball never leaves its cell). Mesh tallies
are not; see decision 2 below.

## 3. The backends, one ball at a time (`ballcheck.log`)

Each test compares 200,000 exact walks with 200,000 backend draws at the same R.

| R | table: mean path error | network: mean path error | table / network: largest CDF gap (noise 0.0043) |
|---|---|---|---|
| 1.5 | +0.22% | +0.45% | 0.0074 / 0.0065 |
| 5 | -0.37% | -0.03% | 0.0074 / 0.0025 |
| 10 | +0.54% | +0.53% | 0.0060 / 0.0025 |
| 29 | +0.15% | -0.58% | 0.0061 / 0.0048 |

- **Both backends are accurate to about half a percent.** The network matches
  the shape of the distributions better: its CDF gaps are closer to the noise
  level.
- **The table's error comes from reusing a fixed sample.** Its 16,384 stored
  walks per radius are reused millions of times. In the strongly absorbing
  sphere, where absorption per mean free path is 0.1, the weight that survives
  a ball depends on rare short walks. There the table's leakage came out 1.2%
  low (z about -3.6 against the 20M reference).
- **The network is slightly off at the top of its range.** At R = 29, near the
  edge of what it was trained on (R up to 30), its surviving weight is +1.4%
  (z 3.3) at absorption 0.01 per mean free path.

## 4. Is it faster? Sometimes, and not with the network as built (`measure.log`)

All runs use 1M histories. Speed-up is plain-MC wall time divided by
ball-run wall time. All ball runs agree with plain MC on leakage within
0.08% (slab, curved).

| problem | plain MC flights per history | best: oracle | best: table | best: network |
|---|---|---|---|---|
| sphere (one R = 10 ball per history) | 18.6 | 2.8x | **6.0x** | 0.53x |
| slab (60 cm shield, 30 mfp) | 22.7 | 1.66x (R* = 3) | **1.93x** (R* = 2) | 0.92x |
| curved (tank, sphere, cone) | 60.9 | 1.41x (R* = 2) | **1.47x** (R* = 2) | 0.96x |
| nested (pins, 3D lattice) | 44.1 | 0.95x | 0.97x | 0.96x |
| lattice3d (1 cm cells) | 8.2 | 1.0x | 1.0x | 1.0x |

**What one draw costs**, at the batch sizes the runs produced:

| backend | cost per draw | in plain flights |
|---|---|---|
| plain MC flight (for comparison) | 175-306 ns | 1 |
| table | 7-13 ns | 0.03-0.06 |
| exact walk, R = 5 (16.5 scatters) | about 210 ns | about 1 |
| network | 2.5-4.2 us | 8-24 |

**Throwaway timing of smaller flows.** These were untrained, since only the
time per draw was measured:

| flow | cost per draw at batch 3,000 | at batch 100,000 |
|---|---|---|
| 4 layers, width 32 (25k weights) | 2.1 us | 1.5 us |
| 4 layers, width 16 (8.7k weights) | 2.0 us | 0.9 us |
| 2 layers, width 16 (4.5k weights) | 0.97 us | 0.39 us |

## 5. Diagnosis

1. **Most of the gain is skipping geometry work, not skipping scatters.** A
   scatter inside the exact walk costs about 12 ns, because the walk happens
   inside a sphere and needs no geometry query. A plain flight costs 175-306 ns,
   because each one locates the particle and measures the distance to every
   nearby surface. That is why even the exact walk speeds things up.
2. **The ceiling is set by how many flights balls remove.**
   - In the slab at R* = 2, flights per history drop from 22.7 to 7.4. With a
     free backend that caps the speed-up near 22.7 / (7.4 + 1.07 balls) =
     2.7x; we reached 1.93x.
   - In nested and lattice3d, almost no ball of R >= 2 fits, so nothing can
     help there. The 1 cm cells in lattice3d are the negative control and
     behave as designed.
   - Checking whether a ball fits costs about 5-10% when no ball fits
     (speed-ups of 0.9-0.97x).
3. **The network is too expensive, not inaccurate.**
   - At 2.5-4 us per draw it pays only for balls with R >= 5-7. In these
     problems, balls that big are rare.
   - Shrinking it to 4.5k weights still leaves about 0.4-1 us per draw. That
     is PyTorch's per-operation overhead on many small spline operations, not
     arithmetic. The table, at about 10 ns, is 40-400x cheaper for the same
     job.
   - CORRECTION (later on 5 Oct): for the network as trained, that is wrong.
     One call is 1,249 operations moving about 100 kB per draw, and on 4 CPU
     cores the cost stays at 4-7 us per draw for batches from 3,000 to
     1,000,000 (torch.compile: about 3 us). It is arithmetic and memory
     traffic, which a GPU accelerates; see section 10.
   - This is the gmc2d speed finding again, but with a twist: with ONE input,
     a table is a strong competitor.

## 6. Limits that stopped it here

- **4 CPU cores, no GPU.** The network timings are CPU-only. A GPU would help
  big batches, but the batches here are 2,700-12,700 requests per call.
- **Error bars on the figure of merit are unreliable.** The figure of merit
  (1 / (variance x time)) in `measure.log` swings by up to 5x between runs
  with no balls at all, because 10 batches give noisy error bars. Use the
  speed-ups and the flights per history instead.
- **The "big-ball share" in `measure.log` is mislabelled.** It counts scatters
  inside exact balls against plain MC's scatters, which exceeds 100% when
  roulette would have ended walks early (the sphere: 310%). `measure.py` now
  labels it as such and also prints flights per history. The log above was
  produced before that relabel.
- **The network was trained once,** at one size and one seed, for 8,000 steps.
  Its validation loss had levelled off (2.18 nats).

## 7. Decisions for Phani (research items)

1. **R2: what should the ball backend be?** With one input, a table is
   about 300x cheaper than the network and about as accurate. The network's
   case is the later stages, where the inputs grow: an anisotropic scattering
   law (Stage 2), the in-group scatter fraction (Stage 3), and continuous
   energy (Stage 5).
   - Options:
     - keep the table for Stage 1 and test the network where tables stop
       scaling;
     - hand-code a tiny network inside the Numba loop (unproven; guessed at
       50-100 ns);
     - keep tuning PyTorch flows (0.4 us floor measured).
   - Recommended: the first.
2. **R4: where does a ball's flux go on a mesh tally?**
   - 'cap' is exact, but on a 1 cm mesh in a 2 cm-mfp slab it shrinks every
     ball below R* = 1, so balls are never used.
   - 'centre' keeps balls but smears the mesh badly: the slab's depth profile
     is off by up to 148 sigma.
   - A third option, not built: put the ball's flux at ONE random point drawn
     from the average track profile inside a ball of that R (and absorption
     ratio). The mesh is then right on average and full-size balls are kept.
     It needs a small two-input table of radial profiles.
   - Recommended: the third.
3. **R1: the threshold R*.**
   - The data say R* = 2 for the table: best on the slab and curved, no loss
     on nested.
   - For the network as built, no R* pays on these problems.
   - The default in `run.py` stays 3 until you choose.

## 8. Head to head with OpenMC (`race.log`), added later on 5 Oct

Every contender solved the same model with 4M histories on the same 4
cores. OpenMC ran in its default analog mode and with survival biasing
("implicit", the same weight game as ours). Seconds of transport:

| problem | OpenMC analog | OpenMC implicit | ours plain | + network (R* = 3) | + exact walk (R* = 2) | + table (R* = 2) |
|---|---|---|---|---|---|---|
| slab | 9.7 | 13.1 | 18.4 | 23.6 | 10.4 | 9.4 |
| sphere | 8.4 | 14.5 | 14.2 | 22.9 | 5.1 | 2.7 |
| curved | 35.2 | 55.1 | 74.4 | 99.4 | 53.0 | 52.1 |
| cask | 14.9 | 28.0 | 31.3 | 46.4 | 16.2 | 15.3 |
| nested | 24.4 | 33.3 | 51.4 | 57.0 | 55.3 | 55.1 |
| lattice3d | 7.6 | 7.2 | 8.9 | 9.7 | 9.5 | 9.6 |

Time to reach OpenMC analog's leakage error bar (FOM ratio, about +-30%):
exact-walk balls 2.87x (slab), 2.01x (sphere), 1.88x (curved), 1.70x
(cask), 0.77x (nested), 0.54x (lattice3d); the network 1.05x, 0.46x, 0.99x,
0.39x, 0.61x, 0.53x. `cask` is a new problem written as a template for
user geometry (`problems/cask.py`).

## 9. Reproduce

From `gmc3d/`, with `/opt/mm/root/envs/gmc/bin/python`:

    python run.py validate      # section 2 (about 4 min)
    python run.py table         # the lookup table (3 s)
    python run.py train         # the network (6.5 min)
    python run.py ballcheck --backends table network oracle   # section 3
    python run.py measure       # section 4 (about 25 min)
    python run.py race          # section 8 (about 20 min)
    python -m pytest -q tests   # 37 unit tests

## 10. What a GPU would change (projection, no GPU here)

Phani pointed out that every timing above is CPU-only. Estimated network
cost per draw on a GPU, from the operation count and traffic above: about
0.7 us with PyTorch as written at ~10k requests per call (launch-bound),
about 0.2 us at 100k+ requests per call, about 0.05 us with the flow fused
into one kernel. Projected onto the race (exact-walk runs at R* = 2, their
transport time and error bars kept), time to OpenMC analog's leakage error
bar becomes:

| problem | network on CPU (measured) | GPU 0.7 us | 0.2 us | 0.05 us | free network |
|---|---|---|---|---|---|
| sphere | 0.46x | 2.0x | 3.2x | 3.9x | 4.2x |
| slab | 1.05x | 2.5x | 3.0x | 3.2x | 3.3x |
| curved | 0.99x | 1.6x | 1.9x | 1.9x | 2.0x |
| cask | 0.39x | 1.4x | 1.8x | 1.9x | 2.0x |
| nested | 0.61x | 0.76x | 0.77x | 0.77x | 0.77x |
| lattice3d | 0.53x | 0.54x | 0.54x | 0.54x | 0.54x |

The ceiling is the transport loop on the CPU. `python -m checks.gpu_bench
--device cuda` measures the real cost per draw on a GPU and prints this
table; `python run.py race --device cuda` runs the race with the network
on the GPU.
