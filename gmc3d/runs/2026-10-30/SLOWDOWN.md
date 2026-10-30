# Why plain MC looks 3x slower than OpenMC, and where the 30% went

30 October 2026, speed track. Handed over from the Innovation track with two
claims to check: that plain MC on `3d-gmc` runs about 3x slower than OpenMC
on every problem (sector 203 s against 62 s), and that plain MC has been
5-16% slower since v1.3 with the cause not found.

Both are real, and both turned out to be something other than what they
looked like.

Everything below was measured in one session on one container (4 Intel Xeon
cores at 2.1 GHz, OpenMC 0.16.0, numba 0.68.0), 500,000 histories
(100,000 x 5) per run, best of 3 after a compile run, with git worktrees at
v1.2.0 (0013942), v1.3.0 (052abb5), v1.4.0 (3db14a2) and v1.4.1 (7b5ed2f)
checked out side by side. Absolute seconds here are NOT comparable with the
29 October logs: that was a different container, and OpenMC's own transport
timer has moved 32-50% between containers before (see
`runs/2026-10-06/SPEED.md` section 9). Only same-session comparisons are
used. Scripts and JSON are beside this file.

## 1. The 3x is mostly a comparison between two different algorithms

Our solver uses implicit capture: a particle is never killed by absorption,
its weight decays instead. OpenMC's default is analog, where absorption
kills. Analog therefore takes far fewer flights per history and less time,
and buys a bigger error bar for it. Comparing wall time across the two says
very little.

| problem | ours, plain MC | OpenMC analog | x analog | OpenMC implicit | x implicit | FOM vs analog |
|---|---|---|---|---|---|---|
| slab | 5.04 s | 2.28 s | 2.21x | 3.67 s | 1.37x | 0.72x |
| cask | 9.39 s | 3.80 s | 2.47x | 7.46 s | 1.26x | 0.76x |
| lattice3d | 3.00 s | 1.05 s | 2.86x | 1.07 s | **2.81x** | **0.44x** |
| hexcore | 13.58 s | 5.21 s | 2.61x | 7.76 s | 1.75x | 0.77x |
| sector | 30.82 s | 11.01 s | 2.80x | 19.61 s | 1.57x | 0.69x |

Three things follow.

**The 2.2-2.9x against analog is not new and not v1.4's fault.** The slab and
the cask have been in the suite since the start and sit at 2.21x and 2.47x,
right alongside the new problems. Whatever the figure is measuring, it is
not something hex lattices or tori introduced.

**Against OpenMC in the same mode we are 1.26-1.75x on four of the five.**
That is the fair wall-time comparison.

**By figure of merit we are 0.44-0.77x of analog**, that is 1.3-2.3x slower
to reach the same error bar. FOM is the only honest way to compare two
variance-reduction schemes, and it should be what gets quoted.

## 2. The real outlier is lattice3d, not sector

`lattice3d` is 2.81x even against OpenMC implicit and 0.44x by figure of
merit, and it has the most expensive flights of anything in the suite at
**736 ns** against the slab's 446.

`sector` is 1.57x against implicit. Its 30.8 s is mostly that it takes
**91.5 flights per history**, three times the cask and four times the slab;
its flights are expensive too (674 ns) but not the worst. So the tori the
Innovation track found and fixed were the right target for the per-flight
cost, and sector's remaining slowness is flight count, which is physics plus
the rotational walls reflecting particles back in, not geometry code.

| problem | ns per flight | flights per history |
|---|---|---|
| slab | 446 | 22.6 |
| cask | 469 | 40.0 |
| hexcore | 530 | 51.2 |
| sector | 674 | 91.5 |
| lattice3d | **736** | 8.2 |

Deep lattice geometry is what costs, which is what `runs/2026-10-14/CEILING.md`
predicted: `_descend` walks the CSG tree from the root on every flight, and
the deeper the tree the worse it is.

## 3. The regression is about 30%, not 5-16%, and it accumulated over three releases

Plain MC with **no tallies at all**, so this is purely the geometry and
flight path. Flight counts and leakage are identical to six figures in all
four versions, so nothing about the physics changed; this is cost per flight.

| problem | v1.2.0 | v1.3.0 | v1.4.0 | v1.4.1 | total |
|---|---|---|---|---|---|
| sphere | 1.60 s | 1.81 s (+13%) | 1.92 s (+6%) | 2.20 s (+15%) | **+38%** |
| cask | 3.26 s | 3.65 s (+12%) | 3.95 s (+8%) | 4.20 s (+6%) | **+29%** |
| slab | 2.26 s | 2.48 s (+10%) | 2.51 s (+1%) | 2.90 s (+16%) | **+28%** |

With each problem's own tallies, the same shape: sphere 2.31 -> 3.15 (+36%),
cask 7.21 -> 9.26 (+28%), slab 3.77 -> 5.01 (+33%).

**There is no single commit to revert.** Each release added roughly 10%, and
v1.4.1 added as much as v1.3 did. The v1.3 investigation found and fixed a
real Numba code-generation problem (a constant negative array index costing
10x in `tallies._combos`), and the remaining increment is not a bug of that
kind -- it is the price of generality, charged on every flight whether the
problem uses the new feature or not.

## 4. One correction to the v1.3 diagnosis

`runs/2026-10-29/v1.3/speed.log` reports that with no tallies the cask ran as
fast as v1.2 (4.96 s against 5.01 s), which pointed the investigation at the
tally path. Measured here the cask with no tallies goes 3.26 s -> 3.65 s at
v1.3, a 12% regression, and on to 4.20 s by v1.4.1.

And the two costs grew in step rather than one causing the other: on the
cask, the tally part went 3.94 s -> 5.06 s (1.28x) while the no-tally part
went 3.26 s -> 4.20 s (1.29x). Same factor. That is the signature of a
shared cost underneath both, not a tally problem.

## 5. What got more expensive in the shared path

Read from the diff rather than measured per change, so this is a reading of
the code and not yet an attribution of cost:

- **v1.3**: `_descend` gained a `path` argument and now writes the cell at
  **every level** of the descent, because a cell filter has to match a cell
  at any level the way OpenMC's does. Every flight and every birth pays it.
- **v1.4.1**: `_descend` also tracks `inst` for distributed materials, and
  the outer-universe fix means flights now stop at would-be lattice element
  walls as well.

Not the cause: the `pth`, `last`, `to` and `univ` buffers are allocated once
per chunk in `_advance`, not per flight, so there is no allocation in the hot
loop. I checked that first and it is clean.

This is consistent with the 14 October profile, where geometry was already
69-73% of a flight's cost. A 30% rise in the whole run is roughly a 40% rise
in the geometry part alone.

## 6. What to do, and what is still unmeasured

1. **Cache the cell across a scatter.** 85-88% of flights end in a scatter,
   which changes direction but never the cell, yet every flight re-walks the
   tree from the root. This attacks the regression and the OpenMC gap at the
   same time, because caching the cell and keeping neighbour lists is exactly
   what OpenMC does and why its flights are cheaper. `CEILING.md` item 3.
2. **Guard the per-level path recording** the way `last[]` is already guarded
   by `L.has_surface`: a problem with no multi-level cell filter should not
   pay for one. From the code this looks like most of the v1.3 increment, but
   that is a prediction, not a measurement.
3. **Stop quoting wall time against OpenMC analog.** Quote the figure of
   merit, or compare against OpenMC implicit. The headline 3x is mostly this.

Still unmeasured, and worth doing before anyone changes code: which of the
three v1.4 geometry changes costs what, by ablating them one at a time; and
how much of `_descend` is cell-finding rather than ray-casting, which is what
decides the size of the prize in item 1.

Nothing was changed. `core/` and `ball/` on `3d-gmc` are untouched, and this
write-up is on `gmc-speed` so it cannot collide with v1.5 work in flight.
