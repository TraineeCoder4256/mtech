# What the transport loop is made of

14 October 2026, branch `gmc-speed`. Measured with
`checks/loop_breakdown.py` on this container (4 Intel Xeon cores at
2.1 GHz, no GPU), 1,000,000 histories per problem, lookup-table backend at
R* = 2, `centre` mesh rule. Raw output in `loop_breakdown.log` and the three
JSON files beside it.

`runs/2026-10-06/SPEED.md` established that once the network is compiled in
float16 it costs 108 ns a draw and is within 8-30% of a free backend, so the
thing that limits gmc3d is no longer the model but the Monte Carlo transport
loop on the CPU. That loop reported its own time as a single number,
`transport_s`, and had never been broken down. This is the breakdown.

The script re-runs the wave loop with a stopwatch on each phase, importing
the same numba kernels from `core/transport.py` — nothing in `core/` or
`ball/` is touched. The answer and the flight count are checked against
`transport.run` before any timing is printed, and matched exactly on all
three problems (slab 0.860324433, sphere 0.031571518, cask 0.002371112).

## Where the wall time goes

| phase | slab | sphere | cask |
|---|---|---|---|
| allocate particles | 4.3% | **14.3%** | 3.0% |
| `_birth` | 9.6% | **29.1%** | 6.0% |
| `_advance` | **81.7%** | 46.0% | **86.7%** |
| gather / scatter the WAITING list | 2.2% | 1.5% | 1.5% |
| `backend.sample` | 2.2% | 1.8% | 2.4% |
| `accum.add_batch` | 0.0% | 7.2% | 0.3% |
| wall | 3.60 s | 1.04 s | 4.98 s |

Two different bottlenecks, not one. The slab and the cask are
flight-dominated and live in `_advance`. The sphere is not: at R* = 2 it
runs **36,107 flights for 1,000,000 histories and zero scatters**, because
one ball takes each particle from the source to the boundary. The physics is
already gone there, and 43% of what remains is per-history bookkeeping.

A note on the race's numbers: `transport_s + backend_s` covers everything
except the per-batch particle allocation, so a race's `seconds` is 3-4% low
on the slab and cask and 22% low on the sphere. Both figures here come from
the same execution; comparing one run's reported seconds against another
run's wall clock mixes this gap with run-to-run noise, which is larger.

## Inside `_advance`: geometry is the cost

Timed on 20,000 real scatter sites harvested from the run itself, each
kernel driven from its own njit loop, single-threaded, nanoseconds per call:

| kernel | slab | sphere | cask |
|---|---|---|---|
| `_descend`, want_near = False | 978 | 766 | 846 |
| `_descend`, want_near = True | 1085 | 780 | 884 |
| the ball step's extra cost | +107 (1.11x) | +14 (1.02x) | +39 (1.05x) |
| `score_segment` | 325 | 316 | 335 |
| `rand` + `isotropic` | 33 | 34 | 34 |
| **geometry's share of the three** | **73%** | **69%** | **70%** |

Two things follow.

**The ball step's geometry requirement is nearly free.** A ball needs the
distance to the nearest surface in *any* direction, not just along the ray,
and that was a fair worry. It costs 2-11%. It is not a problem.

**Every flight pays a full root-down `_descend`, and most flights do not
need one.** Of the slab's 7,426,815 flights, 6,275,671 (85%) ended in a
scatter; on the cask, 8,949,968 of 10,161,858 (88%). A scatter changes the
direction, never the cell, yet the next flight walks the CSG tree from the
root again to rediscover a cell the particle never left. This is what
production codes avoid with a cached cell and neighbour lists.

Sizing that properly needs one more measurement, because `_descend` finds
the cell and the distance to the next surface in a single walk and only the
first part is redundant. So: **the single largest item in the ceiling, worth
up to 70% of flight time on the flight-dominated problems, with the share
that is actually recoverable not yet measured.**

## The sphere's ceiling is bookkeeping, and most of it is removable

`_birth` calls `G.locate` once per particle to find the source cell.
Measured at each problem's own source point: 1,047 ns on the sphere,
1,990 on curved, 1,892 on nested, 1,153 on the slab, single-threaded. Birth
costs 303 ns per particle on 4 threads, so the lookup is about 85% of it.

The sphere, curved and nested all use a **point** source. The position is
identical for every history, so `G.locate` is called a million times with
the same three arguments and returns the same answer. On the sphere that is
about 25% of the entire wall time of the run.

Per-batch allocation is the other piece: `_particles(n)` builds 17 fresh
arrays of n doubles every batch, 13.6 MB at 100,000 particles, and it is
pure zeroing of buffers that could be reused across batches. 14.3% of the
sphere's wall, 3-4% elsewhere.

Together that is roughly 35-39% of the sphere's wall time in work that
changes no physics and no answer. Removing it would move the sphere's
free-backend figure of merit from the measured 5.26x toward 8x.

## What this says to do, in order

1. **Reuse the particle arrays across batches.** 3-14% everywhere, no
   physics, the most contained change in this file.
2. **Cache the source lookup for a point source.** About 25% of the sphere
   and applies to curved and nested too; a point source cannot land in a
   different cell.
3. **Cache the cell across a scatter.** The big one on the slab and cask,
   where 85-88% of flights re-derive a cell that cannot have changed. Needs
   `_descend` split into its cell-finding and ray-casting halves to be
   sized, and that measurement should come before the change.
4. Overlap the backend call with transport (double-buffering), which
   `SPEED.md` section 6 sized at 11-33%. Note this breakdown lowers the
   prize: `backend.sample` is only 1.8-2.4% of the wall with the table, so
   overlapping matters only with a slower backend.

None of these is landed. Items 1 and 2 need no physics judgement; item 3
does.
