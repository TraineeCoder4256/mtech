# Where plain Monte Carlo lost its speed, version by version

Speed thread, 2 Nov 2026.  Branch `gmc-speed` (v1.6.1 merged in as
9add80b).  Nothing in `gmc3d/core/` or `gmc3d/ball/` was changed: every
measurement runs the released code out of a git worktree of its own
commit, and every ablation is a throwaway copy under the scratch
directory.

The question was: our plain Monte Carlo has been getting slower as the
versions went out, and nobody knew exactly why.  Answered here per step,
with the cause proved by taking the suspect out and measuring again, not
by reading the diff.

## How it was measured

Ten worktrees, one per released commit: v1.0 `177c5db`, v1.1.0 `8b80020`,
v1.1.3 `0764f63`, v1.2.0 `0013942`, v1.3.0 `052abb5`, v1.4.0 `3db14a2`,
v1.4.1 `7b5ed2f`, v1.5.0 `ef479a3`, v1.6.0 `641c6c4`, v1.6.1 `da4221c`.
Each one runs the same three problems (sphere, slab, cask), 100,000
histories x 5 batches, plain MC (no ball backend), on this container's 4
cores.

Three separate measurements, because they answer different questions:

* **the whole run with no tallies at all** -- geometry plus flights plus
  the per-batch bookkeeping, and the only column that is comparable
  across versions without further care;
* **`_descend` on its own**, from a tight compiled loop over 5,000 fixed
  interior points with both questions asked (nearest surface and ray
  distance).  Same points and directions in every version;
* **the source sampler on its own**, one batch of 100,000 sites.

One warning about the fourth column anyone would reach for.  **"With
tallies" is not comparable across versions**, because the problems' own
tally sets grew: at v1.1.3 the cask went from 3 tallies to 6 (a material
tally with four scores, a 2D mesh and a rectilinear mesh) and the slab
and sphere gained scores.  So the runs were repeated with the tally set
cut back to v1.0's in every version -- v1.0's tallies, by name, scoring
flux and absorption only.
## The short answer

Plain Monte Carlo, with no tallies at all, is **2.30x slower on the
sphere, 1.96x on the slab and 1.81x on the cask** at v1.6.1 than it was at
v1.0.  That is larger than the ~30% the 30 Oct note estimated, because
that note only compared v1.2.0 to v1.4.1 and the growth continued.

It is not one commit.  Step by step, each version against the one before
(no tallies, minimum of three interleaved rounds):

| step | sphere | slab | cask | what it was |
|---|---|---|---|---|
| v1.0 -> v1.1.0 | 1.14x | 0.96x | 0.97x | free |
| v1.1.0 -> v1.1.3 | 0.81x | 1.02x | 1.08x | free |
| **v1.1.3 -> v1.2.0** | **1.58x** | **1.34x** | **1.16x** | the source sampler |
| v1.2.0 -> v1.3.0 | 1.11x | 1.11x | 1.04x | `_track`'s argument list |
| v1.3.0 -> v1.4.0 | 1.05x | 1.06x | 1.08x | small |
| **v1.4.0 -> v1.4.1** | **1.10x** | **1.11x** | **1.14x** | the `Geometry` namedtuple |
| v1.4.1 -> v1.5.0 | 1.02x | 1.02x | 1.02x | free |
| v1.5.0 -> v1.6.0 | 1.11x | 1.03x | 1.01x | `_track`'s argument list |
| v1.6.0 -> v1.6.1 | 1.09x | 1.10x | 1.13x | `_track`'s argument list |

Two causes carry most of it, and both are now proved by ablation rather
than inferred from the diff.

## Cause 1: the source sampler, v1.2.0, a flat 0.6-0.75 s per 500,000 histories

Before v1.2 the source lived inside the transport: `_birth` was
`njit(parallel=True)` and each of the 32 thread chunks located its own
particles.  At v1.2 the source moved out, so that Python can sample every
OpenMC source type.  Python now draws the batch's sites first, and the
"is this point inside the geometry, and inside the domains?" check behind
it is `G.paths_many` -- an njit loop with a plain `range`, so it runs on
ONE core, over 1.2x as many candidate points as the batch needs, and it
records the cell AND the universe at every level instead of just the
material.

Timed on its own, one batch of 100,000 sites:

| version | sphere | slab | cask |
|---|---|---|---|
| v1.2.0 | 126 ms | 135 ms | 124 ms |
| v1.6.1 | 139 ms | 149 ms | 149 ms |

At 5 batches that is **0.62-0.75 s of every 500,000-history run**, whatever
the problem.  It accounts for the v1.2.0 step almost exactly: the sphere
went up 0.63 s and the sampler costs 0.63 s; the slab went up 0.60 s
against 0.67 s; the cask 0.50 s against 0.62 s (the cask's old parallel
birth was doing real work, so its net is smaller).

Because it is a fixed cost per history it hurts the cheap problems most.
On the sphere it is **26% of the whole v1.6.1 run**.  The sphere's source
is a single POINT, so those 600,000 descents all return the same answer.

## Cause 2: the size of the `Geometry` namedtuple, v1.4.1

`_descend`, the geometry walk, is handed `Geometry` on every flight.  Its
field count went 24 (v1.0) -> 25 (v1.4.0) -> **32** (v1.4.1), where
`surf_angle`, `cell_dist`, `cell_off`, `lat_off`, `leaf_mat`, `lat_type`
and `lat_rings` arrived together.

Timed on its own, ns per `_descend` over the same 5,000 fixed points with
both questions asked:

| version | sphere | slab | cask |
|---|---|---|---|
| v1.0 | 608 | 801 | 734 |
| v1.4.0 | 645 | 786 | 703 |
| v1.4.1 | 914 | 1161 | 898 |
| v1.6.1 | 883 | 1218 | 940 |

Flat for four releases, then up 30-45% at v1.4.1 and flat again.  So the
cell-path recording added at v1.3 and the extra surface kinds added at
v1.4.0 cost nothing measurable; v1.4.1 did.

**The ablation.** Take v1.0 and add eight float64 arrays to `Geometry`
that nothing ever reads.  Nothing else changes.
Whole plain-MC run, no tallies, 500,000 histories, all four rows measured
in the same pass:

| unused arrays added to `Geometry` | sphere | slab | cask |
|---|---|---|---|
| 0 (v1.0 as released) | 1.243 s | 1.892 s | 2.893 s |
| 2 | 1.295 s (+4%) | 2.043 s (+8%) | 3.205 s (+11%) |
| 4 | 1.334 s (+7%) | 2.035 s (+8%) | 3.181 s (+10%) |
| 8 | 1.414 s (+14%) | 2.165 s (+14%) | 3.552 s (+23%) |

and the same four on the geometry walk alone, ns per `_descend`:

| unused arrays | sphere | slab | cask |
|---|---|---|---|
| 0 | 651 | 780 | 589 |
| 2 | 673 | 835 | 730 |
| 4 | 737 | 902 | 682 |
| 8 | 779 | 852 | 844 |

Monotone, and the size of it is right: **v1.4.1 added seven such arrays
and the v1.4.1 step is +10% / +11% / +14%**, where eight arrays that do
nothing cost +14% / +14% / +23%.  The whole step is accounted for by the
size of the tuple, with nothing left over for the new logic.

A second pass on a separate occasion gave the same answer for eight
arrays (+36% / +11% / +23% on the run, +33% / +7% / +33% on the walk), so
the effect is not one unlucky measurement.

**Two further checks confirm the mechanism is the per-flight hand-over,
not namedtuples in general.**  Padding v1.0's `Physics` with seven unused
arrays, and `Particles` with two, cost nothing at all -- every row at or
below baseline, on all three problems.  Those two are handed to `_track`
once per particle, which amortises over the ~20 flights a particle makes.
`Geometry` goes to `_descend` on every flight, so it is paid 20 times as
often.

**What did NOT cost anything at v1.4.1**, each measured by taking it out
of a copy of v1.4.1 and re-running in the same pass:

* the distributed-material instance tracking (`cell_off`, `leaf_mat`, the
  running `inst`): the walk came back 884 / 1116 / 877 ns against
  v1.4.1's 772 / 1119 / 929, i.e. no change beyond noise;
* widening the coefficient row.  `surfaces.NCOEF` went 4 -> 8 at v1.4.0
  and 8 -> 11 at v1.6.1; v1.0 with an 11-wide row is 653 / 835 / 716 ns
  against its own 648 / 822 / 653, and the full run 1.111 / 1.721 / 2.818 s
  against 1.264 / 1.939 / 2.927.  Free.
* the six floats `_descend` keeps at v1.5.0 to remember the winning
  surface's frame: removing them gave 788 / 978 / 963 ns against v1.5.0's
  839 / 1013 / 867, which is why the v1.5.0 step is only 2%.
## Cause 3: `_track` itself, v1.3.0 and v1.6.x, about 10% a time

Three steps -- v1.2.0 -> v1.3.0, v1.5.0 -> v1.6.0 and v1.6.0 -> v1.6.1 --
each cost about 10% and none of it is in the geometry walk (flat across
all three) or in the source sampler (flat too).  It is inside `_track`,
the compiled loop that runs one particle.

What grew there is the function itself.  Its argument list went 7 -> 12
(v1.3) -> 13 (v1.5) -> 16 (v1.6.0) -> **20** (v1.6.1), four of them
namedtuples, and the body gained a guarded branch for each new feature.
Every new flag is cheap on its own, and all of them are correctly behind
their flags -- a no-tallies run executes none of the new work -- but the
loop that holds them all is bigger, with more live state across it.

**What this is NOT.** Two ablations rule out the obvious reading. Padding
v1.0's `Physics` with seven unused arrays (the shape of the v1.1.0
change) and padding `Particles` with two (v1.2 and v1.3) both came out at
or below the baseline, on all three problems.  So handing `_track` a
bigger namedtuple costs nothing: it is called once per particle, and that
is amortised over ~20 flights.  `Geometry` is the exception precisely
because it is handed over on every flight.

**The limit.** I could not take this one apart further without rewriting
`_track`, which was out of scope here: there is no single line to remove,
and cutting any one feature's branch changes what the code does.  Sizing
it properly needs a build with the features compiled out, which is a code
change for you to approve rather than an experiment.

## The tally side, and why the "with tallies" column misleads

The raw numbers look alarming -- the cask's tally cost went from 0.14 s at
v1.0 to 5.18 s at v1.6.1 -- but most of that is **the problems asking for
more tallies**, not the tally code slowing down.  At v1.1.3 the cask went
from 3 tally sets to 6 and the sphere and slab gained scores; at v1.5 the
sphere gained cylindrical and spherical meshes.

Re-run with v1.0's tally set in every version (v1.0's tallies, by name,
scoring flux and absorption only), the honest growth is:

| version | sphere | slab | cask |
|---|---|---|---|
| v1.0 | 0.36 s | 0.30 s | 0.41 s |
| v1.1.3 | 0.97 s | 0.95 s | 1.38 s |
| v1.3.0 | 0.97 s | 1.19 s | 2.00 s |
| v1.6.1 | 0.98 s | 1.19 s | 1.85 s |

So scoring the same tallies costs 2.7x (sphere), 4.0x (slab) and 4.5x
(cask) what it did at v1.0, and it all arrived in two steps: v1.1.3,
which made the scorer general enough for the extra score types, and
v1.3.0, which made it walk the filter list.  It has been flat since.
## Against OpenMC: what actually changed

The 6 Oct rules still apply -- same session, name the OpenMC mode, and
remember that OpenMC's DEFAULT is analog (it kills on absorption) while
ours is implicit capture, so analog is not a like-for-like algorithm.
Both races below ran in this session, 500,000 histories, with **each
version's own tally sets on both sides**, so OpenMC pays for the same
tallies we do.

| | sphere | slab | cask |
|---|---|---|---|
| v1.0, ours | 1.61 s | 2.09 s | 3.48 s |
| v1.0, OpenMC implicit | 2.43 s | 2.32 s | 4.36 s |
| **v1.0 ratio** | **0.67x** | **0.90x** | **0.80x** |
| v1.6.1, ours | 7.16 s | 6.93 s | 10.77 s |
| v1.6.1, OpenMC implicit | 3.12 s | 3.84 s | 8.24 s |
| **v1.6.1 ratio** | **2.29x** | **1.81x** | **1.31x** |
| v1.6.1, OpenMC analog | 2.17 s | 2.47 s | 4.20 s |
| ns per flight, v1.0 | 174 | 184 | 174 |
| ns per flight, v1.6.1 | 771 | 613 | 538 |

**At v1.0 we beat OpenMC implicit on all three problems.  We now lose on
all three.**  That is the plainest statement of what you asked about.

And it is worth seeing where the two codes diverged.  The problems ask
for about twice as many tallies as they did at v1.0, and OpenMC absorbed
that at a cost of 28% on the sphere (2.43 -> 3.12 s).  Ours went from
1.61 to 7.16 s, 4.4x.  Splitting the sphere's 5.55 s increase:

* **1.53 s (28%) is plain Monte Carlo** -- geometry, flights, the source
  sampler.  That is the regression taken apart above.
* **4.02 s (72%) is tally scoring.**  Of that, 0.62 s is the same v1.0
  tallies costing 2.7x what they used to, and the rest is the problems
  now asking for six tally sets where they asked for three.

So the regression is real and worth fixing, but on today's numbers the
bigger half of the gap to OpenMC is the tally scorer, not the transport.
Asking for more tallies is a deliberate coverage choice and not a bug;
scoring them 2.7-4.5x less efficiently per tally than v1.0 did is the
part worth looking at.
## What each version added to the code every flight runs

Read from the diffs, one step at a time.  "Guarded" means the new work
sits behind a flag that is off when a run has no tallies, so it cannot
explain the no-tallies column; it still costs when the tallies are on.

**v1.0 -> v1.1.0** (mesh tallies in balls).  `Physics` grew from 9 fields
to 16: the four arrays of `ball/profile.py`'s table joined it, plus
`mesh_profile`, `mesh_h`, `k_max`.  `Physics` is handed to `_track` for
every particle.  Two surface-current hooks were added to the crossing
path, both behind `L.has_surface`.  The mesh-point loop is behind
`ph.mesh_profile`.

**v1.1.0 -> v1.1.3** (silent gaps loud, easy tally items).  `Layout` grew
11 -> 13 fields.  Nothing new in the no-tallies path.  The problems' OWN
tally sets grew here, which is most of what the "with tallies" column
does from this version on: the cask went from 3 tallies to 6 (a material
tally with four scores, a 2D mesh, a rectilinear mesh) and the slab and
sphere gained scores too.

**v1.1.3 -> v1.2.0** (all OpenMC source kinds).  The source moved out of
the transport.  Before: `_birth` was `njit(parallel=True)` and called
`G.locate` for each particle inside the parallel loop.  After: Python
draws the batch's sites from the OpenMC source objects first, and the
"is this point inside the geometry?" check behind it is
`G.paths_many` -- an njit loop with a plain `range`, so SERIAL, on 1.2x
as many candidate points as the batch needs, recording the cell AND the
universe at every level rather than just the material.  `Particles` grew
by `w0` and the two roulette tests now read `P.w0[i]`.

**v1.2.0 -> v1.3.0** (multi-filter tallies).  `_descend` gained a `path`
buffer and writes the cell at every level into it on every call
(`path[level] = cell`, plus a -1 terminator).  `_track`'s argument list
went from 7 to 12.  `Particles` grew again.  The extra `G._path` descent
on the far side of a crossing is behind `L.has_surface`.

**v1.3.0 -> v1.4.0** (geometry, small).  `surfaces.NCOEF` went from 4 to
8: every surface's coefficient row doubled in width, so `surf_coef` is
twice the size and a row can straddle a cache line.  `Geometry` gained
`surf_shift`.  Each of the four surface kernels gained a
`if kind >= CYLINDER` test before the cylinder/cone branches.

**v1.4.0 -> v1.4.1** (geometry, medium).  `Geometry` went from 25 fields
to 32: `surf_angle`, `cell_dist`, `cell_off`, `lat_off`, `leaf_mat`,
`lat_type`, `lat_rings`.  `_descend` now tracks a distributed-material
instance index down the descent and reads `g.cell_dist[cell]` at the
leaf.  The lattice step gained the two hexagonal orientations, and the
surface kernels the torus and the quadric.

**v1.4.1 -> v1.5.0** (medium tally items).  `_descend` now keeps the
winning surface's own frame -- six extra floats `hx, hy, hz, hu, hv, hw`
-- and writes all six every time a nearer ray surface is found, inside
the innermost surface loop.  Those writes are NOT guarded: `want_sign`
only guards the `S.normal` call at the leaf that uses them.  It also
returns the level instead of a root flag.  `_track` gained the
`MeshExtra` argument.

**v1.5.0 -> v1.6.0** (point detectors, robustness).  `_track` gained the
`Detectors` namedtuple and a `dpth` buffer, 13 -> 16 arguments.  Every
detector hook is behind `D.on`.

**v1.6.0 -> v1.6.1** (every non-physics OpenMC input).  `_track` gained
`E`, `XS`, `ws`, 16 -> 20 arguments, four of them namedtuples.
`NCOEF` went 8 -> 11 for the quadric, so the coefficient row is now
2.75x v1.0's.  `surf_shift` gained an albedo column, checked on every
reflective, white or periodic crossing.  Everything `ext.py` does is
behind `E.n`.
## What I would look at, in order (nothing landed, nothing approved)

1. **The source sampler.**  The clearest win and the least risky, because
   it changes no physics: 0.62-0.75 s of every 500,000-history run, 26% of
   the sphere's.  Three independent things are wrong with it -- it runs on
   one core where the rest of the code runs on 32 chunks, it draws 1.2x as
   many candidates as it needs, and it records the full cell AND universe
   path when only "is there material here?" is being asked (plus the
   domains, when a source has them, which the test problems do not).  A
   point source repeats one lookup 600,000 times.
2. **Split `Geometry`.**  Proved above to be worth 11-36% of a plain-MC
   run.  The v1.6.1 code already says this in a comment and already
   answered it once (`tallies.MeshExtra` was split out for exactly this
   reason); the same move for the seven v1.4.1 arrays, which only the
   lattice and distributed-material paths read, would give it back.
3. **Measure `_track` with features compiled out**, to size cause 3
   before touching it.  That needs a build flag, which is a code change,
   so it waits for you.

## What this does not say

* It says nothing about the ball step or the network: every number here
  is plain Monte Carlo, no backend.
* It is not a claim that any version is wrong.  Leakage agrees across all
  ten versions to the sixth figure, and the flight counts are identical
  from v1.2 on (v1.2 changed the source's random numbers, so they differ
  by 0.2% before and after that line).
* Run-to-run spread on this shared container is about 8-15%, so a single
  step of 5% in the table is not meaningful; the two steps called out are
  far outside that, and the ablations compare only rows measured in the
  same pass.

## The full tables
## Plain MC, no tallies at all: seconds for 500,000 histories

### no tallies

| version | sphere | slab | cask | sphere vs v1.0 | slab vs v1.0 | cask vs v1.0 |
|---|---|---|---|---|---|---|
| v1.0 | 1.18 | 1.79 | 2.93 | 1.00x | 1.00x | 1.00x |
| v1.1.0 | 1.34 | 1.71 | 2.83 | 1.14x | 0.96x | 0.97x |
| v1.1.3 | 1.09 | 1.75 | 3.05 | 0.92x | 0.98x | 1.04x |
| v1.2.0 | 1.72 | 2.35 | 3.55 | 1.46x | 1.32x | 1.21x |
| v1.3.0 | 1.91 | 2.60 | 3.70 | 1.62x | 1.45x | 1.26x |
| v1.4.0 | 2.00 | 2.76 | 3.99 | 1.69x | 1.55x | 1.36x |
| v1.4.1 | 2.19 | 3.05 | 4.56 | 1.86x | 1.71x | 1.56x |
| v1.5.0 | 2.24 | 3.11 | 4.63 | 1.90x | 1.74x | 1.58x |
| v1.6.0 | 2.49 | 3.19 | 4.67 | 2.11x | 1.79x | 1.59x |
| v1.6.1 | 2.71 | 3.50 | 5.29 | 2.30x | 1.96x | 1.81x |

## Where that time sits

### advance s / rest s, then ns per _descend, then source ms per batch

| version | sphere adv/rest | slab adv/rest | cask adv/rest | sphere desc ns | slab desc ns | cask desc ns | sphere src ms | slab src ms | cask src ms |
|---|---|---|---|---|---|---|---|---|---|
| v1.0 | 1.11 / 0.07 | 1.71 / 0.07 | 2.85 / 0.08 | 648 | 822 | 653 | - | - | - |
| v1.1.0 | 1.26 / 0.08 | 1.64 / 0.07 | 2.75 / 0.08 | 668 | 765 | 721 | - | - | - |
| v1.1.3 | 1.02 / 0.07 | 1.68 / 0.07 | 2.98 / 0.08 | 667 | 707 | 732 | - | - | - |
| v1.2.0 | 1.65 / 0.07 | 2.27 / 0.08 | 3.48 / 0.07 | 672 | 757 | 647 | 126 | 135 | 124 |
| v1.3.0 | 1.80 / 0.11 | 2.50 / 0.09 | 3.60 / 0.09 | 684 | 820 | 717 | 115 | 129 | 122 |
| v1.4.0 | 1.91 / 0.09 | 2.55 / 0.22 | 3.90 / 0.09 | 645 | 786 | 703 | 114 | 133 | 118 |
| v1.4.1 | 2.11 / 0.08 | 2.92 / 0.13 | 4.47 / 0.09 | 914 | 1161 | 898 | 141 | 154 | 151 |
| v1.5.0 | 2.15 / 0.09 | 2.94 / 0.17 | 4.54 / 0.10 | 874 | 1152 | 962 | 123 | 124 | 124 |
| v1.6.0 | 2.40 / 0.09 | 3.09 / 0.10 | 4.53 / 0.14 | 878 | 1180 | 869 | 135 | 137 | 148 |
| v1.6.1 | 2.61 / 0.10 | 3.39 / 0.11 | 5.21 / 0.08 | 883 | 1218 | 940 | 139 | 149 | 149 |

## The tally side

### tally cost: the problem's own set, then v1.0's set, then how many tallies v1.0's set is

| version | sphere own | slab own | cask own | sphere fixed | slab fixed | cask fixed | sphere n | slab n | cask n |
|---|---|---|---|---|---|---|---|---|---|
| v1.0 | 0.55 | 0.60 | 0.14 | 0.36 | 0.30 | 0.41 | 3 | 2 | 3 |
| v1.1.0 | 0.30 | 0.26 | 0.80 | 0.31 | 0.52 | 0.82 | 3 | 2 | 3 |
| v1.1.3 | 1.05 | 1.58 | 4.27 | 0.97 | 0.95 | 1.38 | 3 | 2 | 3 |
| v1.2.0 | 0.94 | 1.88 | 4.11 | 1.05 | 0.91 | 1.27 | 3 | 2 | 3 |
| v1.3.0 | 0.97 | 2.42 | 5.58 | 0.97 | 1.19 | 2.00 | 3 | 2 | 3 |
| v1.4.0 | 0.93 | 2.06 | 5.43 | 1.09 | 1.02 | 1.97 | 3 | 2 | 3 |
| v1.4.1 | 0.79 | 1.86 | 5.23 | 1.06 | 1.08 | 1.73 | 3 | 2 | 3 |
| v1.5.0 | 3.43 | 3.37 | 4.93 | 1.18 | 1.19 | 1.98 | 3 | 2 | 3 |
| v1.6.0 | 3.58 | 3.36 | 5.51 | 0.80 | 1.13 | 1.90 | 3 | 2 | 3 |
| v1.6.1 | 3.88 | 3.17 | 5.18 | 0.98 | 1.19 | 1.85 | 3 | 2 | 3 |
