# gmc3d v3 plan: multigroup energy

Nov 18, 2026 · @Phani

## Summary

v3 is Stage 3 of the original SRS: **multigroup energy, with Design B ball steps**. It is the MTech target you set on 27 Oct. Stages 1 and 2 are done: any CSG geometry, one energy, isotropic and anisotropic scattering, with exact-walk balls and an accepted trained model (v1.0 to v2.2.2).

The plan has six pushes. v3.0 adds G groups to plain Monte Carlo and matches OpenMC's multigroup mode. v3.1 adds Design B balls walked exactly, and measures how many collisions a ball replaces per group. That number is the **go/no-go** for training anything. v3.2 to v3.4 add the interface, the trained models and the gate. v3.5 delivers the test problems and the race against OpenMC.

The SRS's done line is met at the end of v3.5: *a G-group fixed-source problem runs with Design B ball steps, matches OpenMC multigroup on the same library, and reports how many collisions each ball step replaces per group.*

## Where we are (v2.2.2, 17 Nov 2026)

Everything below is on branch `3d-gmc` in `gmc3d/`. It is checked against OpenMC 0.16 in multigroup mode with one group.

| Area | What exists | Versions |
| --- | --- | --- |
| Geometry | Any OpenMC CSG model: all surface types, any-axis cylinders and cones, tori, quadrics, rectangular and hex lattices, universes, white, reflective and periodic boundaries | v1.0, v1.4 |
| Sources | Every OpenMC source kind and strength, constraints, MCPL files | v1.2 |
| Tallies | Every OpenMC filter and score that makes sense at one group: meshes of all kinds, surface currents, point detectors, triggers | v1.1, v1.3, v1.5, v1.6 |
| Physics | One group, no fission. Isotropic scattering (v1) and every OpenMC anisotropic law (v2.0) | v1, v2.0 |
| Ball step | Distance field picks the largest one-material ball; the exact walk (oracle) or a trained model draws the exit; absorption is applied after as exp(-kappa s) | v1.0, v2.1 |
| Models | Interface v1: R to (mu, s), nflow seed 0 accepted. Interface v2: R and 12 Legendre moments to (cos\_n, mu, phi, s), cvae v2 seed 0 accepted (m2.1). Four families compared: nflow, cfm, cvae, cgan | m1.x, m2.1 |
| Acceptance | gate2: 343 checks against the exact walk (2% absorbed-share rule); anisocheck 20/20 against OpenMC; score 9.9/10 | v2.1.6 |
| GPU | Training and draws on CUDA (fp16, compile, CUDA graphs), several batches in flight, CPU and GPU overlap, `run.py gpubench`; not yet run on a GPU | v2.2, m2.3, m2.4 |

**Speed so far.** On 4 CPU cores the exact-walk ball beats OpenMC analog on thick problems (5 Oct race: 1.7 to 2.9x FOM). The trained network costs 1.75 us per anisotropic ball, against 1.35 us for the exact walk, so on a CPU the model does not yet pay. A GPU projection from 6 Oct puts it at 1.4 to 2.8x OpenMC. The real GPU numbers are waiting on your `gpubench` run.

**What the one-group results mean for v3.** At one group a lookup table or the exact walk does the job, and the model has to win on cost alone. Multigroup is where a model is first needed. The inputs grow (q, the in-group share, joins R and the angular moments), so a table's size explodes while a network does not.

## What Stage 3 must deliver (from the SRS, 5 Oct)

Ten requirements, all kept. The last column says which push in this plan meets each one.

| ID | Requirement | Accepted when | Push |
| --- | --- | --- | --- |
| S3-XS-1 | Read OpenMC multigroup libraries: total, absorption, scattering matrix with its Legendre order, per material; a script that makes a library | Values read back equal OpenMC's own | v3.0 |
| S3-MC-1 | Plain multigroup transport: group tracking, outgoing group sampled from the scattering matrix, upscatter | Matches OpenMC multigroup within statistics | v3.0 |
| S3-MC-2 | Sources with a group spectrum | Source spectrum test passes | v3.0 |
| S3-TAL-1 | Per-group flux and reaction rates; energy filters on the library's group edges | Matches OpenMC per group | v3.0 |
| S3-DATA-1 | Design B walk generator: R in the group's mean free paths and q (share of scatters that stay in the group); records escape or out-of-group scatter, its state, path length | Reduces to Stage 1 data when q = 1 | v3.1, v3.2 |
| S3-DRV-1 | Driver tracks the group; an out-of-group event samples the new group and direction by Monte Carlo and starts a new ball | Exact-walk balls match plain MC | v3.1 |
| S3-VAL-2 | Collisions replaced per ball step, per group and material | Shows where Design B pays and where not | v3.1 (first), v3.5 (final) |
| S3-MOD-1 | Interface v3: R, q (plus the Stage 2 angular moments) in; event, position or exit state, path length out | Interface tests for the exact walk and the network | v3.2 to v3.4 |
| S3-VAL-1 | Test problems: a 2-group and an 8-group shield or reflector, a graphite block | Scored against OpenMC multigroup | v3.0 (plain MC), v3.5 (balls) |
| S3-OPT-1 | Stretch: Design A (per-material model with exit group and per-group path lengths), only if Design B saves too little | Decision recorded either way | Decision after v3.1 |

**Two things the SRS did not know on 5 Oct.** Stage 2 is now built, so an in-group walk must carry the anisotropic law as well. And you dropped the lookup table, so "interface tests for oracle, table and network" becomes oracle and network.

## How a ball works with groups (Design B)

A ball step stays in one group: it ends when the neutron **escapes** the ball or **scatters out of its group**. The out-of-group scatter is done by plain Monte Carlo, and a new ball can start there. Inside a step the physics is Stage 2's one-group problem with one extra way to stop, so a single universal model still serves every material and every group.

**One step, in group g of material m:**

1. The ball radius R is measured in group g's mean free paths (1 / sigma\_t,g), as today.
2. q = sigma\_s,g->g / sigma\_s,g is the share of scatters that stay in group g. The in-group angular law (its Legendre moments g1..g12) comes from the g->g block of the scattering matrix.
3. The walk scatters in group with probability q at each collision. With probability 1 - q it scatters out, and the step ends there.
4. On **escape**: the outputs are v2's (cos\_n, mu, phi, s), and the particle leaves the ball in group g.
5. On **out-scatter**: the outputs are where the collision happened and which way the neutron was going. Then plain MC picks g' from the scattering matrix row and the new direction from the g->g' angular law.
6. Absorption stays exact and outside the model: weight times exp(-sigma\_a,g / sigma\_t,g x s). Flux in group g is the track length s, so per-group tallies need nothing new.

**Interface v3 (proposal, to agree with Development).**

- In: R, q and g1..g12 of the in-group law (14 numbers; v2 had 13).
- Out: an event (escape or out-scatter), then that event's numbers. Escape: (cos\_n, mu, phi, s) as in v2. Out-scatter: the collision's radius r / R, its angle to the first direction, the incoming direction at the collision as two angles in that frame, and s. That is 5 numbers. One may drop by symmetry; the v3.1 work settles it.
- The event is a yes/no with a known probability per (R, q, law), so the model is a classifier head plus the continuous part conditioned on the event. This mixed output is the main new modelling problem (Development).
- Kept from v2: the exact walk to the second collision. Single-scatter walks are planar and give a point mass that no smooth model can draw. At q < 1 the first collision is also where many steps end.
- q = 1 must give exactly the v2 walk. That is the regression check for the data generator.

**Where Design B pays.** A step replaces about 1 / (1 - q) collisions at most, and fewer in a small ball (a walk to the surface of a ball of R mfp takes about R squared collisions).

| q (in-group share) | Collisions per step, at most | Typical case |
| --- | --- | --- |
| 0.5 | 2 | Fast groups in water: little gain |
| 0.9 | 10 | Broad fast groups, iron |
| 0.99 | 100 | Thermal group, graphite, heavy water: large gain |

So the gain sits in thermal groups and weak moderators, which is where the reflector and graphite problems of S3-VAL-1 live. v3.1 measures it with the exact walk before any model is trained.

## The pushes, v3.0 to v3.5

Each push is a complete, checked result, so stopping after any of them still leaves something to show. Owner: I = Innovation (transport, data, checks), D = Development (models).

| Push | Builds | Checked by | Owner |
| --- | --- | --- | --- |
| v3.0 | G-group plain Monte Carlo. Read G-group libraries (macroscopic and nuclide sums); the group carried by every particle; outgoing group and angle from the scattering matrix (per g->g' law, upscatter included); source group spectra; EnergyFilter on group edges; per-group scores. New problems: a 2-group shield, a 7- or 8-group reflector, a graphite block. Balls switched off when G > 1 | New `run.py mgcheck`: every problem against OpenMC multigroup, per group; a 1-group library gives v2.2 bit for bit | I |
| v3.1 | Design B with the exact walk. The in-group walk (ends on escape or out-scatter, with the anisotropic law); the driver hands out-scatters to plain MC and starts new balls; S3-VAL-2 counters: collisions replaced per step, per group and material | Exact-walk balls match plain MC on every v3.0 problem; q = 1 gives the v2 walk exactly; the **go/no-go report** | I |
| v3.2 | Interface v3 and its data: `ball/interface.py` v3, the walk generator over R, q and the law, held-out laws and q values, data hash; `checks/gate_v3.py` | Generator reduces to v2 data at q = 1; gate3 passes the exact walk itself | I |
| v3.3 | Model families at interface v3: event head plus continuous part, for nflow, cfm, cvae (cgan if wanted); trained on GPU or CPU (m2.3) | Development's comparison; Innovation's gate3 per family and seed | D |
| v3.4 | Accept a model: `network` uses it for multigroup balls; GPU draws, batches in flight and overlap as in v2.2 | gate3 at the agreed rule; `mgcheck` with the model | I + D |
| v3.5 | Results: the S3-VAL-1 problems and the race against OpenMC multigroup (analog and implicit), 3 seeds, same session; final S3-VAL-2 table | The SRS done line | I |

**The gate after v3.1 is yours.** If the exact-walk balls replace too few collisions on the test problems, training a model cannot make them fast. Then the choice is Design A (S3-OPT-1) or writing up Design B's limits as a result. v3.2 onwards waits for that decision.

**Rough size.** v3.0 and v3.1 are like v1.x pushes (a few days each). v3.2 is like v2.1.1. v3.3 is the largest, because of the mixed discrete and continuous output. Expect several model rounds, as m2.0 needed m2.1. v3.5 is mostly compute: the race took about 2.5 h per seed on 4 cores in November.

## Model side (Development, to agree)

The split from 5 Nov stays: Innovation owns the interface, the walk, the data generator and the gate; Development owns `ball/models/`. Disagreements go to you.

- **Data.** As v2: about 40M walks from a training family of laws, now also spread over q. Test-problem laws and some q values are held out to test generalisation. q near 1 needs the most data, because that is where the long walks and the gain are.
- **Families.** The same four, with the same fairness rules (same data, weight count within ±20%, step budget, 2 seeds). Each needs an event head (one Bernoulli output) with the continuous part conditioned on the event. A family may also split this into two models.
- **Training hardware.** GPU training is ready (m2.3). On 4 CPU cores a v2 model took 15 to 33 minutes; v3 data is larger, so the GPU is worth using here.
- **Gate3** (Innovation). gate2's checks per (law, R) cell, now also per q, plus the event probability: |z| and an absolute tolerance, using the two-condition style you chose for the weight rule. The new failure modes to test are a wrong escape probability and a wrong out-scatter position, which moves where the next ball starts.

## Checks and benchmarks

Every comparison is against OpenMC 0.16 in multigroup mode on the **same library**, so any difference is ours.

| Check | What it compares | First in |
| --- | --- | --- |
| `mgcheck` | Plain MC, then exact-walk balls, then the model, against OpenMC per group: flux, absorption, leakage, currents | v3.0 |
| 1-group regression | A 1-group library through the v3 code gives v2.2 results bit for bit | v3.0 |
| q = 1 regression | The v3 walk with q = 1 gives the v2 walk bit for bit | v3.1 |
| Collisions replaced | Per ball step, per group and material (S3-VAL-2) | v3.1 |
| gate3 | The model against the exact walk, per (law, R, q) | v3.2 |
| Race | FOM against OpenMC analog and implicit, same session, median of 3 seeds, about 15% on any ratio, CPU and GPU rows | v3.5 |

**Where the multigroup data comes from.** This container has no continuous-energy nuclear data, so `openmc.mgxs` cannot make libraries here. The plan uses two sources:

1. **Published multigroup sets.** The C5G7 benchmark's 7-group cross sections (UO2, MOX, moderator, guide tube) are public and widely accepted. The moderator and guide-tube materials make a fixed-source reflector problem. This answers the reviewers' request for widely accepted benchmarks, and C5G7 itself becomes the Stage 4 (k-eff) benchmark later.
2. **Libraries we write ourselves** with `openmc.XSdata`: a 2-group shield and an 8-group graphite block with q chosen to span 0.5 to 0.99. These test the method where it should and should not pay.

If you can run OpenMC with continuous-energy data on your machine, `openmc.mgxs` libraries for a real shield (concrete, water, iron) can be added in v3.5.

## Risks, research questions, out of scope

**Risks**

| Risk | Effect | Fallback |
| --- | --- | --- |
| Design B replaces few collisions on realistic problems (q well below 0.9 in fast groups) | Little speed-up, whatever the model | Found out at v3.1, before any training; then Design A or report the limit |
| The mixed output (event plus continuous) is hard to learn, as the single-scatter point mass was in m2.0 | Gate3 fails for several rounds | Exact first two collisions as in m2.1; separate models per event |
| A wrong out-scatter position moves every later ball | Errors compound along the history | gate3 checks the position; `mgcheck` checks whole problems |
| Libraries: no continuous-energy data here | Fewer realistic materials | C5G7 data plus written libraries; real `openmc.mgxs` libraries on your machine |
| v1 and v2 features that touch energy (meshes in balls, point detectors, every filter) need a group | Some v2 combinations refused at first | Recorded in GAPS.md, as every version does |

**Research questions (yours, as agreed on 5 Oct)**

- R-v3a: is "collisions replaced per step" enough to justify Design B, and at what threshold do we stop?
- R-v3b: an alternative to stopping at out-scatters. Treat out-scatter like absorption, as a weight exp(-kappa\_out s), and send the lost weight out as new particles at points along the path. The mesh-profile machinery from v1.1 already draws such points. This saves steps but adds particles and variance. Worth a look only if R-v3a comes out badly.
- R-v3c: which benchmarks the thesis reports. C5G7 is the obvious multigroup one; Kobayashi and Reed are one-group and fit v2.

**Out of scope for v3**

- Fission and k-eff (Stage 4), continuous energy (Stage 5).
- Design A, unless the v3.1 decision asks for it.
- Angle-dependent multigroup data (`representation='angle'`), temperatures.

## Decisions needed from you

Only the first is needed to start; the rest can wait until their push.

| # | Decision | Options | My recommendation | Needed by |
| --- | --- | --- | --- | --- |
| 1 | Start v3.0? | Go / change the plan first | Go: plain multigroup MC is useful on its own and risk-free | v3.0 |
| 2 | Test problems | The three SRS problems; C5G7 materials; both | Both | v3.0 |
| 3 | Go/no-go threshold for Design B | e.g. at least 5 collisions per step on the reflector and graphite problems | Decide when the v3.1 table is in | After v3.1 |
| 4 | Interface v3 output for out-scatters | 5 numbers (position plus incoming direction) or fewer by symmetry | Let v3.1 settle it from the walk | v3.2 |
| 5 | Gate3 rule | Same two-condition style as gate2 (z AND an absolute tolerance) | Same style; tolerances from v3.1's numbers | v3.2 |
| 6 | Which families at v3 | All four, or the best two from v2 (cvae, nflow) | cvae and nflow first; cfm and cgan after | v3.3 |

The open v2 items stay open alongside this: balls with meshes and detectors in anisotropic materials, anisocheck in the score, and the 1-group benchmarks (Reed, Kobayashi).
