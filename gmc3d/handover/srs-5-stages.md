# 3D GMC: Software Requirements Specification

Oct 5, 2026 · @Phani

## Purpose and scope

This spec defines what the 3D GMC system must do at each of five stages, from "any CSG geometry at one energy" to "any OpenMC problem". Each stage is a complete, checkable result, so stopping early still leaves a finished system.

**The method in one line:** at each collision, find the biggest ball of one material around the particle using a distance field, let a learned model jump the particle to the ball's edge, and use ordinary Monte Carlo wherever balls are too small.

**Split of work:** Claude owns the engineering requirements (E-tagged in the research-problem list) on branch `3d-gmc` of TraineeCoder4256/mtech. Phani owns the research questions (R-tagged). Section "Research dependencies" says which requirements wait on which research answer.

| Stage | Adds | Realistic for |
| --- | --- | --- |
| 1 | 3D, any CSG geometry, one energy, isotropic scattering, fixed source | The MTech |
| 2 | Anisotropic scattering | The MTech (stretch) |
| 3 | Multigroup energy | The MTech if time allows |
| 4 | Criticality (k-eff) | Future work |
| 5 | Continuous energy: any OpenMC problem | PhD or team scale |

**Out of scope at every stage:** speed-up in fine detail (pins, gaps, thin layers), and event-by-event tallies such as pulse-height. Those regions run as plain Monte Carlo and stay correct.

**Source documents:** the idea, coverage check, energy explainer and open-problem list in the project folder `innovation/`, all dated 5 Oct 2026.

## Definitions

| Term | Meaning |
| --- | --- |
| Mean free path (mfp) | Average distance between collisions, 1/sigma. A radius "in mfp" is radius x sigma. |
| Ball | The largest sphere around a particle that contains one material only. |
| Distance field | A function giving, at any point, the distance to the nearest material boundary. It must never overestimate. |
| Ball step | One model call that moves a particle from a ball's centre to its edge, replacing every collision in between. |
| Fallback Monte Carlo | Ordinary collision-by-collision transport, used wherever a ball is smaller than the threshold. Exact. |
| Threshold R\* | The smallest ball radius (in mfp) for which a ball step is used. Set by measurement, not guessed. |
| Oracle ball | A ball step done by an exact walk instead of the model. Same answer as Monte Carlo, used to separate driver errors from model errors. |
| Table ball | A ball step that samples from a precomputed lookup table instead of a network. The baseline the network must beat. |
| Implicit capture | Absorption applied as a weight factor exp(-sigma\_a x path length) instead of killing particles. |
| Track-length tally | Flux estimated from the total path length travelled in a region. |
| CSG | Constructive solid geometry: surfaces combined with AND/OR/NOT into regions, as in OpenMC. |
| Multigroup | Energy split into G bins (groups), with cross sections and group-to-group transfer per material. |
| Design B | Multigroup ball step that ends on escape or on a scatter out of the current group. |
| k-eff | Effective multiplication factor of a fissile system; found by fission-source iteration. |

## System overview

&#91;embedded content: system overview · solver loop, training path and OpenMC check\]

Every collision asks the distance field how big a one-material ball fits; big balls take a learned (or exact) ball step, small ones take one ordinary flight, and both loop back. The same problem file runs in OpenMC, and every tally is scored against it.

## Stage 1: any CSG geometry, one energy, fixed source

Stage 1 is done when the exact (oracle) ball driver matches OpenMC within statistics on every test problem, and the table and network ball steps have measured accuracy and speed on each. This stage carries most of the engineering; later stages extend it.

### Input and geometry

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S1-GEO-1 | Read a problem written with OpenMC's Python API (geometry, materials, source, tallies) and convert it to the solver's internal arrays. | One Python file builds both our run and the OpenMC reference run. |
| S1-GEO-2 | Surfaces: planes (axis-aligned and general), spheres, axis-aligned cylinders and cones. | Each surface type passes point-side and ray-distance unit tests. |
| S1-GEO-3 | Regions: half-spaces combined with intersection, union and complement. | Point location agrees with OpenMC's on random points. |
| S1-GEO-4 | Cells filled with a material, void or a universe; nested universes; square and hexagonal lattices; translations and rotations. | Point location agrees with OpenMC's on a nested lattice test problem. |
| S1-GEO-5 | Boundary conditions: vacuum and reflective (periodic as a stretch). | Infinite-medium and half-space tests reproduce analytic answers. |
| S1-GEO-6 | Refuse unsupported input (surfaces, filters, settings) with a message naming the feature. | No unsupported model runs silently. |

### Distance field

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S1-DF-1 | At any point, return a distance to the nearest boundary of the current cell, at every nesting level (cell, lattice element, universe, problem edge). | Unit tests per level. |
| S1-DF-2 | Never overestimate. | A random test of 10^6 points finds zero balls that contain another material. |
| S1-DF-3 | Exact for planes, spheres and cylinders; a safe lower bound for cones and any later surface type. | Unit tests against brute-force distances. |
| S1-DF-4 | Report the cost per query. | Cost per query is printed beside the cost of one collision. |

### Fallback Monte Carlo

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S1-MC-1 | 3D transport, one energy, isotropic scattering, per-material sigma\_t, sigma\_s, sigma\_a. | Matches OpenMC one-group mode within statistics on all test problems. |
| S1-MC-2 | Fixed sources: point and volume, isotropic or one direction. | Source tests match OpenMC. |
| S1-MC-3 | Implicit capture, weight cutoff and Russian roulette, as in today's mc.py. | Unbiased against analog runs. |
| S1-MC-4 | Reproducible, non-overlapping random streams. | Same seed gives identical results; different seeds share no stream (fixes today's seed+64 overlap). |

### Ball data and model

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S1-DATA-1 | Exact ball-walk generator: start at the centre, isotropic, pure scattering; record exit cosine to the surface normal, path length, collision count. Radius range about 0.1 to R\_max mfp. | Reproduces e^-R uncollided fraction and the 5 Oct throwaway numbers. |
| S1-DATA-2 | Train/validation split by radius, with held-out radius bands. | No validation radius is seen in training. |
| S1-MOD-1 | One ball-step interface: input log R; output exit cosine and path length. Uncollided particles handled exactly, outside the model. | All backends below pass the same interface tests. |
| S1-MOD-2 | Three interchangeable backends: oracle (exact walk), table (precomputed), network (generators/ families via the shared training helper). | Each can be swapped in by one setting. |
| S1-MOD-3 | Network sizes from tiny to today's size, with parameter count and cost per call reported. | A cost-vs-accuracy table per backend. |
| S1-MOD-4 | The model is never asked outside its training range. | The driver caps R at R\_max (exact, since any smaller ball is valid). |

### Driver loop

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S1-DRV-1 | At each collision: R = min(distance x sigma\_s, R\_max); ball step if R is at least R\*, otherwise one Monte Carlo flight. | Logged split of ball steps and flights. |
| S1-DRV-2 | Ball step: exit point uniform on the sphere; exit direction from the model's cosine plus a uniform azimuth; weight times exp(-sigma\_a x path). | Oracle ball driver matches plain Monte Carlo. |
| S1-DRV-3 | Batched: all live particles advance one event per iteration, one model call per iteration. | Model time per particle reported. |
| S1-DRV-4 | Leakage, weight cutoff and a maximum-event guard. | No particle is lost or counted twice. |
| S1-DRV-5 | Flux inside a ball recorded by a pluggable rule; "cap the ball at the tally cell size" available as the exact option. | Rules compared against the oracle on a fine mesh. |
| S1-DRV-6 | Timing split: model, distance queries, Monte Carlo, host overhead; collisions replaced per model call. | Printed for every run. |

### Tallies and output

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S1-TAL-1 | Cell, material and regular 3D mesh tallies of flux and absorption rate; leakage; standard errors from batch statistics. | Each matches OpenMC within statistics. |
| S1-TAL-2 | Tallies read from the OpenMC model's tally definitions (mesh and cell filters; flux and absorption scores). | One definition drives both runs. |
| S1-TAL-3 | Mesh results written as VTK for ParaView, plus NumPy files. | Opens in ParaView over the same geometry. |

### Validation and measurement

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S1-VAL-1 | Harness that runs OpenMC (one-group mode) on the same model and scores every tally as error over noise floor, as metrics.py does today. | One command produces the comparison report. |
| S1-VAL-2 | Test problems: homogeneous sphere (analytic checks), thick shield slab, 3D version of today's lattice (negative control), a curved shape, a nested-lattice problem. | All five run end to end. |
| S1-VAL-3 | Checks in order: fallback Monte Carlo against OpenMC, oracle ball against OpenMC, table and network against oracle. | Each step is reported separately. |
| S1-VAL-4 | Ball-level metrics: W1, sliced W1, C2ST and absorber tail, ported from metrics.py. | Reported per backend and radius. |
| S1-VAL-5 | Measurement run: share of collisions in balls of 3, 5, 10, 30 mfp on each test problem, and cost per ball step for each backend against one collision. | Gives R\* and the expected speed-up per problem. |

## Stage 2: anisotropic scattering

Stage 2 is done when forward-peaked scattering runs through both the fallback and the ball step, matches OpenMC's multigroup mode with the same scattering data, and the network-versus-table comparison is reported. This is the first stage where a network can beat a table.

What changes: a scatter no longer forgets the incoming direction, so the ball step is set up in a frame aligned with it. The ball and its symmetry stay; the model gains inputs and outputs.

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S2-MC-1 | Fallback samples the new direction relative to the old one from Legendre moments (up to P3, as in OpenMC's multigroup format) or a tabulated angle distribution, handling expansions that go negative. | Matches OpenMC multigroup mode with the same data on an anisotropic slab. |
| S2-DATA-1 | Ball-walk generator starts at the centre with a known incoming direction and an anisotropic first scatter; records exit position angle from the incoming axis, exit direction (cosine to normal plus azimuth in the local frame) and path length. | Reduces to the Stage 1 data when scattering is isotropic. |
| S2-MOD-1 | Ball-step interface input: log R plus the scattering shape (mean cosine, or P1-P3 coefficients). Output: about 4 numbers. | Interface tests for oracle, table and network. |
| S2-MOD-2 | Table backend extended to the new inputs, with its memory and accuracy reported. | Table vs network at equal cost per call is reported. |
| S2-DRV-1 | Driver builds the local frame from the incoming direction and the exit normal, and rotates the exit state back. | Oracle ball driver still matches plain Monte Carlo. |
| S2-VAL-1 | Test problems with forward-peaked scattering (a water-like shield at P1 and P3). | Oracle, table and network scored against OpenMC. |

## Stage 3: multigroup energy

Stage 3 is done when a G-group fixed-source problem runs with Design B ball steps, matches OpenMC's multigroup mode on the same cross-section library, and reports how many collisions each ball step replaces per group.

What changes: a neutron carries a group index. Design B ends a ball step on escape or on a scatter out of the current group; that scatter is done by the fallback, and a new ball starts there. Inside a step the neutron stays in one group, so the model stays universal.

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S3-XS-1 | Read OpenMC multigroup cross-section libraries (HDF5): total, absorption, scattering matrix with Legendre order, per material. Provide a script that makes a library with `openmc.mgxs`. | Values read back equal OpenMC's own. |
| S3-MC-1 | Fallback multigroup transport: group tracking, sampling the outgoing group from the scattering matrix, upscatter. | Matches OpenMC multigroup mode within statistics. |
| S3-MC-2 | Sources with a group spectrum. | Source spectrum test passes. |
| S3-DATA-1 | Design B ball-walk generator: inputs R in the current group's mean free paths and q, the share of scatters that stay in the group; records the event (escape or out-of-group scatter), its position or exit state, and path length. | Reduces to Stage 1 data when q = 1. |
| S3-MOD-1 | Design B interface: input log R, q (plus Stage 2 anisotropy inputs); output event type, position (radius for an out-scatter, exit angles for an escape), path length. | Interface tests for oracle, table and network. |
| S3-DRV-1 | Driver tracks the group; on an out-of-group event it samples the new group and direction by Monte Carlo and starts a new ball. | Oracle ball driver matches plain Monte Carlo. |
| S3-TAL-1 | Per-group flux and reaction rates; energy filters on the library's group boundaries. | Matches OpenMC per group. |
| S3-VAL-1 | Test problems: a 2-group and an 8-group shield or reflector, and a graphite block. | Scored against OpenMC multigroup mode. |
| S3-VAL-2 | Report collisions replaced per ball step, per group and material. | Shows where Design B pays and where it does not. |
| S3-OPT-1 | (Stretch, only if S3-VAL-2 shows Design B saves too little.) Design A: per-material model outputting exit group and per-group path lengths. | Decision recorded either way. |

## Stage 4: criticality (k-eff)

Stage 4 is done when the system computes k-eff with an error bar that matches OpenMC's multigroup mode, with ball steps active at least in non-fissile regions such as a thick reflector.

What changes: fission creates new neutrons, and the problem is solved by repeating generations until the fission source settles. The safe default is no ball steps inside fissile material (fuel is usually fine detail anyway); ball steps that emit fission sites are an option gated on research.

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S4-XS-1 | Read fission data per group: nu-sigma\_f and the fission spectrum chi. | Values read back equal OpenMC's. |
| S4-MC-1 | Eigenvalue mode: fission-source iteration in batches, inactive and active batches, k estimators (collision, absorption, track length) and their combination. | k matches OpenMC within statistics. |
| S4-MC-2 | Fission bank with position and group, resampled to a fixed population each generation; Shannon entropy to check source convergence. | Entropy settles before active batches on every test problem. |
| S4-DRV-1 | By default, no ball steps in fissile materials; ball steps allowed everywhere else. | Oracle ball driver matches plain Monte Carlo k. |
| S4-MOD-1 | (Gated on research R11.) Ball steps in fissile material that also output fission sites: count, positions inside the ball, group. | Oracle version matches plain Monte Carlo k. |
| S4-TAL-1 | k-eff with standard deviation; fission-rate tallies. | Reported every run. |
| S4-VAL-1 | Test problems: one-group analytic criticality benchmarks (Los Alamos analytic set, LA-13511) and a fissile sphere with a thick reflector. | k within statistics of the analytic value and of OpenMC. |

## Stage 5: continuous energy, any OpenMC problem

Stage 5 is done when the ball step runs inside OpenMC itself on continuous-energy problems, with OpenMC as the fallback, and matches unmodified OpenMC on shielding benchmarks. Rebuilding OpenMC's nuclear-data physics in our own code is ruled out as years of work.

What changes: energy is continuous, almost every collision changes it, so only Design A works: one model per material follows the whole slowing-down inside the ball. Absorption and energy-resolved flux must come from the model.

| ID | Requirement | Accepted when |
| --- | --- | --- |
| S5-ARCH-1 | Ball step added to OpenMC's transport loop (a fork), so geometry, physics, tallies and CAD (DAGMC) come from OpenMC. | The fork with ball steps off reproduces unmodified OpenMC bit for bit. |
| S5-ARCH-2 | Nearest-boundary distance query inside OpenMC for the current cell at every level (OpenMC today has only distance along a ray). | Never-overestimate test as in S1-DF-2. |
| S5-MOD-1 | Per-material (and per-temperature) Design A models: input radius in cm and log energy; output exit energy, exit angles, absorption weight and an energy-binned path-length spectrum. | Oracle version matches OpenMC. |
| S5-MOD-2 | Fast inference inside OpenMC: either OpenMC's event-based mode to batch calls, or a hand-coded small network at about 1 microsecond per call. | Cost per call reported against one OpenMC collision. |
| S5-MOD-3 | Secondary particles born inside a ball (fission neutrons, photons) emitted by the model, or the ball switched off in that material. | No secondary is lost. |
| S5-DATA-1 | Training data from OpenMC runs on single spheres of each material (exit states from surface-source output; in-ball spectra from track output). | Dataset reproduces OpenMC sphere tallies. |
| S5-TAL-1 | Energy-filtered tallies filled from the model's spectrum output; tallies that need each collision switch balls off. | Matches OpenMC per energy bin. |
| S5-SAFE-1 | Ball steps switchable per material; off by default for fissile materials, thermal-scattering (S(alpha,beta)) materials and resonance absorbers until each is validated. | A run reports which materials used ball steps. |
| S5-VAL-1 | Validation against unmodified OpenMC on published shielding benchmarks (for example from the SINBAD database). | Error over noise floor reported per tally. |

## Requirements for every stage

These hold at all five stages; the most important is that the system never gives a wrong answer silently.

| ID | Requirement | Accepted when |
| --- | --- | --- |
| NF-1 | An exact (oracle) ball backend always exists and matches plain Monte Carlo; every learned backend is scored against it. | Oracle check passes at every stage. |
| NF-2 | Anything unsupported or doubtful falls back to plain Monte Carlo or is refused with a message; never a silent wrong answer. | Subset checker and per-material switches in place. |
| NF-3 | Every result is reproducible: seed, settings, code commit and data hashes saved beside it, under a dated runs folder as today. | Rerunning a saved config gives identical numbers. |
| NF-4 | Every run reports what ran, the raw numbers, the timing split and the hardware. | Built into the run script output. |
| NF-5 | Speed is quoted against our own plain Monte Carlo and against OpenMC on the same hardware, CPU only as today's baseline. | Both ratios in every speed report. |
| NF-6 | Every module carries a docstring with its design reasoning, so it can be explained to a guide or examiner. | Reviewed with Phani per stage. |
| NF-7 | Unit tests for geometry and distances, integration tests against OpenMC; a fast suite runs before every push. | Suite green on every pushed commit. |
| NF-8 | All work on branch `3d-gmc`, in a new `gmc3d/` folder; `gmc2d/` and `main` untouched; no pull requests. | Branch diff touches only `gmc3d/` and docs. |
| NF-9 | Runs on Phani's laptop in his existing ML environment (Python, NumPy, Numba, PyTorch, OpenMC 0.16); new dependencies only with a stated reason. | Setup instructions in the README. |

## Research dependencies

Engineering can build every Stage 1 requirement without waiting on research; research answers decide settings (R\*, backend, spreading rule) and whether optional requirements are built. Each research question is answered with a tool the engineering provides.

| Research question | Decides | Engineering tool that answers it |
| --- | --- | --- |
| R1 Is a ball step cheaper than the collisions it replaces? | R\* and the speed claim | S1-VAL-5, S1-DRV-6 |
| R2 Network or lookup table? | Default backend | S1-MOD-2, S1-MOD-3, S2-MOD-2 |
| R3 Ball accuracy and the absorber tail | Whether the network is usable | S1-VAL-4 |
| R4 Where flux goes inside a ball | Default rule in S1-DRV-5 | S1-DRV-5 against the oracle |
| R5 Error bars for the model's bias | A reported bias estimate per run | NF-1 oracle spot checks |
| R6 Novelty and positioning | Thesis framing | (literature, no tool) |
| R7 A real problem where it pays | Headline test problem | S1-VAL-5 on candidate problems |
| R8 Anisotropic ball model | Stage 2 backend | S2-MOD-2, S2-VAL-1 |
| R9 Design B speed per group | Whether S3-OPT-1 is built | S3-VAL-2 |
| R10 Design A model | Stage 3 stretch | S3-OPT-1 |
| R11 Fission sites from inside a ball | Whether S4-MOD-1 is built | S4-DRV-1 measurements of fuel ball sizes |
| R12 Continuous-energy model | Stage 5 | S5-DATA-1, S5-MOD-1 |

## Open decisions for Phani

Each has a default that Stage 1 work will follow unless Phani picks otherwise.

| Decision | Options | Default |
| --- | --- | --- |
| Starting code for `3d-gmc` | `main`, or the development branch with the flow model and shared training helper | The development branch (newest code; it holds the training helper the ball model reuses) |
| Reuse of gmc2d code | Import from `gmc2d/`, or copy and adapt into `gmc3d/` | Copy and adapt, so `gmc2d/` stays frozen and the 2D results stay reproducible |
| Problem input language | OpenMC's Python API, or our own format | OpenMC's Python API |
| Periodic boundaries | Stage 1 or later | Stage 1 stretch |
| Stage 1 headline problem | Thick shield slab, reflector block, moderator block | Thick shield slab, until R7 says otherwise |
| Network size budget | Match today's 1.7M weights, or set by cost per call | Cost per call |
| Build order inside Stage 1 | Model first, or geometry, fallback and oracle first | Geometry, fallback Monte Carlo and oracle first; models after the oracle matches OpenMC |
