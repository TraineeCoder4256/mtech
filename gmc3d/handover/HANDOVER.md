# Handover: Innovation track (gmc3d, the 3D ball-step GMC)

Written 20 Nov 2026 by the Innovation track. It is meant for an AI with no access to Claude's memory or threads, working from the repository plus the files in `/mnt/project-files/` (this handover folder and the rest).

## 1. What this track is

This track is Phani's MTech thesis line that extends GMC (generative Monte Carlo surrogates for neutron transport) to arbitrary 3D CSG geometry. It began on 5 Oct 2026, prompted by a link from his guide about distance fields.

**The method ("ball step", walk-on-spheres).** Right after a scatter, a distance field gives the biggest one-material sphere around the particle. If its radius R (in mean free paths) is at least R*, one call to a backend moves the particle to the sphere's edge, replacing every collision inside. The backend can be the exact walk, a lookup table, or a trained generative network. Otherwise ordinary Monte Carlo takes one flight. Absorption is applied afterwards as weight × exp(−κ·s), where s is the path length inside the sphere. The geometry lives only in the distance field, so one small model serves any shape.

**Prior art.** The exact idea exists in graphics: Leonard, Höhlein and Westermann 2021, arXiv:2011.03082. Earlier related work is Fleck-Canfield 1984 (lookup tables) and collision aggregation (arXiv:2403.10432). The novelty claim has to be the transfer to neutron transport, and that is Phani's research problem.

**Role split (Phani, 5 Oct).** Claude does the engineering. Phani owns the research questions. If a research question blocks progress, stop and ask him.

## 2. The five stages (from the original SRS)

| Stage | Content | State on 20 Nov 2026 |
|---|---|---|
| 1 | Any CSG geometry, one energy, isotropic scattering, fixed source | **Done** (v1.0 to v1.6.5) |
| 2 | Adds anisotropic scattering | **Done** (v2.0.0 to v2.1.7). Model m2.1 accepted. |
| 3 | Adds multigroup energy (Design B) | **Planned.** v3 plan written, waiting for Phani's go on v3.0. This is the MTech target. |
| 4 | Adds criticality (k-eff) | Stretch goal |
| 5 | Continuous energy, inside OpenMC | Future work |

SRS documents (Claude Docs, which may not open outside claude.ai; markdown exports are in `handover/docs/` (srs-5-stages.md, srs-version1.md, v3-plan-multigroup.md, anisotropy-explainer.html)):
- Original 5-stage SRS: https://claude.ai/artifact/595HXfgwYmVUHJrJ63Wh2y
- Version 1 SRS: https://claude.ai/artifact/9ZZ9omJcQphwr2uw7WnMAE
- v3 plan (multigroup): https://claude.ai/artifact/2RCF2TQJSCTJTHwdzoXMw5. A PDF copy is at `/mnt/project-files/innovation/gmc3d-v3-plan-multigroup.pdf`.
- Anisotropy explainer page: https://claude.ai/artifact/FhhSZZa3PVxgimdipynTDr

## 3. Repository and rules

- Repo `TraineeCoder4256/mtech`, branch **`3d-gmc`**, code in **`gmc3d/`**. HEAD on 20 Nov is `fab34d7`, v2.2.2.
- Branch rules:
  - **No PR. Never touch `main`.**
  - Fetch and rebase onto `origin/3d-gmc` right before every push.
  - Never force-push.
  - The cloud git gateway refused tag pushes, so Phani pushes the version tags himself.
- `gmc2d/` (the 2D lattice work) is frozen. Copy from it; never import it.
- **Versioning (Phani, 27 and 29 Oct).** Every push carries a version, recorded in `gmc3d/VERSION` plus an entry in `gmc3d/CHANGELOG.md`. Version numbers go by area: v1.1 mesh, v1.2 sources, v1.3 multi-filter, v1.4 geometry, v1.5 medium tallies, v1.6 hardening, v2.x anisotropy, v3.x multigroup. Fixes inside an area increment the last digit.
- **GAPS rule.** Every implementation updates `gmc3d/GAPS.md`. Its three parts are real gaps, places where OpenMC refuses or is wrong, and open checks. Each gap is marked refused, warned, or silently ignored, and has a "model change?" column.
- **Two owners on one branch (agreed 5 Nov).**
  - **Innovation** owns `core/`, `problems/`, `ball/walk.py`, the walk half of `ball/data.py`, `ball/profile.py`, `ball/interface.py` (the model spec), `ball/laws.py`, `checks/`, `run.py`, README, GAPS, VERSION and the top-level CHANGELOG.
  - **Development** owns `ball/models/`, with its own `ball/models/CHANGELOG.md` numbered m1.0, m1.1 and so on. See `handover/development.md`.
  - Each commit touches only its owner's files.
  - Build only when both sides agree. A disagreement goes to Phani.
  - Handoff: an interface change is its own commit, "interface vN". Development then adapts the models and runs the gate. Innovation decides acceptance and logs "model mX.Y accepted" in the top-level CHANGELOG.

## 4. How Phani wants to work (important)

- **No full fixes or new features without his explicit approval.** Small throwaway tests that point a direction are fine.
- Within a version he has approved, dig into disagreements fully and report at the end, unless blocked.
- Every report gives what was run, the raw numbers, the diagnosis, and the limit that stopped it there.
- Explain the design reasoning in plain language with analogies, so he can defend it to examiners.
- Never headline projections, and never use the lookup table as the answer. He said "drop the table, judge the network with GPU in mind" on 5 Oct.
- Speed quoting rule:
  - OpenMC and our code run in the same session on the same machine.
  - Take the median of at least 3 seeds and quote it as about ±15%.
  - Always say which OpenMC mode the comparison is against (analog or implicit/survival biasing).

## 5. Environment

- Python with OpenMC 0.16 (multigroup mode), numba, scipy and PyTorch. On our container this was a micromamba env at `/opt/mm/root/envs/gmc`. Fresh install:
  `mamba install -c conda-forge openmc=0.16 numba scipy pytorch`
- Run everything from `gmc3d/`. Results go to `runs/<date>/<command>/`.
- Trained weights are not in git (`data/models/` is ignored). The shared home is `/mnt/project-files/models/<family>[_v2]_seed<k>/` (`model.pt` + `manifest.json`). There is also a zip at `models/gmc3d-models-v2.1.6.zip`. The registry copies weights from the shared home if they are missing locally.
- The defaults need two models:
  - `nflow_seed0`: isotropic balls, interface v1.
  - `cvae_v2_seed0`: anisotropic balls, interface v2, the accepted m2.1.
- GPU runbook (setup on Colab, data, training, testing): `/mnt/project-files/innovation/gpu-pipeline.md`. Since v2.2.0, training and inference can also run on the GPU; the commands are in README "Running it".
- **Gotcha: stale numba cache.** With `cache=True`, numba keeps stale cross-module callees. After editing any njit function or changing its signature, delete `**/__pycache__/*.nbi` and `*.nbc`.

Key commands, all in `gmc3d/README.md`:

```
python -m pytest -q tests                  # 288 passed / 38 skipped (CUDA-only) on v2.2.2
python run.py validate                     # Stage 1 problems vs OpenMC
python run.py anisocheck [--backend X]     # 20 anisotropic cases vs OpenMC
python run.py gate2 cvae_v2_seed0          # model acceptance gate, 343 checks
python run.py race [--device cuda]         # FOM vs OpenMC analog + implicit
python run.py gpubench --device cuda       # everything a GPU changes, one go
python run.py score                        # 9.9/10 at v1.6.x
```

## 6. Version history and what each one proved

Full detail is in `gmc3d/CHANGELOG.md`, and per-version reports are in `gmc3d/runs/<date>/` and `/mnt/project-files/innovation/v1.x/`.

| Version | Commit | What it did |
|---|---|---|
| Stage 1 | 32a5823, 73fd9c1, 9af3ee5 | CSG, distance field, plain MC, exact/table/network backends. Results in `runs/2026-10-05/STAGE1.md`; first race in `runs/2026-10-05/race.log`. |
| v1.0 | 177c5db | Stage 1 base. |
| v1.1.0 | 8b80020 | Mesh flux right inside balls, exact on average. Path length drawn ∝ exp(−κT), radius from Q(r\|l,R). Surface currents. See `runs/2026-10-27/MESH.md`. |
| v1.1.1 / v1.1.2 | 3608236 / bd04590 | GAPS.md; `V1_PLAN.md`. |
| v1.1.3 | 0764f63 | Settings warnings, refusals, scatter/total scores, 1D/2D/rectilinear meshes. |
| v1.2.0 | 0013942 | Every OpenMC source kind (`core/sources.py`). Fixed a silent bug where strengths were ignored. |
| v1.3.0 | 052abb5 | Tallies with several filters, partial currents, EnergyFilter. |
| v1.4.0 / v1.4.1 | 3db14a2 / 7b5ed2f | Any-axis cylinders and cones, white and periodic walls, hex lattices, tori, quadrics, rotational periodic, distributed materials. |
| v1.5.0 | ef479a3 | Cylindrical and spherical meshes, currents on any surface, MeshSurfaceFilter. |
| v1.6.0 | 641c6c4 | Point detectors, a robustness suite (30 random models vs OpenMC), `check`, `score`. |
| v1.6.1 | da4221c | Every non-physics OpenMC input: general scorer `core/ext.py`, `gapcheck` 20/20. |
| v1.6.2 | d074f17 | **Interface v1** (`ball/interface.py`): R in; mu and s out. |
| v1.6.3–v1.6.5 | b822a93, 56722a1, 75b53b8 | Every model family raced. 3 seeds, medians in `runs/2026-11-05/race-models`. |
| v2.0.0 | e6f00a2 | Anisotropic scattering in plain MC (`core/scattering.py`), sampled exactly as OpenMC does. anisocheck 13/13. |
| v2.1.0 | 1d9f353 | Anisotropic balls with the exact walk (`walk_aniso`). anisocheck 20/20. |
| v2.1.1 | 0b232f4 | **Interface v2**: R + g1..g12 in; cos_n, mu, phi, s out. Training laws in `ball/laws.py`. |
| v2.1.2 | e12d6cb | gate2 (`checks/gate_v2.py`). |
| v2.1.3 | 2278ce3 | One-scatter walks drawn exactly; models learn k ≥ 2 only. |
| v2.1.4–v2.1.6 | bdba9f0, 358e749, 78aebe6 | **m2.1 accepted**: `cvae_v2_seed0`, gate2 0/343, anisocheck 20/20. Weight rule fixed as \|z\| > 3.29 AND absorbed share off by more than 2%. |
| v2.1.7 | ccb2fd7 | cGAN (m2.2) not accepted: absorbed share 2.1–4.5% low. |
| v2.1.8 | 785a033 | Catch-up anisocheck for every v2 model; v1 race seed 11. |
| v2.1.9 / v2.1.10 | 1219d08 / c17c223 | `--device` everywhere; GPU training script, CPU-identical. |
| v2.2.0 | eb20cc9 | `transport.run(in_flight=k, overlap=)`: bigger GPU calls, CPU tracks while GPU draws. Exact walk stays bit-identical. |
| m2.3 / m2.4 (Development) | 9bfe2b4 / f260efa | Train on GPU (fp16, compile); fast inference (fp16, compile, CUDA graphs, submit/collect). |
| v2.2.1 / v2.2.2 | a2343df / fab34d7 | `run.py gpubench`: tests, model bench, gate2 fp32 vs graphs+fp16, race 5 problems × 3 seeds with CPU and GPU network rows. |

## 7. Results to know (measured; quote with the rules in section 4)

- **Correctness.**
  - Every validation problem matches OpenMC. Score is 9.9/10; the only fail is the slab depth chi², explained in section 9.
  - anisocheck 20/20 with the exact walk and with m2.1.
  - The OpenMC 0.16 bugs we found are listed in GAPS.md part 2b. They include the histogram scatter score being ×order/2, SphericalMesh at θ = π/2, currents in moved universes, the sign of rotational-periodic currents, and curved-mesh inward faces.
- **Speed, 5 Oct race** (4M histories, 4 CPU cores, FOM vs OpenMC analog):
  - Exact-walk balls: slab 2.87, sphere 2.01, curved 1.88, cask 1.70, nested 0.77, lattice3d 0.54.
  - Network on CPU: 0.39–1.05, slower almost everywhere.
- **Speed since then.** By v1.6.1, plain MC had slowed 1.8–2.3x since v1.0, from the source sampler and a 32-array Geometry namedtuple. Balls then no longer beat OpenMC (FOM 0.49/0.50/0.91). Tally scoring is not where we lose. The Speed track has the details (`handover/speed.md`).
- **Draw cost, CPU.**
  - m2.1: 1.75 µs per anisotropic ball, against 1.35 µs for the exact walk.
  - T4 GPU (6 Oct, untrained network of the same size): 0.30 µs per draw compiled at 100k+ per call.
  - **Real GPU numbers for m2.1/m2.4 have not been measured yet.** They come from Phani running `python run.py gpubench --device cuda`.
- **Physics notes.**
  - Anisotropy acts like thinning the problem by 1/(1−g).
  - Energy should help: collisions per region grow as neutrons slow down. See `handover/memory/gmc3d-feature-growth-effects.md`.

## 8. What is waiting

1. **Phani's GPU run.** `python run.py gpubench --device cuda` on his GPU node, with weights `nflow_seed0` and `cvae_v2_seed0` in `data/models/` and OpenMC installed. Output: env.json, tests.log, models/, gate2_fp32 vs gate2_graphs_fp16, race.json. fp16 with graphs may be used only if gate2 stays at 0/343.
2. **Go on v3.0** (see the v3 plan doc and PDF). The plan for Stage 3, multigroup, Design B:
   - A ball step ends on escape or on an out-of-group scatter, using q = Σs,g→g / Σs,g.
   - v3.0: multigroup plain MC plus `mgcheck` against OpenMC MG.
   - v3.1: Design B with the exact walk, plus a go/no-go on the number of collisions replaced.
   - v3.2: interface v3 plus gate3.
   - v3.3: models (Development).
   - v3.4: accept a model.
   - v3.5: problems plus a race.
   - Data: C5G7 7-group and our own `openmc.XSdata`. The container had no continuous-energy data.
3. **Proposal v10 items still unbuilt** (6 Nov audit, `handover/memory/proposal-v10-remaining-dev-6nov.md`):
   - Reed's problem and the Kobayashi benchmarks as gmc3d problems (Innovation).
   - Two-sample tests (MMD, classifier test).
   - Generalisation tests.
   - Time-to-accuracy including training cost.
   - Multigroup.
4. **Open decisions for Phani:**
   - The effective-dof pass rule (section 9).
   - Whether to race against OpenMC with weight windows (needed for the shielding claim).
   - Whether to report the OpenMC bugs upstream.
   - Whether to take a second look at the 2% weight rule ("accept for now, look later if it matters").

## 9. Known open checks (GAPS.md part 3)

- Plain MC is 5–16% slower than v1.2, and 6–7% slower than v1.6.0. The cause is partly unknown.
- The validation pass rule treats mesh bins as independent, but neighbouring bins are correlated (slab N_eff ≈ 6 of 60 bins). A correct code therefore fails about 7 in 40 seeds. An effective-dof rule passes it; Phani has to decide.
- Point detectors inside anisotropic media have no reference to check against.
- The robustness suite only draws isotropic laws.

## 10. Gotchas worth keeping

- **numba**
  - A constant negative index (`a[t, j, -1]`) made a kernel about 10x slower; guard it.
  - `inline='always'` broke early returns.
  - Storing into a namedtuple array inside a `prange` body did not reach the called function; use a helper call.
  - Every array added to the Layout/Geometry namedtuples slows every kernel call. Put new data in a separate namedtuple.
- **OpenMC 0.16**
  - Segfaults on MG track-length `scatter` tallies in void cells.
  - Stops on a distribmat fill with a None entry.
  - Drops `<macroscopic>` when reading XML materials.
  - Never resets the scatter cosine between histories (Mu filter).
  - Tallies are per particle; leakage is per unit source weight.
  - At a crossing, CellFilter is the cell entered and CellFromFilter the cell left.
- **Seeds.** `mc.solve` (gmc2d) seeds its 64 blocks as seed, seed+1, and so on, so seeds closer than 64 apart reuse random numbers.
- **Process kill.** `pkill -f <pattern>` inside a shell call kills that shell itself.
- **Shared model folder.** `run.py train` copies its output to `/mnt/project-files/models/`. A throwaway test seed must be deleted there too.
- **RAM.** Building the v2 data peaks at 9.3 GB.

## 11. Where things are

- **Code and docs:** `gmc3d/README.md` (file map), `CHANGELOG.md`, `GAPS.md`, `ACCEPTED_INPUTS.md`, `V1_PLAN.md`, `runs/`.
- **Shared files:** `/mnt/project-files/innovation/`:
  - idea, open problems R1–R12 and E1–E15
  - OpenMC coverage
  - energy design notes
  - per-version reports
  - the GPU runbook
  - the v3 plan PDF
- **Weights:** `/mnt/project-files/models/`.
- **Claude's shared memory, copied as plain markdown:** `/mnt/project-files/handover/memory/` (start at `MEMORY.md`).
