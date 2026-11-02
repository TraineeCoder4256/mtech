"""v1.6: settle the slab depth-mesh item (plain MC chi2/dof 1.766 at 2M).

OpenMC at 20M histories (10x validate, 2000 batches) is the new reference.
Ours: the v1.5 validate runs (2M, seed 5) and fresh 20M runs at another
seed, plain MC and exact balls.  Run from gmc3d/:
    python runs/2026-10-30/v1.6/slab_ref.py OUTDIR
"""
import sys, time
sys.path.insert(0, "/home/claude/mtech/gmc3d")
from ball.oracle import Oracle
from checks import compare, openmc_ref, validate
from core import openmc_import, transport
from problems import PROBLEMS

out = sys.argv[1]
model = PROBLEMS["slab"]()
pr = openmc_import.load(model)
t = time.perf_counter()
ref20 = openmc_ref.run(model, f"{out}/openmc_slab_20M", 10_000, 2000, seed=777)
print(f"OpenMC 20M (2000 x 10k, seed 777): {time.perf_counter() - t:.0f} s", flush=True)
ref2 = openmc_ref.run(PROBLEMS["slab"](), f"{out}/openmc_slab_2M", 5000, 400, seed=105)
print("OpenMC 2M (validate settings, seed 105) rerun for the old reference", flush=True)

def show(label, r, ref):
    c = compare.compare(r, ref)
    ok, bad = validate.verdict(c)
    print(compare.report(c, f"   {label}"))
    print(f"   {label}: {'PASS' if ok else 'FAIL'} {'; '.join(bad)}", flush=True)

for tl in pr.tallies:        # OpenMC's results carry no filter label
    ref2.tallies[tl.name]["filter"] = tl.filter
    ref2.tallies[tl.name]["scores"] = tl.scores
print("\n== old reference (OpenMC 2M) vs new reference (OpenMC 20M)")
show("OpenMC 2M vs OpenMC 20M", ref2, ref20)
for label, kw, seed, n in (("plain MC 2M seed 5", {}, 5, 20),
                           ("exact balls 2M seed 5", "balls", 5, 20),
                           ("plain MC 20M seed 11", {}, 11, 200),
                           ("exact balls 20M seed 11", "balls", 11, 200)):
    t = time.perf_counter()
    if kw == "balls":
        r = transport.run(pr, 100_000, n, seed=seed, backend=Oracle(),
                          r_star=1.0, mesh_rule="profile")
    else:
        r = transport.run(pr, 100_000, n, seed=seed)
    print(f"\n== {label} ({time.perf_counter() - t:.0f} s)")
    show(f"{label} vs OpenMC 20M", r, ref20)
    if n == 20:
        show(f"{label} vs OpenMC 2M", r, ref2)
