"""cask_p1 (mesh tallies, anisotropic balls held to one voxel) at other
seeds, exact walk vs nflow_v2_seed0: is the seed-3 plan2d chi2 1.22 (p 6e-4)
the model or chance?  Run from gmc3d/."""
import sys; sys.path.insert(0, ".")
import run as R
from checks import aniso as A
be = {"oracle": None, "nflow_v2_seed0": R.backend("nflow_v2_seed0")}
prob, law, st = A.CASES["cask_p1"]
for seed in (5, 7):
    for name, b in be.items():
        print(f"seed {seed} balls by {name}", flush=True)
        out = A._compare(f"cask_p1_s{seed}_{name}", A._build(prob, law, "cask_p1", st),
                         R.DATA / "openmc", A.PARTICLES, A.BATCHES, print,
                         seed=seed, ball_backend=b)
        for lab in ("mc", "balls"):
            c = out[lab]["cmp"]
            print("   ", lab, {k: round(c[k]["chi2_dof"], 3) for k in
                               ("plan2d/flux", "duct_grid/flux", "mesh/flux")},
                  "leak z", round(c["leakage"]["z"], 2), flush=True)
