"""Every number a model is scored by.  Shared by run.py for every model.

Three levels, from one cell up to the whole problem.

1. CELL PHYSICS, MARGINALS -- ported unchanged from evaluate.py.  For a few
   cell shapes at fresh entry conditions, the Wasserstein-1 distance between
   the model's and Monte Carlo's distribution of each exit quantity on its
   own (exit point, Omega_x, log path length), divided by the same distance
   between two Monte Carlo runs.  1.0 = indistinguishable from noise.

2. CELL PHYSICS, JOINT -- new.  Marginals can all look right while the joint
   distribution is wrong: a correlation between exit point and path length,
   a mode dropped in one corner.  Those are exactly how GANs and VAEs tend
   to fail, and this repo has been caught by it before (old-tree commit
   938659a: marginal tests passed while a classifier found 4.22% of path
   lengths clamped to the straight-line minimum).  Two tests on all six
   exit coordinates at once:

     SLICED W1   by the Cramer-Wold theorem two distributions are equal iff
                 every 1-D projection is.  Project onto 64 random directions,
                 average the W1 distances, divide by the MC-vs-MC value.
     C2ST        train a small classifier to tell model samples from MC
                 samples.  Held-out accuracy 50% = it cannot, which is the
                 strongest "they match" a finite sample can give.  Reported
                 with its z against 50%, and with the same test run on two MC
                 samples so the classifier's own bias is visible.

3. THE WHOLE PROBLEM -- scored against the frozen reference (benchmark.py),
   not against a fresh noisy MC run.  Every error is quoted beside the NOISE
   FLOOR: how far an exact method with the same particle count lands from
   the reference purely by chance.  error / floor = 1 is the best possible.
"""
import numpy as np
import torch

import mc

CELL_SHAPES = [(1.0, 1.0), (4.0, 4.0), (4.0, 1.0), (1.0, 4.0), (8.0, 2.0)]
CELL_N = 20000
N_DIRS = 64
C2ST_STEPS, C2ST_WIDTH = 400, 64


def w1(a, b):
    """Wasserstein-1 between two samples, via quantiles."""
    q = np.linspace(0, 1, 400)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------- 1 + 2. one cell
def exit_features(out, W, H):
    """The six exit coordinates the models are trained on, with the exit
    point on a circle so that the wrap-around corner is not a false edge."""
    th = 2 * np.pi * out["p"] / (2 * (W + H))
    return np.column_stack([np.cos(th), np.sin(th), out["dir"],
                            np.log(np.maximum(out["s"], 1e-300))])


def sliced_w1(ref, other, rng):
    """Mean W1 over N_DIRS random projections of standardised features."""
    mu, sd = ref.mean(0), ref.std(0) + 1e-12
    a, b = (ref - mu) / sd, (other - mu) / sd
    dirs = rng.normal(size=(N_DIRS, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return float(np.mean([w1(a @ d, b @ d) for d in dirs]))


def c2st(x0, x1, seed):
    """Held-out accuracy of a classifier trained to separate x0 from x1."""
    g = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.vstack([x0, x1]).astype(np.float32))
    y = torch.cat([torch.zeros(len(x0)), torch.ones(len(x1))])
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    perm = torch.randperm(len(x), generator=g)
    tr, te = perm[: len(x) // 2], perm[len(x) // 2:]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], C2ST_WIDTH), torch.nn.SiLU(),
                              torch.nn.Linear(C2ST_WIDTH, C2ST_WIDTH), torch.nn.SiLU(),
                              torch.nn.Linear(C2ST_WIDTH, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(C2ST_STEPS):
        idx = tr[torch.randint(len(tr), (2048,), generator=g)]
        opt.zero_grad()
        lossf(net(x[idx]).squeeze(1), y[idx]).backward()
        opt.step()
    with torch.no_grad():
        acc = float(((net(x[te]).squeeze(1) > 0).float() == y[te]).float().mean())
    return acc, (acc - 0.5) / np.sqrt(0.25 / len(te))


def cell_check(sampler, rng):
    """Exit distributions vs MC for several shapes, at unseen conditions.

    The marginal part is evaluate.cell_check() line for line -- same draws
    from `rng`, same seeds -- so its numbers reproduce evaluate.py's.  The
    joint tests run on the same three samples afterwards with their own
    seeds, so they cannot disturb it."""
    rows = []
    for j, (W, H) in enumerate(CELL_SHAPES):
        xi = rng.uniform(0.15, 0.85)
        r, th = np.sqrt(rng.uniform(0.1, 0.9)), rng.uniform(-1.0, 1.0)
        ox, oy = max(r * np.cos(th), 1e-3), r * np.sin(th)

        a = mc.sample_cell(CELL_N, W, H, xi=xi, direction=(ox, oy), seed=101)
        b = mc.sample_cell(CELL_N, W, H, xi=xi, direction=(ox, oy), seed=202)
        g = sampler.sample(np.full(CELL_N, W), np.full(CELL_N, H),
                           np.full(CELL_N, xi), np.full(CELL_N, ox),
                           np.full(CELL_N, oy), seed=303)
        out = {"W": W, "H": H, "xi": xi, "ox": ox, "oy": oy,
               "uncollided_mc": float((a["k"] == 0).mean()),
               "uncollided_gmc": float(g["uncollided"].mean())}
        for name, va, vb, vg in (
                ("p", a["p"] / (2 * (W + H)), b["p"] / (2 * (W + H)),
                 g["p"] / (2 * (W + H))),
                ("Ox", a["dir"][:, 0], b["dir"][:, 0], g["dir"][:, 0]),
                ("log s", np.log10(a["s"]), np.log10(b["s"]),
                 np.log10(g["s"]))):
            sd = va.std() + 1e-12
            out[name] = (w1(va, vg) / sd, w1(va, vb) / sd)   # model, floor

        fa, fb, fg = (exit_features(o, W, H) for o in (a, b, g))
        jr = np.random.default_rng(7000 + j)
        sw_model, sw_floor = sliced_w1(fa, fg, jr), sliced_w1(fa, fb, jr)
        out["sliced_w1"] = (sw_model / sw_floor, 1.0)
        out["c2st_model"] = c2st(fa, fg, seed=8000 + j)
        out["c2st_floor"] = c2st(fa, fb, seed=9000 + j)
        rows.append(out)
    return rows


# ------------------------------------------------------- 3. whole problem
def lattice_accuracy(gmc_cells, ref):
    """Score GMC macro-cell fields against the frozen reference.

    gmc_cells: list of independent GMC fields (7x7) at benchmark.N particles.
    ref:       benchmark.load_reference(scale).
    """
    R = ref["mc_cell"]
    floor_runs = ref["floor_cells"]
    floor = [rel_l2(f, R) for f in floor_runs]
    pairs = [rel_l2(floor_runs[i], floor_runs[j])
             for i in range(len(floor_runs)) for j in range(i)]
    errs = [rel_l2(g, R) for g in gmc_cells]
    G = np.mean(gmc_cells, 0)

    # per-cell bias test: one N-particle solve scatters by the spread of the
    # floor runs; the mean of the GMC runs by that over sqrt(#runs)
    sd_n = floor_runs.std(0, ddof=1)
    se = np.sqrt(sd_n ** 2 / len(gmc_cells) + ref["mc_cell_se"] ** 2)
    z = (G - R) / np.where(se > 0, se, np.inf)
    relc = np.abs(G - R) / np.maximum(R, 1e-300)
    out = {
        "error": float(np.mean(errs)), "errors": errs,
        "floor": float(np.mean(floor)), "floor_lo": float(min(floor)),
        "floor_hi": float(max(floor)),
        "floor_pairwise": float(np.mean(pairs)),
        "bias": rel_l2(G, R),
        "bias_noise_only": float(np.mean(floor)) / np.sqrt(len(gmc_cells)),
        "flux_ratio": float(G.sum() / R.sum()),
        "cell_rel_median": float(np.median(relc)),
        "cell_rel_p90": float(np.percentile(relc, 90)),
        "cell_rel_max": float(relc.max()),
        "cell_chi2": float(np.mean(z ** 2)),
        "cell_max_abs_z": float(np.abs(z).max()),
    }
    if len(gmc_cells) > 1:
        out["gmc_pairwise"] = rel_l2(gmc_cells[1], gmc_cells[0])
    out["ratio"] = out["error"] / out["floor"]
    return out
