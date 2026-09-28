"""Every number a model is scored by.  Shared by run.py for every model.

Four checks, from one cell up to the whole problem.

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

2b. THE ABSORBER TAIL -- new.  The weight that survives one absorber cell,
   model vs MC.  It depends on the rare shortest paths, which the tests
   above barely see, and it is where cfm fails as cells thicken.

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
CHI2_MIN_REL = 0.2      # chi2 scores only cells one N-particle MC solve resolves this well
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


# ------------------------------------------- 2b. the absorber-cell tail
ABSORBER_N = 100000
ABSORBER_ENTRIES = [(0.5, 1.0, 0.0), (0.3, 0.6, 0.5)]   # (xi, Omega_x, Omega_y)


def absorber_transmission(sampler, scales):
    """How much weight survives one crossing of an absorber cell, model vs MC.

    solve.py multiplies a particle's weight by exp(-sig_a * s_cm) each time it
    crosses an absorber cell, with s_cm = s / sig_s the path in cm.  So what
    the lattice needs from the model in those cells is not the typical path
    but mean(exp(-(sig_a/sig_s) * s)), and with sig_a/sig_s = 19 that mean is
    decided by the SHORTEST paths: particles that scatter near the entry face
    and leave again.  About 0.8% of crossings, at s ~ 0.05 mean free paths,
    carry two thirds of it, whatever the cell size (runs/2026-09-23/FINDINGS.md).
    The marginal and joint tests above weigh every sample equally and barely
    see that tail; this test weighs it the way the transport does.

    Same cell, same entry, model and mc.sample_cell, for each scale's absorber
    cell (W = H = sig_s x pitch).  Returns per scale and entry: the MC and
    model means, their ratio (1.00 = right), and z of the difference.
    """
    out = {}
    for sc in scales:
        prob = mc.lattice(sc)
        ab = prob["sig_a"] > 0
        ss = float(np.unique(prob["sig_s"][ab]).item())
        kappa = float(np.unique(prob["sig_a"][ab]).item()) / ss
        W = ss * prob["pitch"]
        rows = []
        for j, (xi, ox, oy) in enumerate(ABSORBER_ENTRIES):
            a = mc.sample_cell(ABSORBER_N, W, W, xi=xi, direction=(ox, oy),
                               seed=4_000_000 + 1000 * j + int(sc))
            g = sampler.sample(np.full(ABSORBER_N, W), np.full(ABSORBER_N, W),
                               np.full(ABSORBER_N, xi), np.full(ABSORBER_N, ox),
                               np.full(ABSORBER_N, oy), seed=5_000_000 + 1000 * j + int(sc))
            tm, tg = np.exp(-kappa * a["s"]), np.exp(-kappa * g["s"])
            se = np.hypot(tm.std(), tg.std()) / np.sqrt(ABSORBER_N)
            rows.append({"xi": xi, "ox": ox, "oy": oy, "mc": float(tm.mean()),
                         "model": float(tg.mean()), "ratio": float(tg.mean() / tm.mean()),
                         "z": float((tg.mean() - tm.mean()) / se)})
        out[sc] = {"W": W, "sig_a_over_sig_s": kappa, "entries": rows,
                   "ratio": float(np.mean([r["ratio"] for r in rows]))}
    return out


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

    # per-cell bias test.  Each cell's z compares the mean of the GMC fields
    # with the reference, in units of that mean's own standard error, taken
    # from the SPREAD OF THE GMC RUNS THEMSELVES (a model need not scatter
    # like Monte Carlo).
    #
    # Only cells that one N-particle MC solve resolves to better than
    # CHI2_MIN_REL are scored.  That resolution is read off the 40M-history
    # reference (its standard error scaled from its history count to N),
    # NOT off the 6 floor runs: in a dark cell most N-particle solves score
    # nothing and a rare one scores a lot, so 6 runs can all be zero or all
    # alike and look perfectly resolved while the next run is off by 300%.
    # Those cells made the old test score an EXACT method at chi2 = 101
    # (oracle, scale 10) and 5,900 (fresh MC, scale 10).
    #
    # With n runs the spread has n-1 degrees of freedom, so z follows a
    # Student t and an exact method averages z^2 = (n-1)/(n-3), not 1.  The
    # reported chi2 divides that out.  Calibrated on 12 groups of 6 fresh
    # exact MC solves per scale: group means 1.04, 0.84, 0.63, 1.03 at scales
    # 1, 4, 10, 20, single groups from 0.33 to 2.28.  So ~1 is clean, and
    # well above ~2.5 is a systematic error.
    n = len(gmc_cells)
    rel_n = (ref["mc_cell_se"] / np.maximum(R, 1e-300)
             * np.sqrt(float(ref["mc_histories"]) / float(ref["floor_n"])))
    scored = (R > 0) & (rel_n < CHI2_MIN_REL)
    se = np.sqrt(np.var(gmc_cells, 0, ddof=1) / n + ref["mc_cell_se"] ** 2)
    scored &= se > 0
    z = (G - R)[scored] / se[scored]
    t_var = (n - 1) / (n - 3) if n > 3 else np.nan
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
        "cell_chi2": float(np.mean(z ** 2) / t_var),
        "cell_chi2_cells": int(scored.sum()),
        "cell_max_abs_z": float(np.abs(z).max()),
    }
    if len(gmc_cells) > 1:
        out["gmc_pairwise"] = rel_l2(gmc_cells[1], gmc_cells[0])
    out["ratio"] = out["error"] / out["floor"]
    return out
