"""Chain the sampler across a mesh to solve a whole problem.

A particle is followed cell to cell.  Inside each cell the scattering walk
is not simulated -- the sampler produces the exit state in one shot.  The
tally is the same track-length estimator the MC solver uses, fed with the
sampled path length.

Two things shape the code:

  Batched, not per-particle.  Every ODE solve costs several network
  evaluations, so all live particles advance one cell crossing per
  iteration and the sampler is called once per iteration on the whole
  batch.  Particles die at different times and drop out.

  Cells are macro-cells -- regions of uniform material, here one 1 cm
  lattice cell.  The sampler gives total path length in a cell but not
  where inside it went, so its flux is cell-averaged and must be compared
  against a coarsened MC field, never the fine mesh.

The model is trained with entry on the LEFT face, so a particle entering
through any other face is rotated onto the left face, sampled, and rotated
back.  A clockwise quarter turn about the centre maps (x, y) -> (y, L - x).
"""

import time

import numpy as np

import mc
from sampler import decode_p

BOTTOM, RIGHT, TOP, LEFT = 0, 1, 2, 3
N_ROT = np.array([1, 2, 3, 0])          # quarter turns to bring face -> left
COS = np.array([1, 0, -1, 0])           # cos/sin of k*90 degrees, exactly
SIN = np.array([0, 1, 0, -1])


def rotate(x, y, ox, oy, L, k, back=False):
    """k clockwise quarter turns about the centre of an L x L cell."""
    c, s = COS[k % 4], (-SIN[k % 4] if back else SIN[k % 4])
    u, v = x - L / 2, y - L / 2
    return (u * c + v * s + L / 2, -u * s + v * c + L / 2,
            ox * c + oy * s, -ox * s + oy * c)


def macro(problem):
    """Coarsen a problem to its macro grid; check the cells are uniform."""
    n = problem["per_cm"]
    out = []
    for key in ("sig_s", "sig_a"):
        a = problem[key]
        ny, nx = a.shape[0] // n, a.shape[1] // n
        blocks = a[:ny * n, :nx * n].reshape(ny, n, nx, n)
        assert blocks.std(axis=(1, 3)).max() < 1e-12, "macro cell not uniform"
        out.append(blocks[:, 0, :, 0].copy())
    return out


def coarsen(phi, per_cm):
    """Average a fine flux field onto the macro grid."""
    ny, nx = phi.shape[0] // per_cm, phi.shape[1] // per_cm
    return phi[:ny * per_cm, :nx * per_cm].reshape(
        ny, per_cm, nx, per_cm).mean(axis=(1, 3))


def _tally(tally, iy, ix, w, s_cm, sig_a):
    """Track-length estimator with implicit capture; returns surviving weight.

    iy and ix must be per-particle arrays: np.add.at accumulates elementwise,
    and scalar indices with an array of contributions do not broadcast.
    """
    absorbing = sig_a > 0.0
    safe = np.where(absorbing, sig_a, 1.0)
    np.add.at(tally, (iy, ix), np.where(absorbing,
                                        w * (1 - np.exp(-sig_a * s_cm)) / safe,
                                        w * s_cm))
    return w * np.where(absorbing, np.exp(-sig_a * s_cm), 1.0)


def gmc_solve(problem, n, sampler, seed=1, max_crossings=400):
    """Solve `problem` with the learned sampler.  Returns (flux, stats).

    The stats split wall time into the sampler, the analog birth cell and
    the host-side geometry, which is what says whether the network or the
    Python loop is the cost.
    """
    t0 = time.perf_counter()
    t_net = t_birth = 0.0
    batches = []
    sampler.reset()

    sig_s, sig_a = macro(problem)
    L = problem["pitch"]
    ncy, ncx = sig_s.shape
    tally = np.zeros((ncy, ncx))
    n = int(n)
    rng = np.random.default_rng(seed)

    # ---- birth cell: the model cannot start a history inside a cell, so
    #      the source cell is walked by ordinary MC (one cell out of 49)
    cx0, cy0 = problem["source_cell"]
    ss0, sa0 = sig_s[cy0, cx0], sig_a[cy0, cx0]
    W0 = L * ss0
    _t = time.perf_counter()
    b = mc.sample_cell(n, W0, W0, inside=True,
                       seed=int(rng.integers(1, 2**31 - 1)))
    t_birth += time.perf_counter() - _t

    w = _tally(tally, np.full(n, cy0), np.full(n, cx0), np.ones(n),
               b["s"] / ss0, sa0)
    ex, ey, face = decode_p(b["p"] * (L / W0), np.full(n, L), np.full(n, L))
    ox, oy = b["dir"][:, 0].copy(), b["dir"][:, 1].copy()
    cx = np.full(n, cx0, np.int64)
    cy = np.full(n, cy0, np.int64)
    alive = w >= mc.WEIGHT_CUTOFF
    crossings = 0

    for _ in range(max_crossings):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        # ---- step across the face into the neighbour -------------------
        f = face[idx]
        nx_ = cx[idx] + (f == RIGHT) - (f == LEFT)
        ny_ = cy[idx] + (f == TOP) - (f == BOTTOM)
        # the entry point on the shared face, in the NEW cell's coordinates
        exl = np.where(f == RIGHT, 0.0, np.where(f == LEFT, L, ex[idx]))
        eyl = np.where(f == TOP, 0.0, np.where(f == BOTTOM, L, ey[idx]))

        inside = (nx_ >= 0) & (nx_ < ncx) & (ny_ >= 0) & (ny_ < ncy)
        alive[idx[~inside]] = False                  # leaked out
        keep = idx[inside]
        if keep.size == 0:
            break
        nx_, ny_ = nx_[inside], ny_[inside]
        exl, eyl, f = exl[inside], eyl[inside], f[inside]

        entry_face = np.where(f == RIGHT, LEFT, np.where(f == LEFT, RIGHT,
                       np.where(f == TOP, BOTTOM, TOP)))
        k_rot = N_ROT[entry_face]

        # ---- rotate the entry onto the left face -----------------------
        _, yr, oxc, oyc = rotate(exl, eyl, ox[keep], oy[keep], L, k_rot)
        xi = np.clip(yr / L, 1e-6, 1 - 1e-6)
        oxc = np.maximum(oxc, 1e-6)                  # trained on Omega_x > 0
        norm = np.hypot(oxc, oyc)
        over = norm > 1 - 1e-6
        oxc[over] /= norm[over] / (1 - 1e-6)
        oyc[over] /= norm[over] / (1 - 1e-6)

        # ---- one batched sampler call for every live particle ----------
        W = L * sig_s[ny_, nx_]
        _t = time.perf_counter()
        out = sampler.sample(W, W, xi, oxc, oyc,
                             seed=int(rng.integers(1, 2**31 - 1)))
        t_net += time.perf_counter() - _t
        batches.append(keep.size)
        crossings += keep.size

        w[keep] = _tally(tally, ny_, nx_, w[keep], out["s"] / sig_s[ny_, nx_],
                         sig_a[ny_, nx_])

        # ---- rotate the exit back to the global frame ------------------
        gx, gy, _ = decode_p(out["p"] * (L / W), np.full(keep.size, L),
                             np.full(keep.size, L))
        gx, gy, gox, goy = rotate(gx, gy, out["dir"][:, 0], out["dir"][:, 1],
                                  L, k_rot, back=True)
        tol = 1e-9 * L
        gf = np.where(gy <= tol, BOTTOM, np.where(gx >= L - tol, RIGHT,
              np.where(gy >= L - tol, TOP, LEFT)))

        ex[keep], ey[keep], face[keep] = gx, gy, gf
        ox[keep], oy[keep] = gox, goy
        cx[keep], cy[keep] = nx_, ny_
        alive[keep] &= w[keep] >= mc.WEIGHT_CUTOFF

    wall = time.perf_counter() - t0
    bs = np.array(batches) if batches else np.zeros(1)
    stats = dict(sampler.stats)
    stats.update(particles=n, crossings=crossings,
                 crossings_per_particle=crossings / n,
                 mean_batch=float(bs.mean()), median_batch=float(np.median(bs)),
                 nfe_per_sample=sampler.nfe_per_sample, wall=wall,
                 wall_net=t_net, wall_birth=t_birth,
                 wall_host=wall - t_net - t_birth)
    return tally / (L * L * n), stats
