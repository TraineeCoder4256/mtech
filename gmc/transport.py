"""Chain the sampler across a mesh to solve a real transport problem.

A particle is followed cell to cell.  Inside each cell the scattering walk
is not simulated -- the boundary model samples the exit state in one shot.
The tally is the same track-length estimator the Monte Carlo solver uses,
fed with the sampled path length.

Two things shape the code:

  Batched, not per-particle.  Every ODE solve costs several network
  evaluations, so all live particles advance one cell crossing per
  iteration and the sampler is called once per iteration on the whole
  batch.  Particles die at different times and drop out of the live set.

  Cells are macro-cells.  A GMC cell is a region of uniform material -- for
  the lattice, one 1 cm cell.  The sampler gives total path length inside
  the cell but not where inside it went, so its flux is cell-averaged and
  must be compared against a coarsened MC field, never the fine mesh.

The model is trained with entry on the LEFT face, so a particle entering
through any other face is rotated onto the left face, sampled, and rotated
back.  A clockwise quarter turn about the cell centre maps (x,y) -> (y,L-x).
"""

import time

import numpy as np

from .sampler import perimeter_decode

BOTTOM, RIGHT, TOP, LEFT = 0, 1, 2, 3
# quarter turns needed to bring each face onto the left face
N_ROT = np.array([1, 2, 3, 0])
# cos and sin of k*90 degrees, exactly
COS = np.array([1, 0, -1, 0])
SIN = np.array([0, 1, 0, -1])


def rotate(x, y, ox, oy, L, k, back=False):
    """k clockwise quarter turns about the centre of an L x L cell."""
    c, s = COS[k % 4], SIN[k % 4]
    if back:
        s = -s
    u, v = x - L / 2, y - L / 2
    return (u * c + v * s + L / 2, -u * s + v * c + L / 2,
            ox * c + oy * s, -ox * s + oy * c)


def analog_birth(n, W, seed):
    """Birth cell fallback: the model cannot start a history inside a cell."""
    from mc2d import sample_single_cell
    return sample_single_cell(n, W, W, mode="internal", seed=seed, n_blocks=8)


def macro_problem(problem, pitch, cells_per_pitch):
    """Coarsen a fine problem to its macro grid, checking material uniformity."""
    n = cells_per_pitch
    ss, sa = problem["sig_s"], problem["sig_a"]
    ncy, ncx = ss.shape[0] // n, ss.shape[1] // n
    ss_m = ss[:ncy * n, :ncx * n].reshape(ncy, n, ncx, n)
    sa_m = sa[:ncy * n, :ncx * n].reshape(ncy, n, ncx, n)
    uniform = (ss_m.std(axis=(1, 3)).max() < 1e-12 and
               sa_m.std(axis=(1, 3)).max() < 1e-12)
    return ss_m[:, 0, :, 0].copy(), sa_m[:, 0, :, 0].copy(), uniform


def coarsen(phi, cells_per_pitch):
    """Average a fine flux field onto the macro grid."""
    n = cells_per_pitch
    ncy, ncx = phi.shape[0] // n, phi.shape[1] // n
    return phi[:ncy * n, :ncx * n].reshape(ncy, n, ncx, n).mean(axis=(1, 3))


def _tally(tally, iy, ix, w, s_cm, sa):
    """Track-length estimator; returns the surviving weight.

    Absorption is applied as attenuation along the track (implicit capture),
    which is what makes a pure-scattering sampler usable in absorbing media.
    """
    absorbing = sa > 0.0
    contrib = np.where(absorbing,
                       w * (1.0 - np.exp(-sa * s_cm)) / np.where(absorbing, sa, 1.0),
                       w * s_cm)
    np.add.at(tally, (iy, ix), contrib)
    return w * np.where(absorbing, np.exp(-sa * s_cm), 1.0)


def run_gmc_transport(sig_s, sig_a, pitch, n_particles, sampler, source_cell,
                      seed=1, weight_cutoff=1e-12, max_crossings=400):
    """Solve a macro-cell problem with the learned sampler.

    Returns (flux per source particle, stats).  The stats separate wall time
    into the sampler, the analog birth cell and the host-side geometry, which
    is what tells you whether the network or the Python loop is the cost.
    """
    t0 = time.perf_counter()
    t_sampler = t_birth = 0.0
    batches = []
    sampler.reset()

    rng = np.random.default_rng(seed)
    ncy, ncx = sig_s.shape
    tally = np.zeros((ncy, ncx))
    n = int(n_particles)
    L = pitch

    # ---- birth cell: analog MC, one vectorised call --------------------
    icx0, icy0 = source_cell
    ss0, sa0 = sig_s[icy0, icx0], sig_a[icy0, icx0]
    W0 = L * ss0
    _t = time.perf_counter()
    b = analog_birth(n, W0, int(rng.integers(1, 2**31 - 1)))
    t_birth += time.perf_counter() - _t

    w = _tally(tally, icy0, icx0, np.ones(n), b["s"] / ss0, sa0)
    ex, ey, face = perimeter_decode(b["p"] * (L / W0), np.full(n, L),
                                    np.full(n, L))
    ox, oy = b["dir"][:, 0].copy(), b["dir"][:, 1].copy()
    icx = np.full(n, icx0, np.int64)
    icy = np.full(n, icy0, np.int64)
    alive = w >= weight_cutoff
    n_crossings = 0

    for _ in range(max_crossings):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        # ---- step across the face into the neighbouring cell -----------
        f = face[idx]
        nx_ = icx[idx] + (f == RIGHT) - (f == LEFT)
        ny_ = icy[idx] + (f == TOP) - (f == BOTTOM)
        # the entry point on the shared face, in the NEW cell's coordinates
        exl = np.where(f == RIGHT, 0.0, np.where(f == LEFT, L, ex[idx]))
        eyl = np.where(f == TOP, 0.0, np.where(f == BOTTOM, L, ey[idx]))

        inside = (nx_ >= 0) & (nx_ < ncx) & (ny_ >= 0) & (ny_ < ncy)
        alive[idx[~inside]] = False              # leaked out of the domain
        keep = idx[inside]
        if keep.size == 0:
            break
        nx_, ny_, exl, eyl = nx_[inside], ny_[inside], exl[inside], eyl[inside]
        f = f[inside]
        # entering through the face opposite the one just exited
        entry_face = np.where(f == RIGHT, LEFT, np.where(f == LEFT, RIGHT,
                      np.where(f == TOP, BOTTOM, TOP)))
        k_rot = N_ROT[entry_face]

        # ---- rotate the entry onto the left face -----------------------
        _, yr, oxc, oyc = rotate(exl, eyl, ox[keep], oy[keep], L, k_rot)
        xi = np.clip(yr / L, 1e-6, 1 - 1e-6)
        oxc = np.maximum(oxc, 1e-6)              # model is trained on Ox > 0
        nrm = np.hypot(oxc, oyc)
        over = nrm > 1.0 - 1e-6
        oxc[over] /= nrm[over] / (1.0 - 1e-6)
        oyc[over] /= nrm[over] / (1.0 - 1e-6)

        # ---- one batched sampler call for every live particle ----------
        Wk = L * sig_s[ny_, nx_]
        _t = time.perf_counter()
        out = sampler.sample(Wk, Wk, xi, oxc, oyc,
                             seed=int(rng.integers(1, 2**31 - 1)))
        t_sampler += time.perf_counter() - _t
        batches.append(keep.size)
        n_crossings += keep.size

        w[keep] = _tally(tally, ny_, nx_, w[keep], out["s"] / sig_s[ny_, nx_],
                         sig_a[ny_, nx_])

        # ---- rotate the exit back into the global frame ----------------
        gx, gy, _ = perimeter_decode(out["p"] * (L / Wk),
                                     np.full(keep.size, L),
                                     np.full(keep.size, L))
        gx, gy, gox, goy = rotate(gx, gy, out["dir"][:, 0], out["dir"][:, 1],
                                  L, k_rot, back=True)
        tol = 1e-9 * L
        gf = np.where(gy <= tol, BOTTOM, np.where(gx >= L - tol, RIGHT,
              np.where(gy >= L - tol, TOP, LEFT)))

        ex[keep], ey[keep], face[keep] = gx, gy, gf
        ox[keep], oy[keep] = gox, goy
        icx[keep], icy[keep] = nx_, ny_
        alive[keep] &= w[keep] >= weight_cutoff

    wall = time.perf_counter() - t0
    bs = np.array(batches) if batches else np.zeros(1)
    stats = dict(sampler.stats)
    stats.update(
        particles=n, crossings=n_crossings, crossings_per_particle=n_crossings / n,
        mean_batch=float(bs.mean()), median_batch=float(np.median(bs)),
        nfe_per_sample=sampler.nfe_per_sample, wall=wall,
        wall_sampler=t_sampler, wall_birth=t_birth,
        wall_overhead=wall - t_sampler - t_birth)
    return tally / (L * L * n), stats
