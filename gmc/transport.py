"""End-to-end GMC transport: chain the learned sampler across a real mesh.

A particle is followed cell to cell.  Inside each cell the scattering random
walk is *not* simulated -- the boundary model samples the exit state in one
shot.  The tally is the same track-length estimator as the Monte Carlo
solver, fed with the sampled path length.

Two design points matter.

*Batched, not per-particle.*  Each ODE solve costs 25 network evaluations,
so tracking one particle at a time in Python would be hopeless.  Instead all
live particles advance one cell crossing per iteration and the sampler is
called once per iteration on the whole batch.  Particles die at different
times and simply drop out of the live set.

*Cells are macro-cells, not tally-mesh cells.*  The GMC cell is a region of
constant material -- for the lattice, a 1 cm lattice cell.  The sampler
returns the total path length inside the cell but not where inside it went,
so the flux it produces is inherently cell-averaged.  Any comparison against
the Monte Carlo solver must therefore coarsen the MC field to the same
macro-cell grid; comparing against the fine 112x112 field would be
comparing different quantities.

Canonical frame
---------------
The model is trained with entry on the left face.  A particle entering
through face f is rotated by ``n_rot[f]`` clockwise quarter turns to bring
f onto the left face, sampled, and rotated back:

    face      left  bottom  right  top
    n_rot        0       1      2    3

Clockwise about the cell centre maps (x,y) -> (y, L-x) and a direction
(ox,oy) -> (oy,-ox); the inverse maps (x,y) -> (L-y, x) and (ox,oy) ->
(-oy,ox).  Omega_z is untouched by an in-plane rotation.
"""

import numpy as np

from .sampler import perimeter_decode

# face indices as produced by perimeter_decode
BOTTOM, RIGHT, TOP, LEFT = 0, 1, 2, 3
_N_ROT = np.array([1, 2, 3, 0], dtype=np.int64)   # indexed by face


def _rot_cw(x, y, ox, oy, L, k):
    """Apply k clockwise quarter turns about the centre of an L x L cell."""
    x, y, ox, oy = map(np.asarray, (x, y, ox, oy))
    x, y, ox, oy = x.copy(), y.copy(), ox.copy(), oy.copy()
    for _ in range(int(k)):
        x, y = y.copy(), L - x
        ox, oy = oy.copy(), -ox
    return x, y, ox, oy


def _rot_ccw(x, y, ox, oy, L, k):
    """Inverse of _rot_cw."""
    x, y, ox, oy = map(np.asarray, (x, y, ox, oy))
    x, y, ox, oy = x.copy(), y.copy(), ox.copy(), oy.copy()
    for _ in range(int(k)):
        x, y = L - y, x.copy()
        ox, oy = -oy, ox.copy()
    return x, y, ox, oy


def macro_problem(problem, pitch, cells_per_pitch):
    """Coarsen a fine problem dict to its macro-cell (material) grid.

    Returns sig_s, sig_a on the macro grid and verifies each macro cell is
    materially uniform -- the GMC cell assumption.
    """
    ss, sa = problem["sig_s"], problem["sig_a"]
    n = cells_per_pitch
    ncy, ncx = ss.shape[0] // n, ss.shape[1] // n
    ss_m = ss[: ncy * n, : ncx * n].reshape(ncy, n, ncx, n)
    sa_m = sa[: ncy * n, : ncx * n].reshape(ncy, n, ncx, n)
    uniform = (ss_m.std(axis=(1, 3)).max() < 1e-12 and
               sa_m.std(axis=(1, 3)).max() < 1e-12)
    return ss_m[:, 0, :, 0].copy(), sa_m[:, 0, :, 0].copy(), uniform


def coarsen(phi, cells_per_pitch):
    """Average a fine flux field onto the macro-cell grid."""
    n = cells_per_pitch
    ncy, ncx = phi.shape[0] // n, phi.shape[1] // n
    return phi[: ncy * n, : ncx * n].reshape(ncy, n, ncx, n).mean(axis=(1, 3))


def run_gmc_transport(sig_s, sig_a, pitch, n_particles, sampler,
                      source_cell, seed=1, weight_cutoff=1e-12,
                      max_crossings=400, birth_sampler=None,
                      return_stats=False):
    """Solve a macro-cell problem with the learned boundary sampler.

    sig_s, sig_a : (ncy, ncx) macro-cell cross sections (cm^-1)
    pitch        : macro-cell side (cm)
    source_cell  : (icx, icy) of the isotropic volumetric source
    birth_sampler: callable(n, W, seed) -> dict like sample_single_cell,
                   used for the *birth* cell only (analog MC, since the
                   boundary model does not cover internal births)

    Returns flux[ncy, ncx] normalised per source particle.
    """
    rng = np.random.default_rng(seed)
    ncy, ncx = sig_s.shape
    tally = np.zeros((ncy, ncx))
    n = int(n_particles)

    icx0, icy0 = source_cell
    ss0 = sig_s[icy0, icx0]
    sa0 = sig_a[icy0, icx0]

    # ---- birth cell: analog MC (one call, vectorised) -------------------
    W0 = pitch * ss0
    b = birth_sampler(n, W0, int(rng.integers(1, 2**31 - 1)))
    s_phys = b["s"] / ss0                                  # mfp -> cm
    w = np.ones(n)
    if sa0 > 0.0:
        att = np.exp(-sa0 * s_phys)
        np.add.at(tally, (icy0, icx0), (w * (1.0 - att) / sa0).sum())
        w = w * att
    else:
        np.add.at(tally, (icy0, icx0), (w * s_phys).sum())

    # exit point of the birth cell, in cell-local physical coordinates
    Lp = pitch
    ex, ey, face = perimeter_decode(b["p"] * (Lp / W0),
                                    np.full(n, Lp), np.full(n, Lp))
    ox, oy = b["dir"][:, 0].copy(), b["dir"][:, 1].copy()
    icx = np.full(n, icx0, dtype=np.int64)
    icy = np.full(n, icy0, dtype=np.int64)

    alive = w >= weight_cutoff
    n_crossings = 0
    n_calls = 0

    for _ in range(max_crossings):
        if not alive.any():
            break
        # ---- step across the face into the neighbouring cell ------------
        idx = np.flatnonzero(alive)
        f = face[idx]
        nx_, ny_ = icx[idx].copy(), icy[idx].copy()
        # entry coordinate on the shared face of the NEW cell
        exl, eyl = ex[idx].copy(), ey[idx].copy()
        nx_ = np.where(f == RIGHT, nx_ + 1, np.where(f == LEFT, nx_ - 1, nx_))
        ny_ = np.where(f == TOP, ny_ + 1, np.where(f == BOTTOM, ny_ - 1, ny_))
        exl = np.where(f == RIGHT, 0.0, np.where(f == LEFT, Lp, exl))
        eyl = np.where(f == TOP, 0.0, np.where(f == BOTTOM, Lp, eyl))

        inside = (nx_ >= 0) & (nx_ < ncx) & (ny_ >= 0) & (ny_ < ncy)
        alive[idx[~inside]] = False
        keep = idx[inside]
        if keep.size == 0:
            break
        nx_, ny_ = nx_[inside], ny_[inside]
        exl, eyl = exl[inside], eyl[inside]
        oxk, oyk = ox[keep], oy[keep]

        # entry face of the NEW cell is opposite the exit face of the old
        entry_face = np.where(f[inside] == RIGHT, LEFT,
                      np.where(f[inside] == LEFT, RIGHT,
                       np.where(f[inside] == TOP, BOTTOM, TOP)))

        ss_k = sig_s[ny_, nx_]
        sa_k = sig_a[ny_, nx_]
        Wk = pitch * ss_k

        # ---- canonicalise: bring the entry face onto the left face ------
        xi = np.empty(keep.size)
        oxc = np.empty(keep.size)
        oyc = np.empty(keep.size)
        for fc in (LEFT, BOTTOM, RIGHT, TOP):
            m = entry_face == fc
            if not m.any():
                continue
            k_rot = _N_ROT[fc]
            xr, yr, oxr, oyr = _rot_cw(exl[m], eyl[m], oxk[m], oyk[m],
                                       Lp, k_rot)
            xi[m] = np.clip(yr / Lp, 1e-6, 1 - 1e-6)
            oxc[m], oyc[m] = oxr, oyr

        # grazing guard: the model is trained on Omega_x > 0
        oxc = np.maximum(oxc, 1e-6)
        nrm = np.sqrt(oxc**2 + oyc**2)
        over = nrm > 1.0 - 1e-6
        if over.any():
            oxc[over] /= nrm[over] / (1.0 - 1e-6)
            oyc[over] /= nrm[over] / (1.0 - 1e-6)

        # ---- one batched sampler call for every live particle -----------
        out = sampler.sample(Wk, Wk, xi, oxc, oyc,
                             seed=int(rng.integers(1, 2**31 - 1)))
        n_calls += 1
        n_crossings += keep.size

        s_cm = out["s"] / ss_k
        wk = w[keep]
        pos = sa_k > 0.0
        contrib = np.where(pos,
                           wk * (1.0 - np.exp(-sa_k * s_cm)) /
                           np.where(pos, sa_k, 1.0),
                           wk * s_cm)
        np.add.at(tally, (ny_, nx_), contrib)
        w[keep] = wk * np.where(pos, np.exp(-sa_k * s_cm), 1.0)

        # ---- decode the exit and rotate back to the global frame --------
        gx, gy, gface = perimeter_decode(out["p"] * (Lp / Wk),
                                         np.full(keep.size, Lp),
                                         np.full(keep.size, Lp))
        goxa, goya = out["dir"][:, 0].copy(), out["dir"][:, 1].copy()
        for fc in (LEFT, BOTTOM, RIGHT, TOP):
            m = entry_face == fc
            if not m.any():
                continue
            k_rot = _N_ROT[fc]
            xb, yb, oxb, oyb = _rot_ccw(gx[m], gy[m], goxa[m], goya[m],
                                        Lp, k_rot)
            gx[m], gy[m], goxa[m], goya[m] = xb, yb, oxb, oyb

        # recompute the exit face in the global frame from the position
        tolr = 1e-9 * Lp
        gf = np.where(gy <= tolr, BOTTOM,
             np.where(gx >= Lp - tolr, RIGHT,
              np.where(gy >= Lp - tolr, TOP, LEFT)))

        ex[keep], ey[keep], face[keep] = gx, gy, gf
        ox[keep], oy[keep] = goxa, goya
        icx[keep], icy[keep] = nx_, ny_
        alive[keep] &= w[keep] >= weight_cutoff

    volume = pitch * pitch
    flux = tally / (volume * n)
    if return_stats:
        return flux, {"crossings": n_crossings, "batched_calls": n_calls,
                      "crossings_per_particle": n_crossings / n}
    return flux
