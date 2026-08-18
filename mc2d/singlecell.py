"""Single-cell transmission sampler.

This is the pure-scattering random walk the paper uses to
  * generate GMC training data ("Training data is generated from standard
    Monte Carlo simulations of pure-scattering transport ... a standard
    Monte Carlo random walk is executed until the particle exits the cell"),
  * produce the MC reference statistics of Fig. 2b, and
  * measure the MC cost-vs-optical-thickness curves of Fig. 4b.

The cell is a rectangle of optical size W x H (units: mean free paths,
i.e. sigma_s = 1 internally; divide path lengths by a physical sigma_s to
get cm).  Absorption is not simulated here -- the paper's framework scales
a pure-scattering solution to problems with absorption by exponential
attenuation of the returned path length.

Exit state, matching Eq. (11)  y = (p_exit, Omega_exit, s):
  * p_exit: unwrapped perimeter coordinate, counter-clockwise from the
    origin (0,0):  bottom [0, W), right [W, W+H), top [W+H, 2W+H) with x
    running W->0, left [2W+H, 2W+2H) with y running H->0.
  * Omega_exit = (ox, oy, oz) direction cosines; u_exit = ox is the
    "x-direction cosine" plotted in Fig. 2b.
  * s: total path length in the cell (mean free paths).
Also returned: the number of scattering events (the cost standard MC pays
and GMC avoids).
"""

import time
import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _isotropic_direction():
    oz = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - oz * oz))
    return r * np.cos(phi), r * np.sin(phi), oz


@njit(cache=True, inline="always")
def _cosine_hemisphere_px():
    ox = np.sqrt(np.random.random())
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - ox * ox))
    return ox, r * np.cos(phi), r * np.sin(phi)


@njit(cache=True, inline="always")
def _walk(x, y, ox, oy, oz, W, H):
    """Random walk in the pure-scattering rectangle until exit.

    Returns (x, y, ox, oy, oz, s_total, n_scat) at the exit point.
    """
    s_tot = 0.0
    n_scat = 0
    while True:
        ds = -np.log(np.random.random())  # sigma_s = 1 (optical units)
        if ox > 0.0:
            tx = (W - x) / ox
        elif ox < 0.0:
            tx = -x / ox
        else:
            tx = 1.0e300
        if oy > 0.0:
            ty = (H - y) / oy
        elif oy < 0.0:
            ty = -y / oy
        else:
            ty = 1.0e300
        db = tx if tx < ty else ty
        if ds < db:
            x += ox * ds
            y += oy * ds
            s_tot += ds
            ox, oy, oz = _isotropic_direction()
            n_scat += 1
        else:
            x += ox * db
            y += oy * db
            s_tot += db
            # snap onto the exited face
            if tx < ty:
                x = W if ox > 0.0 else 0.0
            else:
                y = H if oy > 0.0 else 0.0
            return x, y, ox, oy, oz, s_tot, n_scat


@njit(cache=True, inline="always")
def _perimeter(x, y, W, H):
    if y <= 0.0:
        return x                          # bottom
    if x >= W:
        return W + y                      # right
    if y >= H:
        return W + H + (W - x)            # top
    return 2.0 * W + H + (H - y)          # left


@njit(cache=True, parallel=True)
def _sample(n, W, H, mode, xin, yin, mu_in, seed, n_blocks):
    p = np.empty(n)
    dirs = np.empty((n, 3))
    s = np.empty(n)
    k = np.empty(n, dtype=np.int64)
    per = (n + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        np.random.seed(seed + b)
        lo = b * per
        hi = min(n, lo + per)
        for i in range(lo, hi):
            if mode == 0:      # boundary, left face, cosine-law incidence
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox, oy, oz = _cosine_hemisphere_px()
            elif mode == 1:    # boundary, left face, fixed entry mu = Omega_x
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox = mu_in
                r = np.sqrt(max(0.0, 1.0 - ox * ox))
                phi = 2.0 * np.pi * np.random.random()
                oy = r * np.cos(phi)
                oz = r * np.sin(phi)
            elif mode == 2:    # internal, isotropic (volumetric source)
                x = xin * W if xin >= 0.0 else W * np.random.random()
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox, oy, oz = _isotropic_direction()
            else:              # boundary, left face, isotropic incidence
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox = np.random.random()          # uniform in solid angle
                r = np.sqrt(max(0.0, 1.0 - ox * ox))
                phi = 2.0 * np.pi * np.random.random()
                oy = r * np.cos(phi)
                oz = r * np.sin(phi)
            xe, ye, oxe, oye, oze, st, ns = _walk(x, y, ox, oy, oz, W, H)
            p[i] = _perimeter(xe, ye, W, H)
            dirs[i, 0] = oxe
            dirs[i, 1] = oye
            dirs[i, 2] = oze
            s[i] = st
            k[i] = ns
    return p, dirs, s, k


def sample_single_cell(n, W, H, mode="boundary", entry_pos=None,
                       entry_mu=None, entry_law="cosine", seed=1,
                       n_blocks=64):
    """Sample n single-cell transmissions through a W x H (mfp) rectangle.

    mode="boundary": particles enter the left face.  entry_pos (in [0,1],
      fraction of H) fixes the entry height, None samples it uniformly.
      entry_mu fixes the entry x-direction cosine (azimuth random); None
      samples it by entry_law: "cosine" (isotropic incident angular flux)
      or "isotropic" (uniform over the incoming solid angle).
    mode="internal": particles born inside the cell (entry_pos = (fx, fy)
      fractions, None = uniform), isotropic direction.

    Returns dict with perimeter coord p, exit directions (n,3), path
    lengths s, and scattering counts k.
    """
    if mode == "boundary":
        yin = -1.0 if entry_pos is None else float(entry_pos)
        if entry_mu is not None:
            m, xin, mu = 1, -1.0, float(entry_mu)
        elif entry_law == "cosine":
            m, xin, mu = 0, -1.0, 0.0
        else:
            m, xin, mu = 3, -1.0, 0.0
    elif mode == "internal":
        if entry_pos is None:
            xin = yin = -1.0
        else:
            xin, yin = float(entry_pos[0]), float(entry_pos[1])
        m, mu = 2, 0.0
    else:
        raise ValueError(mode)
    p, dirs, s, k = _sample(int(n), float(W), float(H), m, xin, yin, mu,
                            int(seed), int(n_blocks))
    return {"p": p, "dir": dirs, "s": s, "k": k, "W": W, "H": H}


def time_single_cell(n, L, seed=1, n_blocks=8, repeats=1):
    """Wall-clock time (s) to push n particles through an L x L (mfp) cell
    (left-face cosine-law incidence), plus the mean scattering count.
    Used for the Fig. 4b MC cost-scaling curves."""
    best = np.inf
    for r in range(repeats):
        t0 = time.perf_counter()
        _, _, _, k = _sample(int(n), float(L), float(L), 0, -1.0, -1.0, 0.0,
                             int(seed + 1000 * r), int(n_blocks))
        best = min(best, time.perf_counter() - t0)
    return best, float(k.mean())
