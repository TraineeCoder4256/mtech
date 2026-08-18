"""Full-domain standard Monte Carlo transport solver (the paper's MC baseline).

Physics, following arXiv:2512.13965v1 "Monte Carlo Method" section:

* Steady-state, monoenergetic transport on a 2D Cartesian mesh with
  isotropic scattering.  Geometry is 2D (x, y); particle directions live on
  the full 3D unit sphere ("2D XY" geometry).  The paper encodes directions
  as points on the unit *disk* of direction cosines (Sec. "Network
  Architecture and Inputs"), which is exactly the projection of a 3D unit
  vector onto the x-y plane, so 3D-in-angle is the faithful reading.
* Implicit capture (continuous absorption): collision distances are sampled
  from the *scattering* cross section only,  ds = -ln(xi)/sigma_s  (Eq. 4),
  and the weight is attenuated by Beer-Lambert along every track segment,
  w_out = w_in * exp(-sigma_a * ds_tot)                            (Eq. 5).
* Track-length scalar-flux estimator (Eq. 7): each track segment of length
  ds in a mesh cell of volume V deposits
      phi_est = Delta_w / (V * sigma_a),  Delta_w = w (1 - e^{-sigma_a ds})
  which reduces to the standard  w * ds / V  when sigma_a -> 0.
  The tally is normalized per source particle.
* Histories terminate when the particle leaves the domain (vacuum
  boundaries) or its weight falls below a cutoff ("its energy weight falls
  below a prescribed cutoff threshold" -- the paper discards low-weight
  particles; no Russian roulette).
"""

import numpy as np
from numba import njit, prange

WEIGHT_CUTOFF = 1.0e-12  # resolves flux levels down to ~1e-12 per particle


def make_uniform_grid(Lx, Ly, nx, ny):
    """Cell-edge coordinates of a uniform nx x ny mesh on [0,Lx]x[0,Ly]."""
    return np.linspace(0.0, Lx, nx + 1), np.linspace(0.0, Ly, ny + 1)


@njit(cache=True, inline="always")
def _isotropic_direction():
    """Uniform direction on the 3D unit sphere."""
    oz = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - oz * oz))
    return r * np.cos(phi), r * np.sin(phi), oz


@njit(cache=True, inline="always")
def _cosine_hemisphere_px():
    """Cosine-law direction about +x (isotropic incident angular flux)."""
    ox = np.sqrt(np.random.random())
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - ox * ox))
    return ox, r * np.cos(phi), r * np.sin(phi)


@njit(cache=True)
def _history(x, y, ox, oy, sig_s, sig_a, dx, dy, nx, ny, tally, wcut):
    """Follow one history to termination, accumulating track-length tallies.

    `tally` accumulates  sum of estimator contributions * V  (divided out by
    the caller), i.e. we add Delta_w/sigma_a (or w*ds for sigma_a=0).
    """
    w = 1.0
    ix = int(x / dx)
    iy = int(y / dy)
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        return
    while True:
        ss = sig_s[iy, ix]
        sa = sig_a[iy, ix]
        # distance to next scattering event, Eq. (4)
        if ss > 0.0:
            ds = -np.log(np.random.random()) / ss
        else:
            ds = 1.0e300
        # distance to cell boundary along the flight direction
        if ox > 0.0:
            tx = ((ix + 1) * dx - x) / ox
        elif ox < 0.0:
            tx = (ix * dx - x) / ox
        else:
            tx = 1.0e300
        if oy > 0.0:
            ty = ((iy + 1) * dy - y) / oy
        elif oy < 0.0:
            ty = (iy * dy - y) / oy
        else:
            ty = 1.0e300
        db = tx if tx < ty else ty
        scatter = ds < db
        step = ds if scatter else db

        # continuous absorption along the segment, Eqs. (5)-(7)
        if sa > 0.0:
            att = np.exp(-sa * step)
            tally[iy, ix] += w * (1.0 - att) / sa
            w *= att
        else:
            tally[iy, ix] += w * step

        if scatter:
            x += ox * step
            y += oy * step
            ox, oy, _ = _isotropic_direction()
        else:
            # advance exactly onto the crossed grid line and step the index
            x += ox * step
            y += oy * step
            if tx < ty:
                ix += 1 if ox > 0.0 else -1
                x = (ix if ox > 0.0 else ix + 1) * dx
                if ix < 0 or ix >= nx:
                    return
            else:
                iy += 1 if oy > 0.0 else -1
                y = (iy if oy > 0.0 else iy + 1) * dy
                if iy < 0 or iy >= ny:
                    return
        if w < wcut:
            return


@njit(cache=True, parallel=True)
def _run(n_particles, sig_s, sig_a, dx, dy, nx, ny,
         source_kind, src_lo_x, src_lo_y, src_hi_x, src_hi_y,
         seed, n_blocks, wcut):
    """source_kind 0: isotropic volumetric source uniform in the given box.
    source_kind 1: boundary source on the left edge x=0, uniform in
                   y in [src_lo_y, src_hi_y], cosine-law incident (+x).
    source_kind 2: same boundary source, isotropic over the incoming
                   hemisphere (uniform in solid angle with Omega_x > 0).
    """
    tallies = np.zeros((n_blocks, ny, nx))
    per = (n_particles + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        np.random.seed(seed + b)
        lo = b * per
        hi = min(n_particles, lo + per)
        for _ in range(lo, hi):
            if source_kind == 0:
                x = src_lo_x + (src_hi_x - src_lo_x) * np.random.random()
                y = src_lo_y + (src_hi_y - src_lo_y) * np.random.random()
                ox, oy, _ = _isotropic_direction()
            elif source_kind == 1:
                x = 0.0
                y = src_lo_y + (src_hi_y - src_lo_y) * np.random.random()
                ox, oy, _ = _cosine_hemisphere_px()
            else:
                x = 0.0
                y = src_lo_y + (src_hi_y - src_lo_y) * np.random.random()
                while True:
                    ox, oy, oz = _isotropic_direction()
                    if ox > 0.0:
                        break
            _history(x, y, ox, oy, sig_s, sig_a, dx, dy, nx, ny,
                     tallies[b], wcut)
    out = np.zeros((ny, nx))
    for b in range(n_blocks):
        out += tallies[b]
    return out


def run_transport(problem, n_particles, seed=1, n_blocks=64,
                  weight_cutoff=WEIGHT_CUTOFF):
    """Run the standard MC solver on a problem dict (see problems.py).

    Returns the scalar flux phi[ny, nx], track-length estimator,
    normalized per source particle (Fig. 3 caption).
    """
    nx, ny = problem["nx"], problem["ny"]
    dx = problem["Lx"] / nx
    dy = problem["Ly"] / ny
    src = problem["source"]
    if src["kind"] == "volumetric":
        kind = 0
    elif src.get("angular", "cosine") == "cosine":
        kind = 1
    else:
        kind = 2
    lo_x, lo_y, hi_x, hi_y = src["box"]
    tally = _run(n_particles, problem["sig_s"], problem["sig_a"],
                 dx, dy, nx, ny, kind, lo_x, lo_y, hi_x, hi_y,
                 int(seed), int(n_blocks), weight_cutoff)
    volume = dx * dy
    return tally / (volume * n_particles)
