"""Single-cell transmission: walk one particle through one cell, record how
it comes out.  This is the physics the generative model learns to skip.

A particle enters a rectangular cell, scatters isotropically some number of
times, and eventually a flight carries it out.  We record:

    p      where it left, as one coordinate around the perimeter
    Omega  which way it was going, as direction cosines
    s      total path length inside the cell
    k      number of scatters -- the cost MC pays and GMC avoids

Lengths are in mean free paths (sigma_s = 1), so the distance to the next
scatter is just -ln(uniform).  This is why one trained model covers every
cell size: a cell is described by its optical size alone, not by its
physical size and cross section separately.

The cell is pure scattering.  Absorption is applied afterwards by
attenuating the returned path length, which is what keeps the walk
dependent on optical size only.

Perimeter coordinate, anticlockwise from the bottom-left corner:

        p=2W+H ......... p=2W+2H      bottom  [0,      W)     x: 0 -> W
             +-----------------+      right   [W,      W+H)   y: 0 -> H
      p=W+H  |                 | p=W  top     [W+H,    2W+H)  x: W -> 0
             +-----------------+      left    [2W+H, 2W+2H)   y: H -> 0
        p=0 ............... p=W

One number instead of two, and it wraps cleanly -- which is what lets the
model encode it as a point on a circle.
"""

import time

import numpy as np
from numba import njit, prange

# How the entry state is chosen.  These are passed into the compiled kernel
# as plain ints; the public function below translates friendly keyword
# arguments into them.
ENTRY_COSINE = 0     # left face, cosine law   (isotropic incident flux)
ENTRY_FIXED_MU = 1   # left face, fixed Omega_x, random azimuth
BORN_INSIDE = 2      # volumetric source: born in the cell, isotropic
ENTRY_ISOTROPIC = 3  # left face, uniform over the incoming solid angle
ENTRY_FIXED_DIR = 4  # left face, fixed (Omega_x, Omega_y)  <- training data

FAR_AWAY = 1.0e300   # stand-in for "never hits this face"


@njit(cache=True, inline="always")
def _isotropic_direction():
    """Uniform on the 3-D unit sphere: uniform z-cosine, uniform azimuth."""
    oz = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - oz * oz))            # radius of the z-slice
    return r * np.cos(phi), r * np.sin(phi), oz


@njit(cache=True, inline="always")
def _cosine_hemisphere_px():
    """Into +x with density proportional to Omega_x: what an isotropic flux
    looks like crossing a surface.  sqrt(xi) inverts that cosine law."""
    ox = np.sqrt(np.random.random())
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - ox * ox))
    return ox, r * np.cos(phi), r * np.sin(phi)


@njit(cache=True, inline="always")
def _walk(x, y, ox, oy, oz, W, H):
    """Follow one particle to its exit.  This is the whole simulation.

    Each iteration is one flight: work out how far to the next scatter and
    how far to the wall, and whichever is shorter is what happens.
    """
    path_length = 0.0
    n_scatters = 0

    while True:
        # free path is exponential with sigma_s = 1
        dist_to_scatter = -np.log(np.random.random())

        # distance to each pair of walls; a zero direction component never
        # reaches that pair
        if ox > 0.0:
            dist_to_x_wall = (W - x) / ox          # heading right
        elif ox < 0.0:
            dist_to_x_wall = -x / ox               # heading left
        else:
            dist_to_x_wall = FAR_AWAY
        if oy > 0.0:
            dist_to_y_wall = (H - y) / oy          # heading up
        elif oy < 0.0:
            dist_to_y_wall = -y / oy               # heading down
        else:
            dist_to_y_wall = FAR_AWAY

        hits_x_wall_first = dist_to_x_wall < dist_to_y_wall
        dist_to_wall = dist_to_x_wall if hits_x_wall_first else dist_to_y_wall

        if dist_to_scatter < dist_to_wall:
            # scatter: move there and forget the old direction entirely
            x += ox * dist_to_scatter
            y += oy * dist_to_scatter
            path_length += dist_to_scatter
            ox, oy, oz = _isotropic_direction()
            n_scatters += 1
        else:
            # escape: move to the wall, keeping the current direction
            x += ox * dist_to_wall
            y += oy * dist_to_wall
            path_length += dist_to_wall
            # snap onto the face so floating point cannot leave the exit
            # point a hair inside or outside
            if hits_x_wall_first:
                x = W if ox > 0.0 else 0.0
            else:
                y = H if oy > 0.0 else 0.0
            return x, y, ox, oy, oz, path_length, n_scatters


@njit(cache=True, inline="always")
def _perimeter(x, y, W, H):
    """Exit point -> perimeter coordinate.  Order matters: a corner belongs
    to whichever face is tested first, which keeps p single-valued."""
    if y <= 0.0:
        return x                          # bottom, left to right
    if x >= W:
        return W + y                      # right, bottom to top
    if y >= H:
        return W + H + (W - x)            # top, right to left
    return 2.0 * W + H + (H - y)          # left, top to bottom


@njit(cache=True, parallel=True)
def _sample(n, W, H, mode, xin, yin, mu_in, ox_in, oy_in, seed, n_blocks):
    """Walk n particles and collect their exit states.

    Split into n_blocks parallel chunks, each with its own seed (numba's RNG
    is per-thread).  xin/yin are entry positions as a fraction of the cell;
    negative means "sample uniformly".
    """
    p = np.empty(n)
    dirs = np.empty((n, 3))
    ent = np.empty((n, 4))          # entry state: x0, y0, Omega_x, Omega_y
    s = np.empty(n)
    k = np.empty(n, dtype=np.int64)
    per_block = (n + n_blocks - 1) // n_blocks

    for b in prange(n_blocks):
        np.random.seed(seed + b)
        lo = b * per_block
        hi = min(n, lo + per_block)
        for i in range(lo, hi):

            # ---- choose the entry state ------------------------------
            if mode == ENTRY_COSINE:
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox, oy, oz = _cosine_hemisphere_px()

            elif mode == ENTRY_FIXED_MU:
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox = mu_in
                # spread the remaining direction budget over a random azimuth
                r = np.sqrt(max(0.0, 1.0 - ox * ox))
                phi = 2.0 * np.pi * np.random.random()
                oy = r * np.cos(phi)
                oz = r * np.sin(phi)

            elif mode == BORN_INSIDE:
                x = xin * W if xin >= 0.0 else W * np.random.random()
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox, oy, oz = _isotropic_direction()

            elif mode == ENTRY_FIXED_DIR:
                # what training data uses: the caller pins the full in-plane
                # direction, which is what the model conditions on
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox = ox_in
                oy = oy_in
                # Omega_z never affects the trajectory, but it IS part of
                # the recorded exit state, and an uncollided particle
                # carries its entry value straight through.  Both signs are
                # equally likely -- fixing it positive biases the data.
                rz = 1.0 - ox * ox - oy * oy
                oz = np.sqrt(rz) if rz > 0.0 else 0.0
                if np.random.random() < 0.5:
                    oz = -oz

            else:   # ENTRY_ISOTROPIC
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox = np.random.random()      # uniform in solid angle, not flux
                r = np.sqrt(max(0.0, 1.0 - ox * ox))
                phi = 2.0 * np.pi * np.random.random()
                oy = r * np.cos(phi)
                oz = r * np.sin(phi)

            # the entry state is the label the exit is conditioned on
            ent[i, 0] = x
            ent[i, 1] = y
            ent[i, 2] = ox
            ent[i, 3] = oy

            xe, ye, oxe, oye, oze, path_length, n_scatters = \
                _walk(x, y, ox, oy, oz, W, H)
            p[i] = _perimeter(xe, ye, W, H)
            dirs[i, 0] = oxe
            dirs[i, 1] = oye
            dirs[i, 2] = oze
            s[i] = path_length
            k[i] = n_scatters

    return p, dirs, s, k, ent


def sample_single_cell(n, W, H, mode="boundary", entry_pos=None,
                       entry_mu=None, entry_dir=None, entry_law="cosine",
                       seed=1, n_blocks=64):
    """Sample n transmissions through a W x H cell (optical units).

    mode="boundary" -- particles enter through the left face.
        entry_pos     entry height as a fraction of H; None samples it
                      uniformly.
        entry_dir     (Omega_x, Omega_y), pinning the full in-plane
                      direction.  This is what generative training data
                      needs, because it is what the model conditions on.
                      Must have Omega_x > 0 (it has to be going inwards)
                      and lie inside the unit disk.  Overrides entry_mu.
        entry_mu      pins only Omega_x; the azimuth is random.
        entry_law     used when neither is given: "cosine" for an isotropic
                      incident flux (the physical case), or "isotropic" for
                      uniform over the incoming solid angle.

    mode="internal" -- particles born inside the cell with an isotropic
        direction, which is what a volumetric source looks like.
        entry_pos = (fx, fy) as fractions; None samples uniformly.

    Returns a dict:
        p     (n,)    exit perimeter coordinate
        dir   (n, 3)  exit direction cosines
        s     (n,)    path length travelled inside the cell
        k     (n,)    number of scatters -- the cost MC pays, GMC avoids
        ent   (n, 4)  entry state (x0, y0, Omega_x, Omega_y), the label
        W, H          echoed back for convenience
    """
    ox_in = oy_in = 0.0

    if mode == "boundary":
        yin = -1.0 if entry_pos is None else float(entry_pos)
        if entry_dir is not None:
            ox_in, oy_in = float(entry_dir[0]), float(entry_dir[1])
            if ox_in <= 0.0:
                raise ValueError("entry_dir must have Omega_x > 0 to enter "
                                 f"the left face, got {ox_in}")
            if ox_in * ox_in + oy_in * oy_in > 1.0:
                raise ValueError("entry_dir must lie inside the unit disk, "
                                 f"got |Omega_xy|^2 = {ox_in**2 + oy_in**2}")
            m, xin, mu = ENTRY_FIXED_DIR, -1.0, 0.0
        elif entry_mu is not None:
            m, xin, mu = ENTRY_FIXED_MU, -1.0, float(entry_mu)
        elif entry_law == "cosine":
            m, xin, mu = ENTRY_COSINE, -1.0, 0.0
        else:
            m, xin, mu = ENTRY_ISOTROPIC, -1.0, 0.0

    elif mode == "internal":
        if entry_pos is None:
            xin = yin = -1.0
        else:
            xin, yin = float(entry_pos[0]), float(entry_pos[1])
        m, mu = BORN_INSIDE, 0.0

    else:
        raise ValueError(f"unknown mode {mode!r}")

    p, dirs, s, k, ent = _sample(int(n), float(W), float(H), m, xin, yin, mu,
                                 ox_in, oy_in, int(seed), int(n_blocks))
    return {"p": p, "dir": dirs, "s": s, "k": k, "ent": ent, "W": W, "H": H}


def time_single_cell(n, L, seed=1, n_blocks=8, repeats=1):
    """Wall time to push n particles through an L x L cell, and the mean
    scatter count.  Time grows with L -- that growth is what GMC flattens."""
    best = np.inf
    for r in range(repeats):
        t0 = time.perf_counter()
        _, _, _, k, _ = _sample(int(n), float(L), float(L), ENTRY_COSINE,
                                -1.0, -1.0, 0.0, 0.0, 0.0,
                                int(seed + 1000 * r), int(n_blocks))
        best = min(best, time.perf_counter() - t0)
    return best, float(k.mean())
