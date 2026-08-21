"""Single-cell transmission: walk one particle through one cell, record how
it came out.

This is the physics the generative model is trying to learn to skip, and it
is the only Monte Carlo the training data ever sees.

THE EXPERIMENT
--------------
A particle enters a rectangular cell.  Inside, it scatters off the material
some number of times, each scatter throwing it in a completely new random
direction, until eventually a flight carries it across the boundary and out.
We record three things about the exit:

    p        WHERE it left      -- a single coordinate running around the
                                   perimeter (see PERIMETER COORDINATE below)
    Omega    WHICH WAY it was   -- direction cosines (Omega_x, Omega_y, Omega_z)
             going
    s        HOW FAR it went    -- total path length inside the cell,
                                   summed over every flight

and, as a diagnostic, k, the number of scatters it took.  k is the cost
standard Monte Carlo pays and the generative sampler avoids, so it is the
number the whole speed argument turns on.

UNITS
-----
Lengths are in mean free paths, which is the same as setting sigma_s = 1.
The distance to the next scatter is then just -ln(xi) for a uniform random
xi.  To convert back to centimetres, divide by the physical sigma_s.  This
is why one trained model covers every cell size and material: a cell is
fully described by its optical size W x H, not by its physical size and
cross section separately.

Absorption is deliberately NOT simulated here.  The cell is pure scattering,
and absorption is applied afterwards by attenuating the returned path length
with exp(-sigma_a * s).  Keeping absorption out of the walk is what makes
the walk depend on optical size alone.

PERIMETER COORDINATE
--------------------
The exit point is somewhere on the boundary, which is a 1-D curve, so it
needs only one number rather than two.  We unwrap the perimeter
anticlockwise from the bottom-left corner:

        p = 2W+H .... p = 2W+2H          going anticlockwise:
             +-----------------+           bottom  [0,      W)      x: 0 -> W
             |                 |           right   [W,      W+H)    y: 0 -> H
    p = W+H  |                 | p = W     top     [W+H,    2W+H)   x: W -> 0
             |                 |           left    [2W+H, 2W+2H)    y: H -> 0
             +-----------------+
        p = 0 ............ p = W

One coordinate instead of two, and it wraps cleanly at p = 2(W+H), which is
what lets the model encode it as a point on a circle.
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
    """A direction drawn uniformly over the full 3-D unit sphere.

    Sample the z-cosine uniformly in [-1, 1] and the azimuth uniformly in
    [0, 2pi); that combination is uniform on the sphere (Archimedes), where
    sampling two angles uniformly would not be.
    """
    oz = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - oz * oz))            # radius of the z-slice
    return r * np.cos(phi), r * np.sin(phi), oz


@njit(cache=True, inline="always")
def _cosine_hemisphere_px():
    """A direction into +x with density proportional to Omega_x.

    This is what an isotropic *flux* looks like crossing a surface: a
    particle at a grazing angle is less likely to cross than a
    perpendicular one, by exactly the cosine factor.  sqrt(xi) is the
    inverse-CDF of that cosine law.
    """
    ox = np.sqrt(np.random.random())
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - ox * ox))
    return ox, r * np.cos(phi), r * np.sin(phi)


@njit(cache=True, inline="always")
def _walk(x, y, ox, oy, oz, W, H):
    """Follow one particle from entry to exit.  This is the whole simulation.

    Each iteration is one flight.  We work out two competing distances --
    how far to the next scatter, and how far to the wall -- and whichever is
    shorter is what actually happens.

    Returns the exit state: (x, y, ox, oy, oz, path_length, n_scatters).
    """
    path_length = 0.0
    n_scatters = 0

    while True:
        # 1. How far until the next scattering event?  In a medium with
        #    sigma_s = 1 the free path is exponentially distributed, and
        #    -ln(uniform) is exactly an exponential draw.
        dist_to_scatter = -np.log(np.random.random())

        # 2. How far until the particle would hit a wall?  Check the two
        #    vertical faces and the two horizontal ones separately; a
        #    direction component of zero never reaches that pair of faces.
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

        # 3. Whichever comes first is what happens.
        if dist_to_scatter < dist_to_wall:
            # --- scatter: move to the collision point and pick a brand new
            #     direction.  Scattering is isotropic, so the particle
            #     completely forgets where it was heading.
            x += ox * dist_to_scatter
            y += oy * dist_to_scatter
            path_length += dist_to_scatter
            ox, oy, oz = _isotropic_direction()
            n_scatters += 1
        else:
            # --- escape: move to the wall and stop.  The direction is
            #     unchanged, so the exit direction is whatever the last
            #     flight was travelling in.
            x += ox * dist_to_wall
            y += oy * dist_to_wall
            path_length += dist_to_wall
            # Snap exactly onto the face that was crossed, so that floating
            # point never leaves the exit point a hair inside or outside.
            if hits_x_wall_first:
                x = W if ox > 0.0 else 0.0
            else:
                y = H if oy > 0.0 else 0.0
            return x, y, ox, oy, oz, path_length, n_scatters


@njit(cache=True, inline="always")
def _perimeter(x, y, W, H):
    """Exit point (x, y) -> perimeter coordinate p (see module docstring).

    Order matters: the corners belong to whichever face is tested first,
    which keeps p single-valued.
    """
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

    Work is split into n_blocks independent chunks running in parallel, each
    with its own RNG seed, because numba's random state is per-thread.  The
    seeding is deterministic, so a given (seed, n_blocks) always reproduces
    the same numbers.

    xin / yin are entry positions as a FRACTION of the cell (0 to 1), with a
    negative value meaning "sample it uniformly instead".
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
                # The mode the generative training data uses: the caller
                # pins the full in-plane direction, because (Omega_x,
                # Omega_y) is exactly what the model is conditioned on.
                x = 0.0
                y = yin * H if yin >= 0.0 else H * np.random.random()
                ox = ox_in
                oy = oy_in
                # Omega_z is fixed in magnitude by |Omega| = 1, and it is a
                # passenger for the trajectory -- _walk only ever advances
                # x and y.  But it IS part of the recorded exit state, and
                # an uncollided particle carries its entry Omega_z straight
                # through to the exit.  Both signs are equally likely, so
                # the sign has to be sampled; fixing it positive biases the
                # Omega_z distribution the model then learns.
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

            # remember the entry state: it is the label the exit state is
            # conditioned on when this becomes training data
            ent[i, 0] = x
            ent[i, 1] = y
            ent[i, 2] = ox
            ent[i, 3] = oy

            # ---- run the walk and record the exit --------------------
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
    """Wall time (s) to push n particles through an L x L cell, and the mean
    scatter count.

    This is the MC side of the cost comparison: the time grows with L
    because a thicker cell means more scatters per crossing, which is
    exactly the growth the generative sampler is meant to flatten.
    """
    best = np.inf
    for r in range(repeats):
        t0 = time.perf_counter()
        _, _, _, k, _ = _sample(int(n), float(L), float(L), ENTRY_COSINE,
                                -1.0, -1.0, 0.0, 0.0, 0.0,
                                int(seed + 1000 * r), int(n_blocks))
        best = min(best, time.perf_counter() - t0)
    return best, float(k.mean())
