"""Monte Carlo transport: the baseline the generative sampler has to beat.

Two solvers live here.

  sample_cell   one particle through ONE rectangular cell, tracked to its
                exit.  This makes the training data, and it is exactly the
                work the generative model learns to replace.

  solve         a full problem on a mesh, used as the accuracy reference
                and the speed baseline.

Units: inside a cell, lengths are in mean free paths (sigma_s = 1), so the
distance to the next scatter is just -ln(uniform).  A cell is therefore
described by its OPTICAL size W x H alone -- not by its physical size and
cross section separately -- which is why one trained model covers every
cell size and material.

Cells are pure scattering.  Absorption is applied by attenuating the
returned path length, which keeps the walk dependent on optical size only.

Perimeter coordinate p, anticlockwise from the bottom-left corner of a
W x H cell (total perimeter 2(W+H)):

        p=2W+H ......... p=2W+2H      bottom  [0,      W)     x: 0 -> W
             +-----------------+      right   [W,      W+H)   y: 0 -> H
      p=W+H  |                 | p=W  top     [W+H,    2W+H)  x: W -> 0
             +-----------------+      left    [2W+H, 2W+2H)   y: H -> 0
        p=0 ............... p=W

One number instead of two, and it wraps -- which is what lets the model
encode it as a point on a circle.
"""

import numpy as np
from numba import njit, prange

ENTER_LEFT = 0    # enters the left face with a given direction
BORN_INSIDE = 1   # born in the cell, isotropic (a volumetric source)
FAR = 1.0e300     # stand-in for "never reaches that pair of walls"
WEIGHT_CUTOFF = 1.0e-12


@njit(cache=True, inline="always")
def _isotropic():
    """Uniform on the 3-D unit sphere: uniform z-cosine, uniform azimuth."""
    oz = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(max(0.0, 1.0 - oz * oz))
    return r * np.cos(phi), r * np.sin(phi), oz


# ----------------------------------------------------------- single cell
@njit(cache=True, inline="always")
def _walk(x, y, ox, oy, oz, W, H):
    """Follow one particle to its exit.  This is the whole simulation.

    Each iteration is one flight: how far to the next scatter, how far to
    the wall, and whichever is shorter is what happens.
    """
    path, n_scatter = 0.0, 0
    while True:
        d_scatter = -np.log(np.random.random())     # sigma_s = 1

        if ox > 0.0:
            dx = (W - x) / ox
        elif ox < 0.0:
            dx = -x / ox
        else:
            dx = FAR
        if oy > 0.0:
            dy = (H - y) / oy
        elif oy < 0.0:
            dy = -y / oy
        else:
            dy = FAR
        x_wall_first = dx < dy
        d_wall = dx if x_wall_first else dy

        if d_scatter < d_wall:
            # scatter: move there and forget the old direction entirely
            x += ox * d_scatter
            y += oy * d_scatter
            path += d_scatter
            ox, oy, oz = _isotropic()
            n_scatter += 1
        else:
            # escape: move to the wall, keeping the current direction
            x += ox * d_wall
            y += oy * d_wall
            path += d_wall
            # snap onto the face so rounding cannot leave the exit point a
            # hair inside or outside
            if x_wall_first:
                x = W if ox > 0.0 else 0.0
            else:
                y = H if oy > 0.0 else 0.0
            return x, y, ox, oy, oz, path, n_scatter


@njit(cache=True, inline="always")
def _perimeter(x, y, W, H):
    """Exit point -> perimeter coordinate.  A corner belongs to whichever
    face is tested first, which keeps p single-valued."""
    if y <= 0.0:
        return x
    if x >= W:
        return W + y
    if y >= H:
        return W + H + (W - x)
    return 2.0 * W + H + (H - y)


@njit(cache=True, parallel=True)
def _sample_cell(n, W, H, mode, xi, ox_in, oy_in, seed, n_blocks):
    p = np.empty(n)
    dirs = np.empty((n, 3))
    s = np.empty(n)
    k = np.empty(n, np.int64)
    per = (n + n_blocks - 1) // n_blocks

    for b in prange(n_blocks):
        np.random.seed(seed + b)            # numba's RNG is per thread
        for i in range(b * per, min(n, b * per + per)):
            if mode == ENTER_LEFT:
                x, y = 0.0, xi * H
                ox, oy = ox_in, oy_in
                # Omega_z never affects the trajectory, but it IS part of
                # the exit state, and an uncollided particle carries its
                # entry value straight through.  Both signs are equally
                # likely -- fixing it positive would bias the data.
                rz = 1.0 - ox * ox - oy * oy
                oz = np.sqrt(rz) if rz > 0.0 else 0.0
                if np.random.random() < 0.5:
                    oz = -oz
            else:
                x = W * np.random.random()
                y = H * np.random.random()
                ox, oy, oz = _isotropic()

            xe, ye, oxe, oye, oze, path, ns = _walk(x, y, ox, oy, oz, W, H)
            p[i] = _perimeter(xe, ye, W, H)
            dirs[i, 0], dirs[i, 1], dirs[i, 2] = oxe, oye, oze
            s[i] = path
            k[i] = ns
    return p, dirs, s, k


def sample_cell(n, W, H, xi=0.5, direction=(1.0, 0.0), inside=False,
                seed=1, n_blocks=8):
    """Push n particles through one W x H cell (optical units).

    inside=False: they enter the left face at height xi*H (xi in [0,1])
                  travelling along `direction` = (Omega_x, Omega_y), which
                  must point inwards and lie in the unit disk.
    inside=True:  they are born uniformly inside, isotropic.

    Returns p (exit perimeter coord), dir (n,3), s (path length),
    k (number of scatters -- the cost MC pays and GMC avoids).
    """
    if inside:
        mode, ox, oy = BORN_INSIDE, 0.0, 0.0
    else:
        mode, (ox, oy) = ENTER_LEFT, (float(direction[0]), float(direction[1]))
        if ox <= 0.0 or ox * ox + oy * oy > 1.0:
            raise ValueError(f"direction {(ox, oy)} must enter the left face "
                             f"and lie in the unit disk")
    p, dirs, s, k = _sample_cell(int(n), float(W), float(H), mode, float(xi),
                                 ox, oy, int(seed), int(n_blocks))
    return {"p": p, "dir": dirs, "s": s, "k": k}


# ------------------------------------------------------------ full domain
@njit(cache=True)
def _history(x, y, ox, oy, sig_s, sig_a, dx, dy, nx, ny, tally, stats):
    """One history on the mesh, accumulating track-length flux.

    Implicit capture: the flight length is drawn from sigma_s alone and the
    weight is attenuated by exp(-sigma_a * step) along each segment, rather
    than killing the particle on absorption.  stats counts work done and
    takes no part in the physics.
    """
    w = 1.0
    ix, iy = int(x / dx), int(y / dy)
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        return
    while True:
        ss, sa = sig_s[iy, ix], sig_a[iy, ix]
        step_scatter = -np.log(np.random.random()) / ss if ss > 0.0 else FAR

        if ox > 0.0:
            tx = ((ix + 1) * dx - x) / ox
        elif ox < 0.0:
            tx = (ix * dx - x) / ox
        else:
            tx = FAR
        if oy > 0.0:
            ty = ((iy + 1) * dy - y) / oy
        elif oy < 0.0:
            ty = (iy * dy - y) / oy
        else:
            ty = FAR
        x_first = tx < ty
        step_wall = tx if x_first else ty

        scatter = step_scatter < step_wall
        step = step_scatter if scatter else step_wall

        # track-length estimator, with absorption along the segment
        if sa > 0.0:
            att = np.exp(-sa * step)
            tally[iy, ix] += w * (1.0 - att) / sa
            w *= att
        else:
            tally[iy, ix] += w * step

        x += ox * step
        y += oy * step
        if scatter:
            stats[0] += 1.0
            ox, oy, _ = _isotropic()
        else:
            stats[1] += 1.0
            if x_first:
                ix += 1 if ox > 0.0 else -1
                x = (ix if ox > 0.0 else ix + 1) * dx
                if ix < 0 or ix >= nx:
                    return
            else:
                iy += 1 if oy > 0.0 else -1
                y = (iy if oy > 0.0 else iy + 1) * dy
                if iy < 0 or iy >= ny:
                    return
        if w < WEIGHT_CUTOFF:
            return


@njit(cache=True, parallel=True)
def _solve(n, sig_s, sig_a, dx, dy, nx, ny, box, seed, n_blocks):
    tallies = np.zeros((n_blocks, ny, nx))
    stats = np.zeros((n_blocks, 2))
    per = (n + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        np.random.seed(seed + b)
        for _ in range(b * per, min(n, b * per + per)):
            x = box[0] + (box[2] - box[0]) * np.random.random()
            y = box[1] + (box[3] - box[1]) * np.random.random()
            ox, oy, _ = _isotropic()
            _history(x, y, ox, oy, sig_s, sig_a, dx, dy, nx, ny,
                     tallies[b], stats[b])
    out = np.zeros((ny, nx))
    tot = np.zeros(2)
    for b in range(n_blocks):
        out += tallies[b]
        tot += stats[b]
    return out, tot


def solve(problem, n, seed=1, n_blocks=64):
    """Solve a problem with standard MC.  Returns (flux, work counts).

    Flux is the track-length estimator per source particle.  The counts are
    what the cost comparison needs: a scattering event is the MC unit of
    work, so that is the number the sampler has to beat.
    """
    nx, ny = problem["n"], problem["n"]
    dx = dy = problem["L"] / nx
    tally, st = _solve(int(n), problem["sig_s"], problem["sig_a"], dx, dy,
                       nx, ny, np.asarray(problem["source"], np.float64),
                       int(seed), int(n_blocks))
    return tally / (dx * dy * n), {
        "particles": int(n), "scatters": float(st[0]),
        "mesh_crossings": float(st[1]),
        "scatters_per_particle": float(st[0]) / n}


# -------------------------------------------------------- test geometry
ABSORBERS = [(1, 1), (3, 1), (5, 1), (2, 2), (4, 2), (1, 3), (5, 3),
             (2, 4), (4, 4), (1, 5), (5, 5)]


def lattice(scale=1.0, n=112):
    """7x7 cm checkerboard lattice, isotropic source in the middle cell.

    This is the END-TO-END TEST ONLY -- the model is trained on general
    rectangular cells and never sees this geometry.  `scale` multiplies
    every cross section, making cells optically thicker without changing
    the layout, which is how the speed crossover is found.
    """
    sig_s = np.full((n, n), 1.0 * scale)
    sig_a = np.zeros((n, n))
    per_cm = n // 7
    for (x0, y0) in ABSORBERS:
        sl = (slice(y0 * per_cm, (y0 + 1) * per_cm),
              slice(x0 * per_cm, (x0 + 1) * per_cm))
        sig_s[sl] = 0.5 * scale
        sig_a[sl] = 9.5 * scale
    return {"L": 7.0, "n": n, "pitch": 1.0, "per_cm": per_cm,
            "sig_s": sig_s, "sig_a": sig_a,
            "source": (3.0, 3.0, 4.0, 4.0), "source_cell": (3, 3)}
