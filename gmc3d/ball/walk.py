"""The exact walk inside a ball: what every backend imitates.

Used by: oracle.py (runs it live), data.py (records it as training data),
table.py (pre-computes it on a grid of radii).

The setting, in mean-free-path units (sigma_s = 1, no absorption -- the
driver applies absorption afterwards, see core/transport.py):

    a particle starts at the centre of a ball of radius R, just scattered,
    so its first direction is isotropic; it is GIVEN that it scatters at
    least once before reaching the edge (the driver handles the
    straight-through case itself).

Output for one walk:

    mu    cosine between the exit direction and the outward normal at the
          exit point (0 < mu <= 1: it is leaving)
    s     total path length inside the ball, in mean free paths (s > R)
    k     number of scatters inside (the work a backend saves)

Nothing else is needed: the exit point is uniform on the sphere and the
azimuth of the exit direction is uniform, by symmetry, so the driver draws
those itself.

Cost: about R^2 scatters per walk for big R (a diffusing particle needs
~R^2 steps to travel R), which is the whole case for replacing it.
"""

import numpy as np
from numba import njit, prange

from core import rng


@njit(cache=True)
def walk(R, state, i):
    """One collided walk from the centre of a ball of radius R (mfp),
    using random stream i of `state`.  Returns (mu, s, k)."""
    # first flight: exponential, conditioned to end inside the ball
    xi = rng.rand(state, i)
    l1 = -np.log(1.0 - xi * (-np.expm1(-R)))
    u, v, w = rng.isotropic(state, i)
    x, y, z = l1 * u, l1 * v, l1 * w
    s = l1
    k = 0
    while True:
        k += 1
        u, v, w = rng.isotropic(state, i)       # scatter
        # distance to the sphere along (u, v, w): t^2 + 2 b t + c = 0
        b = x * u + y * v + z * w
        c = x * x + y * y + z * z - R * R
        t_exit = -b + np.sqrt(max(b * b - c, 0.0))
        l = -np.log(1.0 - rng.rand(state, i))
        if l < t_exit:
            x, y, z = x + l * u, y + l * v, z + l * w
            s += l
        else:
            x, y, z = x + t_exit * u, y + t_exit * v, z + t_exit * w
            s += t_exit
            mu = (x * u + y * v + z * w) / R
            return min(max(mu, 0.0), 1.0), s, k


@njit(cache=True, parallel=True)
def walk_many(R, state, idx):
    """walk() for each R[j], using random stream idx[j] of `state`."""
    n = R.size
    mu, s = np.empty(n), np.empty(n)
    k = np.empty(n, np.int64)
    for j in prange(n):
        mu[j], s[j], k[j] = walk(R[j], state, idx[j])
    return mu, s, k


def sample(R, n, seed=1, first=0):
    """n walks in a ball of radius R (one stream each, starting at
    particle number `first` of the master seed).  For data and tables."""
    state = rng.streams(np.uint64(seed), first, n)
    return walk_many(np.full(n, float(R)), state, np.arange(n))
