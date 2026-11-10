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

ANISOTROPIC SCATTERING (v2.1, `walk_aniso`).  With a scattering law the
first direction is no longer isotropic: it is the direction u0 the driver
has just sampled from the law (or the source's, at a birth), and every
scatter inside turns the particle by a cosine from the law.  The exit
point is then not uniform on the sphere.  The walk is unchanged by any
rotation about u0, and by the mirror through the plane holding u0 and the
exit normal, so four numbers describe its exit:

    cos_n  cosine between the exit point's outward normal n and u0
    mu     cosine between the exit direction and n (as above)
    phi    azimuth of the exit direction about n, from the half-plane of
           (n, u0) on u0's side, in [0, pi]; its sign is +/- with equal
           chance by the mirror symmetry, so the driver draws the sign
    s, k   as above

An isotropic law gives cos_n uniform on (-1, 1) and phi uniform on
[0, pi], with (mu, s) distributed as walk() (checked in tests).

Cost: about R^2 scatters per walk for big R (a diffusing particle needs
~R^2 steps to travel R), which is the whole case for replacing it.
"""

import numpy as np
from numba import njit, prange

from core import rng
from core import scattering as SC


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


@njit(cache=True, inline="always")
def _rotate(u, v, w, mu, phi):
    """core.transport.rotate (OpenMC's rotate_angle), kept here so that
    ball/ does not import the driver."""
    a = np.sqrt(max(0.0, 1.0 - mu * mu))
    b = np.sqrt(max(0.0, 1.0 - w * w))
    cp, sp = np.cos(phi), np.sin(phi)
    if b > 1.0e-10:
        return (mu * u + a * (u * w * cp - v * sp) / b,
                mu * v + a * (v * w * cp + u * sp) / b,
                mu * w - a * b * cp)
    b = np.sqrt(max(0.0, 1.0 - v * v))
    return (mu * u + a * (u * v * cp + w * sp) / b,
            mu * v - a * b * cp,
            mu * w + a * (v * w * cp - u * sp) / b)


@njit(cache=True, inline="always")
def exit_frame(n0, n1, n2, a0, a1, a2):
    """(e1, e2) at the exit normal n: e1 the unit vector along the part of
    a (the first direction) perpendicular to n, e2 = n x e1.  phi is
    measured from e1 towards e2 (walk_aniso and the driver share this)."""
    c = n0 * a0 + n1 * a1 + n2 * a2
    p0, p1, p2 = a0 - c * n0, a1 - c * n1, a2 - c * n2
    q = np.sqrt(p0 * p0 + p1 * p1 + p2 * p2)
    if q < 1.0e-12:                       # n along a: any perpendicular
        if abs(n0) < 0.9:
            p0, p1, p2 = 1.0 - n0 * n0, -n0 * n1, -n0 * n2
        else:
            p0, p1, p2 = -n1 * n0, 1.0 - n1 * n1, -n1 * n2
        q = np.sqrt(p0 * p0 + p1 * p1 + p2 * p2)
    p0, p1, p2 = p0 / q, p1 / q, p2 / q
    return (p0, p1, p2, n1 * p2 - n2 * p1, n2 * p0 - n0 * p2,
            n0 * p1 - n1 * p0)


@njit(cache=True)
def walk_aniso(R, ang, m, state, i):
    """One collided walk from the centre of a ball of radius R (mfp) whose
    material m scatters by the law ang[m] (core/scattering.py), first
    direction along +z.  Returns (cos_n, mu, phi, s, k); see the module
    docstring."""
    xi = rng.rand(state, i)
    l1 = -np.log(1.0 - xi * (-np.expm1(-R)))
    u, v, w = 0.0, 0.0, 1.0
    x, y, z = 0.0, 0.0, l1
    s = l1
    k = 0
    while True:
        k += 1
        mu_s = SC.sample_mu(ang, m, state, i)
        u, v, w = _rotate(u, v, w, mu_s, 2.0 * np.pi * rng.rand(state, i))
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
            n0, n1, n2 = x / R, y / R, z / R
            mu = n0 * u + n1 * v + n2 * w
            e10, e11, e12, e20, e21, e22 = exit_frame(n0, n1, n2,
                                                      0.0, 0.0, 1.0)
            phi = abs(np.arctan2(u * e20 + v * e21 + w * e22,
                                 u * e10 + v * e11 + w * e12))
            return (min(max(n2, -1.0), 1.0), min(max(mu, 0.0), 1.0), phi,
                    s, k)


@njit(cache=True, parallel=True)
def walk_many_aniso(R, mats, ang, state, idx):
    """walk_aniso() for each R[j] in material mats[j], stream idx[j]."""
    n = R.size
    cn, mu, phi, s = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    k = np.empty(n, np.int64)
    for j in prange(n):
        cn[j], mu[j], phi[j], s[j], k[j] = walk_aniso(R[j], ang, mats[j],
                                                      state, idx[j])
    return cn, mu, phi, s, k


def sample_aniso(R, ang, m, n, seed=1, first=0):
    """n anisotropic walks in a ball of radius R, material m of `ang`."""
    state = rng.streams(np.uint64(seed), first, n)
    return walk_many_aniso(np.full(n, float(R)), np.full(n, m, np.int64),
                           np.asarray(ang, float), state, np.arange(n))
