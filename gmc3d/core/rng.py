"""Random numbers: one independent stream per particle, as OpenMC does it.

Used by: transport.py (every flight, scatter and source draw) and the ball
backends in ball/ (the exact walk and the table draw from the particle's
own stream).

Why per particle: the solver runs particles on several threads and pauses
them for batched ball steps.  If they shared one generator, the numbers a
particle got would depend on thread timing, and a rerun would not repeat.
With its own stream, particle k of a run always sees the same numbers, so
results are reproducible bit for bit (SRS S1-MC-4).

The generator is OpenMC's: a 63-bit linear congruential generator

    seed  <-  (G * seed + C)  mod 2^63,      random number = seed / 2^63

and particle k starts k * STRIDE steps along the sequence from the master
seed.  "Skipping ahead" n steps costs log2(n) operations (`skip_ahead`), so
any particle's start is found directly.  A particle that used more than
STRIDE numbers would run into the next particle's stream; OpenMC accepts
the same risk with the same stride.

State is a uint64 array, one entry per particle; `rand(state, i)` advances
entry i and returns a number in [0, 1).
"""

import numpy as np
from numba import njit

G = np.uint64(2806196910506780709)
C = np.uint64(1)
MASK = np.uint64((1 << 63) - 1)
NORM = 2.0 ** -63
STRIDE = 152917


@njit(cache=True, inline="always")
def rand(state, i):
    """Next number in [0, 1) from stream i (updates state[i])."""
    s = (G * state[i] + C) & MASK
    state[i] = s
    return s * NORM


@njit(cache=True)
def skip_ahead(n, seed):
    """The seed n steps after `seed`, in O(log n) steps (OpenMC's
    future_seed).  Uses that n steps of the generator are themselves one
    step of a generator with multiplier G^n and increment C(G^n-1)/(G-1)."""
    g, c = G, C
    g_new, c_new = np.uint64(1), np.uint64(0)
    n = np.uint64(n) & MASK
    while n > np.uint64(0):
        if n & np.uint64(1):
            g_new = g_new * g
            c_new = c_new * g + c
        c = c * (g + np.uint64(1))
        g = g * g
        n = n >> np.uint64(1)
    return (g_new * np.uint64(seed) + c_new) & MASK


@njit(cache=True)
def streams(master_seed, first, n):
    """Start states for particles first .. first+n-1."""
    out = np.empty(n, np.uint64)
    for k in range(n):
        out[k] = skip_ahead(np.uint64(first + k) * np.uint64(STRIDE),
                            master_seed)
    return out


@njit(cache=True, inline="always")
def isotropic(state, i):
    """A direction uniform on the unit sphere."""
    w = 2.0 * rand(state, i) - 1.0
    phi = 2.0 * np.pi * rand(state, i)
    r = np.sqrt(max(0.0, 1.0 - w * w))
    return r * np.cos(phi), r * np.sin(phi), w
