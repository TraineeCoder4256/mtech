"""The oracle: a "model" with zero model error.  The ACCURACY goalpost.

Instead of a network, sample() runs the real Monte Carlo walk from mc.py for
each particle and hands back the exact exit state.  It goes through the same
CellSampler as every learned model -- the same encoding into six numbers,
the same float32 standardisation, the same decode and the same clamp -- and
through the same transport loop.

So whatever error the oracle shows against the reference is error that the
MODEL did not cause: macro-cell tallies, the rotation onto the left face, the
float32 round trip, the clamp.  A learned model's error above the oracle's
is the model's fault; error at the oracle's level is not.  Every comparison
table gets this row, so "is the model the problem?" is answered by
subtraction rather than argued.

It must reproduce the COLLIDED distribution only, because CellSampler has
already taken the uncollided particles analytically.  So each walk that
happens to cross without scattering is thrown away and walked again
(rejection).  That is exactly the distribution the learned models are
trained on: data.py drops the same k == 0 rows.

Its cost is real Monte Carlo cost and says nothing about speed; nfe is 0.
"""
import numpy as np
import torch
from numba import njit, prange

from data import encode_targets
from generators.base import Generator
from mc import _walk, _perimeter


@njit(cache=True, parallel=True)
def _collided_walks(W, H, xi, ox_in, oy_in, seed, n_blocks):
    n = W.size
    p, s = np.empty(n), np.empty(n)
    dirs = np.empty((n, 3))
    tries = np.zeros(n, np.int64)
    per = (n + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        np.random.seed(seed + b)            # numba's RNG is per thread
        for i in range(b * per, min(n, b * per + per)):
            ox, oy = ox_in[i], oy_in[i]
            rz = 1.0 - ox * ox - oy * oy
            oz_mag = np.sqrt(rz) if rz > 0.0 else 0.0
            while True:
                # entry state exactly as mc.sample_cell builds it: Omega_z
                # takes either sign with equal probability
                oz = -oz_mag if np.random.random() < 0.5 else oz_mag
                xe, ye, oxe, oye, oze, path, ns = _walk(
                    0.0, xi[i] * H[i], ox, oy, oz, W[i], H[i])
                tries[i] += 1
                if ns > 0:
                    break
            p[i] = _perimeter(xe, ye, W[i], H[i])
            dirs[i, 0], dirs[i, 1], dirs[i, 2] = oxe, oye, oze
            s[i] = path
    return p, dirs, s, tries


class Oracle(Generator):
    name = "oracle"
    nfe = 0
    trainable = False

    def __init__(self, ynorm, n_blocks=8):
        self.ynorm, self.n_blocks = ynorm, n_blocks
        self.walks = 0                      # walks run, rejections included

    def sample(self, c, generator, cond=None):
        if cond is None:
            raise ValueError("the oracle needs the physical conditions")
        W, H = (np.ascontiguousarray(cond[k], np.float64) for k in ("W", "H"))
        xi, ox, oy = (np.ascontiguousarray(cond[k], np.float64)
                      for k in ("xi", "ox_in", "oy_in"))
        seed = int(torch.randint(1, 2**31 - 1 - self.n_blocks, (1,),
                                 generator=generator))
        p, d, s, tries = _collided_walks(W, H, xi, ox, oy, seed, self.n_blocks)
        self.walks += int(tries.sum())
        y = encode_targets(p, W, H, d[:, 0], d[:, 1], d[:, 2], s)
        return self.ynorm.transform(y)
