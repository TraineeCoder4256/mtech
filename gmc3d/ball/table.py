"""Backend 2 of 3: a lookup table of pre-computed walks.

Used by: core/transport.py (as `backend`); checks/ball_metrics.py.

The idea is the classic one (Fleck and Canfield's "random walk" method,
1984, used the same trick for radiation transport): since the walk inside a
ball depends on R alone, run it in advance for a grid of radii, keep the
results, and at run time draw a stored result instead of walking.

How it is stored.  N_GRID radii, evenly spaced in log R from r_min to
r_max, and for each one N_PER stored walks (mu, s).  Also the mean path
s_mean at each radius.

How a draw works, for a radius R between grid points R_k and R_k+1:

    1. pick R_k or R_k+1 at random, with the chance of R_k+1 equal to how far
       R is from R_k (in log R) -- so the draw is a blend of the two
    2. pick one of that radius's stored walks at random
    3. stretch its path s by  s_mean(R) / s_mean(R_k)  (s_mean interpolated
       in log-log), so the AVERAGE path is right at R itself, not just at
       the grid point.  The average path is what the flux depends on most.

The residual error is the blend: the exit-angle distribution at R is
replaced by a mix of its neighbours'.  On a fine grid that is a tiny,
second-order effect; checks/ball_metrics.py measures it.

Cost: a few random numbers and a memory read, independent of R -- the
benchmark any network has to beat (research item R2).  Memory: N_GRID x
N_PER x 2 numbers (25 MB at the defaults).  N_PER is large because the
table is a FIXED sample reused millions of times: its own sampling error
(about 1/sqrt(N_PER) of the spread) becomes a small systematic error.
"""

from pathlib import Path

import numpy as np
from numba import njit, prange

from core import rng
from . import walk

N_GRID, N_PER = 96, 16384
R_MIN, R_MAX = 1.0, 30.0


@njit(cache=True, parallel=True)
def _draw(R, state, idx, log_grid, mu_tab, s_tab, log_smean):
    n = R.size
    mu, s = np.empty(n), np.empty(n)
    g = log_grid.size
    per = mu_tab.shape[1]
    step = (log_grid[-1] - log_grid[0]) / (g - 1)
    for j in prange(n):
        i = idx[j]
        f = (np.log(R[j]) - log_grid[0]) / step
        k0 = min(max(int(np.floor(f)), 0), g - 2)
        frac = min(max(f - k0, 0.0), 1.0)
        # mean path at R itself: log s_mean interpolated between neighbours
        lsm = (1.0 - frac) * log_smean[k0] + frac * log_smean[k0 + 1]
        k = k0 + 1 if rng.rand(state, i) < frac else k0
        m = min(int(rng.rand(state, i) * per), per - 1)
        mu[j] = mu_tab[k, m]
        s[j] = s_tab[k, m] * np.exp(lsm - log_smean[k])
    return mu, s


class Table:
    name = "table"

    def __init__(self, log_grid, mu_tab, s_tab):
        self.log_grid = np.ascontiguousarray(log_grid)
        self.mu_tab = np.ascontiguousarray(mu_tab)
        self.s_tab = np.ascontiguousarray(s_tab)
        self.log_smean = np.log(self.s_tab.mean(axis=1))
        self.r_min = float(np.exp(log_grid[0]))
        self.r_max = float(np.exp(log_grid[-1]))
        self.draws = 0

    @classmethod
    def build(cls, r_min=R_MIN, r_max=R_MAX, n_grid=N_GRID, n_per=N_PER,
              seed=101):
        """Run n_grid x n_per exact walks (seconds at the defaults)."""
        log_grid = np.linspace(np.log(r_min), np.log(r_max), n_grid)
        mu = np.empty((n_grid, n_per))
        s = np.empty((n_grid, n_per))
        for k, lr in enumerate(log_grid):
            mu[k], s[k], _ = walk.sample(np.exp(lr), n_per, seed=seed,
                                         first=k * n_per)
        return cls(log_grid, mu, s)

    def sample(self, R, seeds, idx):
        self.draws += R.size
        return _draw(np.ascontiguousarray(R, np.float64), seeds, idx,
                     self.log_grid, self.mu_tab, self.s_tab, self.log_smean)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, log_grid=self.log_grid, mu=self.mu_tab, s=self.s_tab)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["log_grid"], d["mu"], d["s"])
