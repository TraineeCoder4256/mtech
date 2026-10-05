"""Backend 1 of 3: the exact walk, run live.

Used by: core/transport.py (as `backend`), checks/ (as the reference every
other backend is compared with, and to count the work balls could save).

It gives exactly the right answer, at exactly the cost the ball method is
meant to save (about R^2 scatters per ball).  Its job is to prove that the
DRIVER is right: a run with the oracle must match a run without balls,
within statistics, on any geometry.  Any later difference with a table or
network is then the backend's fault alone.

It also keeps a histogram of the balls it was asked for: how many, and how
many scatters they contained, per range of R.  checks/measure.py reads it
to say how much work a free backend could remove at each threshold R*.
"""

import numpy as np

from . import walk

HIST_EDGES = np.array([1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, np.inf])


class Oracle:
    name = "oracle"
    r_min = 0.0
    r_max = np.inf

    def __init__(self):
        self.walks = 0
        self.scatters = 0                       # work done inside balls
        self.hist_balls = np.zeros(HIST_EDGES.size - 1, np.int64)
        self.hist_scatters = np.zeros(HIST_EDGES.size - 1, np.int64)

    def sample(self, R, seeds, idx):
        """(mu, s) for each R, drawn from the waiting particles' streams."""
        R = np.ascontiguousarray(R)
        mu, s, k = walk.walk_many(R, seeds, idx)
        self.walks += R.size
        self.scatters += int(k.sum())
        b = np.clip(np.searchsorted(HIST_EDGES, R, side="right") - 1, 0,
                    HIST_EDGES.size - 2)
        self.hist_balls += np.bincount(b, minlength=self.hist_balls.size)
        self.hist_scatters += np.bincount(b, weights=k,
                                          minlength=self.hist_balls.size
                                          ).astype(np.int64)
        return mu, s
