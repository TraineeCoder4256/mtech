"""Standard Monte Carlo particle transport in 2D Cartesian geometry.

Faithful reimplementation of the *standard MC* baseline of

    Farmer, Murray, Krotz, McClarren,
    "Generative Monte Carlo Sampling for Constant-Cost Particle Transport",
    arXiv:2512.13965v1 (2025)

covering everything except the neural-network (CFM) sampler:
  * steady-state, monoenergetic linear Boltzmann transport (Eq. 1)
  * implicit capture / continuous absorption (Eqs. 4-6)
  * track-length scalar-flux estimator (Eq. 7), per-source-particle norm
  * the lattice and linearized-hohlraum benchmarks of Fig. 3
  * the single-cell transmission sampler used for training data,
    Fig. 2b statistics and the Fig. 4b cost-scaling study
"""

from .transport import run_transport, make_uniform_grid
from .problems import lattice_problem, hohlraum_problem
from .singlecell import sample_single_cell, time_single_cell
