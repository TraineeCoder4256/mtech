"""Generative Monte Carlo: conditional flow-matching boundary model.

Implements the *boundary model* of arXiv:2512.13965v1 -- a conditional
generative sampler that replaces the in-cell random walk: given a particle's
entry state into a rectangular pure-scattering cell it samples the exit state

    y = (p_exit, Omega_exit, s)            (Eq. 11)

conditioned on

    c = (W~, H~, xi_in, Omega_in)          (Eq. 10)

The paper's architecture lives in non-public supplementary material; every
concrete choice here (network shape, encodings, ODE solver) is ours and is
documented in docs/boundary_model_training.md and the module docstrings.

Pipeline:
    data.py     dataset -> encoded/normalised training tensors (leak-free split)
    model.py    the velocity field v_theta(x, t, c)
    cfm.py      conditional flow-matching objective + EMA
    sampler.py  ODE integration, output decoding, analytic uncollided branch
"""

from .data import load_boundary_dataset, encode_conditions, encode_targets, Normalizer
from .model import VelocityField
from .cfm import cfm_loss, EMA
from .sampler import GMCBoundarySampler, chord_length
from .geometry import perimeter_decode, s_min_of
from .device import pick_device, describe
from .transport import (run_gmc_transport, macro_problem,
                        coarsen)
