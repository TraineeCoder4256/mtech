"""Generative Monte Carlo: a flow-matching sampler that replaces the
in-cell random walk (arXiv:2512.13965v1, boundary model).

Given a particle entering a square scattering cell, sample where it leaves,
which way it is going, and how far it travelled -- in one shot, at a fixed
cost, however optically thick the cell is.

    data.py       dataset -> encoded, normalised tensors
    model.py      the velocity field, the CFM loss, weight averaging
    sampler.py    cell geometry, ODE solve, exit-state decoding
    transport.py  chain the sampler across a mesh to solve a real problem
"""

from .data import load_dataset, encode_conditions, encode_targets, Normalizer
from .data import save_normalizers, load_normalizers
from .model import VelocityField, cfm_loss, EMA
from .sampler import (GMCBoundarySampler, integrate, perimeter_decode,
                      chord_length, NFE_PER_STEP)
from .transport import run_gmc_transport, macro_problem, coarsen, analog_birth
