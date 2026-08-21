"""Generative Monte Carlo: a conditional flow-matching boundary model.

Reproduces the *boundary model* of arXiv:2512.13965v1 -- a generative
sampler that replaces the in-cell random walk.  Given a particle's entry
state into a square pure-scattering cell it samples the exit state

    y = (p_exit, Omega_exit, s)          in one shot,

conditioned on

    c = (W, H, xi_in, Omega_in).

Monte Carlo cost per cell grows with optical thickness, because a thicker
cell means more scattering events.  This costs a fixed number of network
evaluations no matter how thick the cell is.  That is the whole idea, and
scripts/evaluate.py measures whether it pays off.

The paper's architecture is in non-public supplementary material; every
concrete choice here -- network shape, encodings, ODE solver -- is ours.

    data.py      dataset -> encoded, normalised training tensors
    model.py     the velocity field v(x, t, c)
    cfm.py       the conditional flow-matching loss + EMA
    geometry.py  cell geometry shared by the encoder and the decoder
    sampler.py   ODE integration, decoding, analytic uncollided branch
    transport.py chains the sampler across a mesh to solve a real problem
    device.py    cpu / cuda / mps selection
"""

from .data import load_dataset, encode_conditions, encode_targets, Normalizer
from .model import VelocityField
from .cfm import cfm_loss, EMA
from .geometry import perimeter_decode, chord_length
from .sampler import GMCBoundarySampler, integrate, NFE_PER_STEP
from .transport import run_gmc_transport, macro_problem, coarsen
from .device import pick_device, describe
