"""Cell geometry shared by the encoder and the sampler.

These three functions are the only place the square/rectangular cell shape
is written down.  They live in their own module because both gmc.data (which
encodes training targets) and gmc.sampler (which decodes model output) need
them, and importing one from the other would be circular.

Convention (the canonical reference frame of docs Sec 1.2): the particle
always enters the LEFT face at (0, xi*H) travelling with Omega_x > 0; the
perimeter coordinate p runs anticlockwise from the origin, bottom -> right
-> top -> left, with total length 2(W + H).
"""

import numpy as np


def chord_length(W, H, xi, oxi, oyi):
    """Straight-line distance (mfp) from entry (0, xi*H) along (oxi, oyi)
    to the cell boundary.  oxi > 0 by construction (entering left face)."""
    y = xi * H
    tx = W / oxi
    with np.errstate(divide="ignore"):
        ty = np.where(oyi > 0, (H - y) / np.where(oyi > 0, oyi, 1.0),
                      np.where(oyi < 0, -y / np.where(oyi < 0, oyi, 1.0),
                               np.inf))
    return np.minimum(tx, ty)


def perimeter_decode(p, W, H):
    """Perimeter coordinate -> (x, y) on the cell boundary + face index
    (0 bottom, 1 right, 2 top, 3 left)."""
    p, W, H = np.broadcast_arrays(p, W, H)
    p = np.mod(p, 2.0 * (W + H))
    x = np.empty_like(p)
    y = np.empty_like(p)
    face = np.empty(p.shape, dtype=np.int8)
    b0, b1, b2 = W, W + H, 2 * W + H
    m = p < b0
    x[m], y[m], face[m] = p[m], 0.0, 0
    m = (p >= b0) & (p < b1)
    x[m], y[m], face[m] = W[m], p[m] - b0[m], 1
    m = (p >= b1) & (p < b2)
    x[m], y[m], face[m] = W[m] - (p[m] - b1[m]), H[m], 2
    m = p >= b2
    x[m], y[m], face[m] = 0.0, H[m] - (p[m] - b2[m]), 3
    return x, y, face


def s_min_of(p, W, H, xi):
    """Shortest path length that can connect the entry point to the exit.

    A particle entering at (0, xi*H) and leaving at the perimeter coordinate
    p cannot have travelled less than the straight line between the two, so
    this is a hard lower bound on the path length s given the exit position.
    It is the quantity the ``detour`` path-length parameterisation divides
    out, which turns the bound from something the network has to learn into
    something the decoder cannot violate.
    """
    ex, ey, _ = perimeter_decode(p, W, H)
    _, H_, xi_ = np.broadcast_arrays(np.asarray(p, float), H, xi)
    return np.hypot(ex, ey - xi_ * H_)
