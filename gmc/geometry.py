"""Cell geometry shared by the encoder and the sampler.

These two functions are the only place the cell shape is written down.

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
