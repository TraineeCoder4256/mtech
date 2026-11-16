"""v2.2.1: the exact anisotropic walk returns on an ISO law row (it used
to loop for ever) and gives the isotropic walk's statistics there."""
import numpy as np

from ball import laws as LW, walk
from core import rng, scattering as SC


def test_first_scatter_on_iso_row_returns():
    ang = SC.pack(list(LW.held_out().values())[:2] + [(SC.ISO, None)])
    R = np.linspace(1.0, 20.0, 2500)
    mats = np.arange(R.size) % 3
    st = rng.streams(np.uint64(5), 0, R.size)
    ex, cn, mu, phi, s = walk.first_scatter_many_aniso(
        R, mats, ang, st, np.arange(R.size))
    assert np.all(np.isfinite(s[ex]))
