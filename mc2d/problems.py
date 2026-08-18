"""The two benchmark problems of arXiv:2512.13965v1, Fig. 3.

All cross sections in cm^-1, all lengths in cm, exactly as printed in the
paper ("Benchmark Performance" section).  Region geometry was measured from
the vector graphics of Fig. 3a / Fig. 3d.
"""

import numpy as np


def _paint(sig_s, sig_a, edges_x, edges_y, box, ss, sa):
    """Assign (ss, sa) to every cell whose center lies inside box=(x0,y0,x1,y1)."""
    xc = 0.5 * (edges_x[:-1] + edges_x[1:])
    yc = 0.5 * (edges_y[:-1] + edges_y[1:])
    x0, y0, x1, y1 = box
    mx = (xc > x0) & (xc < x1)
    my = (yc > y0) & (yc < y1)
    sig_s[np.ix_(my, mx)] = ss
    sig_a[np.ix_(my, mx)] = sa


# 1x1 cm absorber blocks of the lattice, lower-left corners (Fig. 3a).
# Checkerboard inside [1,6]^2 minus the "chimney" cell at (3,5).
LATTICE_ABSORBERS = [(1, 1), (3, 1), (5, 1),
                     (2, 2), (4, 2),
                     (1, 3), (5, 3),
                     (2, 4), (4, 4),
                     (1, 5), (5, 5)]


def lattice_problem(n=112):
    """7x7 cm heterogeneous lattice, 112x112 mesh.

    Background: sigma_a = 0, sigma_s = 1 (sigma_t = 1).
    Absorbers:  sigma_a = 9.5, sigma_s = 0.5 (sigma_t = 10).
    Isotropic volumetric source in the central 1x1 cm subdivision [3,4]^2.
    """
    Lx = Ly = 7.0
    ex = np.linspace(0.0, Lx, n + 1)
    ey = np.linspace(0.0, Ly, n + 1)
    sig_s = np.full((n, n), 1.0)
    sig_a = np.zeros((n, n))
    for (x0, y0) in LATTICE_ABSORBERS:
        _paint(sig_s, sig_a, ex, ey, (x0, y0, x0 + 1, y0 + 1), 0.5, 9.5)
    return {
        "name": "lattice", "Lx": Lx, "Ly": Ly, "nx": n, "ny": n,
        "sig_s": sig_s, "sig_a": sig_a,
        "source": {"kind": "volumetric", "box": (3.0, 3.0, 4.0, 4.0)},
        # Lineout locations: the white marker lines in Fig. 3b.  (The Fig. 3
        # caption says "through the source center", but the plotted Fig. 3c
        # curves quantitatively match the marker lines y=5.5 / x=1.5, not
        # cuts through (3.5, 3.5).)
        "lineout_y": 5.5, "lineout_x": 1.5,
    }


def hohlraum_problem(n=112):
    """1.3x1.3 cm linearized hohlraum, 112x112 mesh, left boundary source.

    Materials (cm^-1), as printed in the paper:
      black walls   sigma_a = 50, sigma_s = 50   (top/bottom/right, 0.05 thick)
      red stripe    sigma_a = 5,  sigma_s = 95   x in [0,0.05],  y in [0.25,1.05]
      green frame   sigma_a = 10, sigma_s = 90   x in [0.45,0.85], y in [0.25,1.05]
      blue absorber sigma_a = 95, sigma_s = 5    x in [0.50,0.85], y in [0.30,1.00]
      white         sigma_a = 0,  sigma_s = 5    everywhere else
    Region extents measured from Fig. 3d (ticks at x = 0.45, 0.85 and
    y = 0.25, 1.05; walls 0.05 cm thick; blue block inset 0.05 cm into the
    green frame on its left/bottom/top sides, flush with it on the right).
    """
    L = 1.3
    ex = np.linspace(0.0, L, n + 1)
    ey = np.linspace(0.0, L, n + 1)
    sig_s = np.full((n, n), 5.0)   # white near-vacuum background
    sig_a = np.zeros((n, n))
    t = 0.05
    _paint(sig_s, sig_a, ex, ey, (0.0, 0.0, L, t), 50.0, 50.0)        # bottom wall
    _paint(sig_s, sig_a, ex, ey, (0.0, L - t, L, L), 50.0, 50.0)      # top wall
    _paint(sig_s, sig_a, ex, ey, (L - t, 0.0, L, L), 50.0, 50.0)      # right wall
    _paint(sig_s, sig_a, ex, ey, (0.0, 0.25, 0.05, 1.05), 95.0, 5.0)  # red stripe
    _paint(sig_s, sig_a, ex, ey, (0.45, 0.25, 0.85, 1.05), 90.0, 10.0)  # green
    _paint(sig_s, sig_a, ex, ey, (0.50, 0.30, 0.85, 1.00), 5.0, 95.0)   # blue
    return {
        "name": "hohlraum", "Lx": L, "Ly": L, "nx": n, "ny": n,
        "sig_s": sig_s, "sig_a": sig_a,
        # boundary source: left wall x=0, full height, isotropic incident
        # angular flux (cosine-law sampling)
        "source": {"kind": "boundary_left", "box": (0.0, 0.0, 0.0, L)},
        # lineout locations measured from the white markers in Fig. 3e:
        "lineout_y": 1.115, "lineout_x": 0.66,
    }
