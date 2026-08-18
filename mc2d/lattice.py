"""Isolated, size-parameterised lattice geometry.

Everything that defines the lattice benchmark of arXiv:2512.13965v1 Fig. 3a
lives here: how the absorber blocks are placed, how they are painted onto a
mesh, and how the geometry is drawn.  ``problems.lattice_problem`` remains
the frozen, hard-coded paper case; this module reproduces it exactly at the
default parameters and generalises it to other sizes.

Structure of the paper lattice
------------------------------
The 7x7 cm domain is a K=5 lattice of 1 cm cells on [1,6]^2 surrounded by a
1 cm background border.  Absorbers occupy the even-parity cells of that
interior, i.e. ``(i + j) even`` with 1-based cell indices, *except*:

  * the centre cell, which carries the volumetric source, and
  * the cells directly above the centre, which are cleared to leave a
    "chimney": an unobstructed streaming channel from the source to the top
    boundary.

For K=5 the even-parity set has 13 cells; removing the centre (3,3) and the
chimney block (3,5) leaves the 11 blocks listed in ``problems.py``.  Note
that (3,4) is odd-parity and so was never an absorber, which is why the whole
column above the source is void.

Sizes
-----
``K`` is the number of interior lattice cells per side and must be odd so a
unique centre cell exists.  ``pitch`` is the physical size of one lattice
cell and ``border`` the background margin, so the domain is
``K * pitch + 2 * border`` on a side.  ``cells_per_pitch`` fixes the mesh
resolution; the paper's 112x112 mesh is 16 cells per 1 cm pitch.
"""

import numpy as np

from .problems import _paint

# Cross sections of the paper lattice (cm^-1).
BACKGROUND = {"sig_s": 1.0, "sig_a": 0.0}    # sigma_t = 1
ABSORBER = {"sig_s": 0.5, "sig_a": 9.5}      # sigma_t = 10


def absorber_cells(k=5, chimney=True):
    """1-based (i, j) lattice-cell indices of the absorber blocks.

    Even-parity checkerboard on the K x K interior, minus the centre cell
    (the source) and, when ``chimney`` is set, minus every cell above the
    centre in the centre column.
    """
    if k % 2 == 0:
        raise ValueError(f"k must be odd so the lattice has a centre cell, got {k}")
    c = (k + 1) // 2
    cells = {(i, j) for i in range(1, k + 1) for j in range(1, k + 1)
             if (i + j) % 2 == 0}
    cells.discard((c, c))
    if chimney:
        cells -= {(c, j) for j in range(c, k + 1)}
    return sorted(cells)


def lattice_spec(k=5, pitch=1.0, border=1.0, chimney=True):
    """Geometry of the lattice in physical (cm) units, independent of mesh.

    Returns the domain size, the centre/source box and the absorber boxes as
    ``(x0, y0, x1, y1)`` tuples, so the same description drives both the mesh
    painting and the plotting.
    """
    c = (k + 1) // 2
    L = k * pitch + 2.0 * border

    def box(i, j):
        x0 = border + (i - 1) * pitch
        y0 = border + (j - 1) * pitch
        return (x0, y0, x0 + pitch, y0 + pitch)

    return {
        "k": k, "pitch": pitch, "border": border, "chimney": chimney,
        "L": L, "centre": c,
        "source_box": box(c, c),
        "absorber_boxes": [box(i, j) for (i, j) in absorber_cells(k, chimney)],
        "chimney_boxes": ([box(c, j) for j in range(c + 1, k + 1)]
                          if chimney else []),
    }


def build_lattice(k=5, pitch=1.0, border=1.0, chimney=True,
                  cells_per_pitch=16, background=None, absorber=None):
    """Problem dict for ``run_transport``, generalising ``lattice_problem``.

    At the defaults this is bit-for-bit the paper's 7x7 cm / 112x112 case.
    """
    bg = dict(BACKGROUND if background is None else background)
    ab = dict(ABSORBER if absorber is None else absorber)
    spec = lattice_spec(k, pitch, border, chimney)
    L = spec["L"]

    n = int(round(L / pitch * cells_per_pitch))
    ex = np.linspace(0.0, L, n + 1)
    ey = np.linspace(0.0, L, n + 1)
    sig_s = np.full((n, n), bg["sig_s"])
    sig_a = np.full((n, n), bg["sig_a"])
    for bx in spec["absorber_boxes"]:
        _paint(sig_s, sig_a, ex, ey, bx, ab["sig_s"], ab["sig_a"])

    sx0, sy0, sx1, sy1 = spec["source_box"]
    return {
        "name": f"lattice_k{k}", "Lx": L, "Ly": L, "nx": n, "ny": n,
        "sig_s": sig_s, "sig_a": sig_a,
        "source": {"kind": "volumetric", "box": spec["source_box"]},
        # Paper lineouts are the Fig. 3b marker lines. Expressed relative to
        # the geometry they are the row through the middle of the second
        # lattice row from the top and the column through the second column.
        "lineout_y": border + (k - 0.5) * pitch,
        "lineout_x": border + 0.5 * pitch,
        "spec": spec,
    }


def draw_lattice(ax, spec, absorber_color="navy", source_color="crimson",
                 show_chimney=True, label=True):
    """Draw the lattice geometry (absorbers, source, chimney) onto ``ax``."""
    from matplotlib.patches import Rectangle

    L = spec["L"]
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")

    for (x0, y0, x1, y1) in spec["absorber_boxes"]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=absorber_color, edgecolor="none"))
    if show_chimney:
        for (x0, y0, x1, y1) in spec["chimney_boxes"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor="none", edgecolor="0.6",
                                   ls=":", lw=1.0))
    x0, y0, x1, y1 = spec["source_box"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=source_color, hatch="xxxx", lw=1.2))
    if label:
        ax.set_title(f"K={spec['k']}, pitch={spec['pitch']:g} cm, "
                     f"L={L:g} cm, {len(spec['absorber_boxes'])} blocks",
                     fontsize=9)
    return ax
