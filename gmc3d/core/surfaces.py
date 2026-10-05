"""The surfaces every geometry is built from, as Numba kernels.

Used by: geometry.py (which surfaces bound a cell, and how far they are).
Filled by: openmc_import.py (turns OpenMC surface objects into the two
arrays below).

A surface is two things in the compiled geometry:

    kind[i]      an integer code (the constants below)
    coef[i, :]   up to 4 numbers, in the same order as OpenMC's own
                 coefficients, so a surface reads the same in both codes

Each surface is the zero set of a function f(x, y, z), with the same sign
convention as OpenMC: the "-" half-space is f < 0, the "+" half-space f > 0.

    plane       f = A x + B y + C z - D
    sphere      f = |p - c|^2 - r^2
    cylinder    f = (distance to the axis)^2 - r^2
    cone        f = (distance to the axis)^2 - r2 (axial offset)^2
                (r2 is OpenMC's "R squared": the slope of the cone, squared)

Four things are asked of a surface, and each has its own kernel:

    value      f at a point                          -> which side we are on
    ray        distance along a direction to f = 0   -> where a flight stops
    nearest    distance to the closest point of f=0  -> how big a ball fits
    normal     gradient of f                         -> how a reflection turns

`nearest` is the one the ball method stands on.  It must NEVER be larger
than the true distance, or a ball would poke through the surface and the
answer would be wrong.  It is exact for planes, spheres, cylinders and cones;
any surface type added later must return a safe lower bound.
"""

import numpy as np
from numba import njit

INF = 1.0e300

# --- surface kinds (an axis-aligned surface stores its axis in the code) ----
X_PLANE, Y_PLANE, Z_PLANE, PLANE = 0, 1, 2, 3
SPHERE = 4
X_CYLINDER, Y_CYLINDER, Z_CYLINDER = 5, 6, 7
X_CONE, Y_CONE, Z_CONE = 8, 9, 10

KIND_NAMES = {
    X_PLANE: "x-plane", Y_PLANE: "y-plane", Z_PLANE: "z-plane", PLANE: "plane",
    SPHERE: "sphere", X_CYLINDER: "x-cylinder", Y_CYLINDER: "y-cylinder",
    Z_CYLINDER: "z-cylinder", X_CONE: "x-cone", Y_CONE: "y-cone",
    Z_CONE: "z-cone",
}


@njit(cache=True, inline="always")
def _axis_frame(kind, x, y, z):
    """Reorder (x, y, z) as (a, b, c) with c along the axis of a cylinder or
    cone.  An x-cylinder has its axis along x, so c = x and (a, b) = (y, z)."""
    if kind == X_CYLINDER or kind == X_CONE:
        return y, z, x
    if kind == Y_CYLINDER or kind == Y_CONE:
        return x, z, y
    return x, y, z


@njit(cache=True)
def value(kind, c, x, y, z):
    """f(x, y, z): negative on the "-" side, positive on the "+" side."""
    if kind == X_PLANE:
        return x - c[0]
    if kind == Y_PLANE:
        return y - c[0]
    if kind == Z_PLANE:
        return z - c[0]
    if kind == PLANE:
        return c[0] * x + c[1] * y + c[2] * z - c[3]
    if kind == SPHERE:
        dx, dy, dz = x - c[0], y - c[1], z - c[2]
        return dx * dx + dy * dy + dz * dz - c[3] * c[3]
    if kind <= Z_CYLINDER:
        # cylinder: c = (centre a, centre b, r) in the two transverse axes
        a, b, _ = _axis_frame(kind, x, y, z)
        da, db = a - c[0], b - c[1]
        return da * da + db * db - c[2] * c[2]
    # cone: c = (x0, y0, z0, r2) -- the apex in global axes, then slope^2
    a, b, cc = _axis_frame(kind, x - c[0], y - c[1], z - c[2])
    return a * a + b * b - c[3] * cc * cc


@njit(cache=True, inline="always")
def _smallest_positive_root(qa, qk, qc):
    """Smallest t > 0 with qa t^2 + 2 qk t + qc = 0, or INF.

    Written with the half-coefficient qk so the discriminant is qk^2 - qa qc,
    and with the numerically stable pairing of the two roots (no cancellation
    when one root is tiny, which is exactly the case just after a crossing).
    """
    if abs(qa) < 1.0e-14:                      # degenerates to a line
        if abs(qk) < 1.0e-300:
            return INF
        t = -qc / (2.0 * qk)
        return t if t > 0.0 else INF
    disc = qk * qk - qa * qc
    if disc < 0.0:
        return INF
    sq = np.sqrt(disc)
    q = -(qk + sq) if qk >= 0.0 else -(qk - sq)
    t1 = q / qa
    t2 = qc / q if q != 0.0 else INF
    lo, hi = (t1, t2) if t1 < t2 else (t2, t1)
    if lo > 0.0:
        return lo
    if hi > 0.0:
        return hi
    return INF


@njit(cache=True)
def ray(kind, c, x, y, z, u, v, w):
    """Distance from (x, y, z) along the unit direction (u, v, w) to the
    surface, or INF if the ray never reaches it."""
    if kind <= PLANE:
        if kind == X_PLANE:
            f, g = x - c[0], u
        elif kind == Y_PLANE:
            f, g = y - c[0], v
        elif kind == Z_PLANE:
            f, g = z - c[0], w
        else:
            f = c[0] * x + c[1] * y + c[2] * z - c[3]
            g = c[0] * u + c[1] * v + c[2] * w
        if g == 0.0:
            return INF
        t = -f / g
        return t if t > 0.0 else INF
    if kind == SPHERE:
        dx, dy, dz = x - c[0], y - c[1], z - c[2]
        return _smallest_positive_root(
            1.0, dx * u + dy * v + dz * w,
            dx * dx + dy * dy + dz * dz - c[3] * c[3])
    if kind <= Z_CYLINDER:
        a, b, _ = _axis_frame(kind, x, y, z)
        ua, ub, _ = _axis_frame(kind, u, v, w)
        da, db = a - c[0], b - c[1]
        return _smallest_positive_root(
            ua * ua + ub * ub, da * ua + db * ub,
            da * da + db * db - c[2] * c[2])
    a, b, cc = _axis_frame(kind, x - c[0], y - c[1], z - c[2])
    ua, ub, uc = _axis_frame(kind, u, v, w)
    r2 = c[3]
    return _smallest_positive_root(
        ua * ua + ub * ub - r2 * uc * uc,
        a * ua + b * ub - r2 * cc * uc,
        a * a + b * b - r2 * cc * cc)


@njit(cache=True)
def nearest(kind, c, x, y, z):
    """Distance from (x, y, z) to the nearest point on the surface.

    Exact for every kind here.  The ball method needs only a LOWER bound, so
    a future surface without a closed form may return something smaller,
    never larger.
    """
    if kind == X_PLANE:
        return abs(x - c[0])
    if kind == Y_PLANE:
        return abs(y - c[0])
    if kind == Z_PLANE:
        return abs(z - c[0])
    if kind == PLANE:
        nrm = np.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
        return abs(c[0] * x + c[1] * y + c[2] * z - c[3]) / nrm
    if kind == SPHERE:
        dx, dy, dz = x - c[0], y - c[1], z - c[2]
        return abs(np.sqrt(dx * dx + dy * dy + dz * dz) - c[3])
    if kind <= Z_CYLINDER:
        a, b, _ = _axis_frame(kind, x, y, z)
        da, db = a - c[0], b - c[1]
        return abs(np.sqrt(da * da + db * db) - c[2])
    # Cone: work in the half-plane through the axis and the point, with
    # rho = distance from the axis and h = offset along it.  The surface
    # there is two rays from the apex, rho = k |h| with k = sqrt(r2).  The
    # distance to the cone is the distance to the nearer ray.
    a, b, h = _axis_frame(kind, x - c[0], y - c[1], z - c[2])
    rho = np.sqrt(a * a + b * b)
    k = np.sqrt(c[3])
    nrm = np.sqrt(k * k + 1.0)
    best = INF
    for sgn in (1.0, -1.0):
        d0, d1 = k / nrm, sgn / nrm          # unit vector along the ray
        t = rho * d0 + h * d1
        if t < 0.0:
            t = 0.0                           # nearest point is the apex
        e0, e1 = rho - t * d0, h - t * d1
        dist = np.sqrt(e0 * e0 + e1 * e1)
        if dist < best:
            best = dist
    return best


@njit(cache=True)
def normal(kind, c, x, y, z):
    """Gradient of f at (x, y, z) -- not normalised.  Used for reflection."""
    if kind == X_PLANE:
        return 1.0, 0.0, 0.0
    if kind == Y_PLANE:
        return 0.0, 1.0, 0.0
    if kind == Z_PLANE:
        return 0.0, 0.0, 1.0
    if kind == PLANE:
        return c[0], c[1], c[2]
    if kind == SPHERE:
        return 2.0 * (x - c[0]), 2.0 * (y - c[1]), 2.0 * (z - c[2])
    if kind <= Z_CYLINDER:
        a, b, _ = _axis_frame(kind, x, y, z)
        ga, gb, gc = 2.0 * (a - c[0]), 2.0 * (b - c[1]), 0.0
    else:
        a, b, cc = _axis_frame(kind, x - c[0], y - c[1], z - c[2])
        ga, gb, gc = 2.0 * a, 2.0 * b, -2.0 * c[3] * cc
    # undo the axis reordering of _axis_frame
    if kind == X_CYLINDER or kind == X_CONE:
        return gc, ga, gb
    if kind == Y_CYLINDER or kind == Y_CONE:
        return ga, gc, gb
    return ga, gb, gc
