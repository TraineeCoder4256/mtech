"""The surfaces every geometry is built from, as Numba kernels.

Used by: geometry.py (which surfaces bound a cell, and how far they are).
Filled by: openmc_import.py (turns OpenMC surface objects into the two
arrays below).

A surface is two things in the compiled geometry:

    kind[i]      an integer code (the constants below)
    coef[i, :]   up to NCOEF numbers, in the same order as OpenMC's own
                 coefficients, so a surface reads the same in both codes
                 (a cylinder or cone on any axis: x0, y0, z0, r or r2,
                 then the axis direction dx, dy, dz, made unit length;
                 a quadric: a ... k, then lam, the size of its largest
                 curvature, worked out once on import)

Each surface is the zero set of a function f(x, y, z), with the same sign
convention as OpenMC: the "-" half-space is f < 0, the "+" half-space f > 0.

    plane       f = A x + B y + C z - D
    sphere      f = |p - c|^2 - r^2
    cylinder    f = (distance to the axis)^2 - r^2
    cone        f = (distance to the axis)^2 - r2 (axial offset)^2
                (r2 is OpenMC's "R squared": the slope of the cone, squared)
    torus       f = h^2 / b^2 + (rho - a)^2 / c^2 - 1, with h the offset
                along the axis and rho the distance from it: a tube of
                elliptical cross-section (c across, b along the axis) round
                a circle of radius a
    quadric     f = A x^2 + B y^2 + C z^2 + D xy + E yz + F xz
                    + G x + H y + J z + K

The axis of a cylinder or cone is x, y or z, or (since v1.4) any unit
direction d.  For any axis, with q = p - (x0, y0, z0) and h = q . d its
offset along the axis, the distance to the axis is |q - h d|: the same
formulas, measured in the axis's own frame.

Four things are asked of a surface, and each has its own kernel:

    value      f at a point                          -> which side we are on
    ray        distance along a direction to f = 0   -> where a flight stops
    nearest    distance to the closest point of f=0  -> how big a ball fits
    normal     gradient of f                         -> how a reflection turns

`nearest` is the one the ball method stands on.  It must NEVER be larger
than the true distance, or a ball would poke through the surface and the
answer would be wrong.  It is exact for planes, spheres, cylinders, cones
and tori (for a torus, the distance to the tube's ellipse in the plane
through the axis, found by bisection).  A general quadric has no formula,
so it returns a guaranteed LOWER bound: balls there are only smaller than
they could be, never too big.
"""

import numpy as np
from numba import njit

INF = 1.0e300

# --- surface kinds (an axis-aligned surface stores its axis in the code) ----
X_PLANE, Y_PLANE, Z_PLANE, PLANE = 0, 1, 2, 3
SPHERE = 4
X_CYLINDER, Y_CYLINDER, Z_CYLINDER = 5, 6, 7
X_CONE, Y_CONE, Z_CONE = 8, 9, 10
CYLINDER, CONE = 11, 12          # any axis (OpenMC's Cylinder, Cone)
X_TORUS, Y_TORUS, Z_TORUS = 13, 14, 15
QUADRIC = 16
NCOEF = 11                       # columns of coef

KIND_NAMES = {
    X_PLANE: "x-plane", Y_PLANE: "y-plane", Z_PLANE: "z-plane", PLANE: "plane",
    SPHERE: "sphere", X_CYLINDER: "x-cylinder", Y_CYLINDER: "y-cylinder",
    Z_CYLINDER: "z-cylinder", X_CONE: "x-cone", Y_CONE: "y-cone",
    Z_CONE: "z-cone", CYLINDER: "cylinder", CONE: "cone",
    X_TORUS: "x-torus", Y_TORUS: "y-torus", Z_TORUS: "z-torus",
    QUADRIC: "quadric",
}


@njit(cache=True, inline="always")
def _any_axis(c, x, y, z):
    """For a cylinder or cone on any axis: (q, h, rho2) with q = p - apex
    (or a point on the axis), h = q . d and rho2 = |q - h d|^2, the squared
    distance to the axis (from the perpendicular vector itself, not
    |q|^2 - h^2, which cancels badly far along the axis)."""
    qx, qy, qz = x - c[0], y - c[1], z - c[2]
    h = qx * c[4] + qy * c[5] + qz * c[6]
    tx, ty, tz = qx - h * c[4], qy - h * c[5], qz - h * c[6]
    return qx, qy, qz, h, tx * tx + ty * ty + tz * tz


@njit(cache=True, inline="always")
def _axis_frame(kind, x, y, z):
    """Reorder (x, y, z) as (a, b, c) with c along the axis of a cylinder or
    cone.  An x-cylinder has its axis along x, so c = x and (a, b) = (y, z)."""
    if kind == X_CYLINDER or kind == X_CONE:
        return y, z, x
    if kind == Y_CYLINDER or kind == Y_CONE:
        return x, z, y
    return x, y, z


@njit(cache=True, inline="always")
def _torus_frame(kind, c, x, y, z):
    """For a torus: (qa, qb, qc, rho, h) -- the offset from its centre in
    the axis frame (qc along the axis), the distance rho from the axis and
    the axial offset h = qc."""
    if kind == X_TORUS:
        qa, qb, qc = y - c[1], z - c[2], x - c[0]
    elif kind == Y_TORUS:
        qa, qb, qc = x - c[0], z - c[2], y - c[1]
    else:
        qa, qb, qc = x - c[0], y - c[1], z - c[2]
    return qa, qb, qc, np.sqrt(qa * qa + qb * qb), qc


@njit(cache=True)
def _ellipse_root(r0, z0, z1):
    """The root s of F(s) = (r0 z0 / (s + r0))^2 + (z1 / (s + 1))^2 - 1 in
    Eberly's point-to-ellipse distance (2013).  F is convex and falls for
    s > -1, so Newton steps from z1 - 1 (where F >= 0) climb to the root
    without overshooting it."""
    n0 = r0 * z0
    s = z1 - 1.0
    for _ in range(100):
        a, b = n0 / (s + r0), z1 / (s + 1.0)
        f = a * a + b * b - 1.0
        if f <= 0.0:
            break
        df = -2.0 * (a * a / (s + r0) + b * b / (s + 1.0))
        ds = -f / df
        s += ds
        if ds <= 1.0e-15 * (1.0 + abs(s)):
            break
    return s


@njit(cache=True)
def _ellipse_distance(e0, e1, y0, y1):
    """Distance from (y0, y1) to the ellipse (x0/e0)^2 + (x1/e1)^2 = 1,
    the axes in either order (Eberly's robust bisection)."""
    y0, y1 = abs(y0), abs(y1)
    if e0 < e1:
        e0, e1, y0, y1 = e1, e0, y1, y0
    if y1 > 0.0:
        if y0 > 0.0:
            z0, z1 = y0 / e0, y1 / e1
            g = z0 * z0 + z1 * z1 - 1.0
            if g == 0.0:
                return 0.0
            r0 = (e0 / e1) ** 2
            sb = _ellipse_root(r0, z0, z1)
            x0, x1 = r0 * y0 / (sb + r0), y1 / (sb + 1.0)
            return np.sqrt((x0 - y0) ** 2 + (x1 - y1) ** 2)
        return abs(y1 - e1)
    num, den = e0 * y0, e0 * e0 - e1 * e1
    if num < den:
        xd = num / den
        x0, x1 = e0 * xd, e1 * np.sqrt(max(1.0 - xd * xd, 0.0))
        return np.sqrt((x0 - y0) ** 2 + x1 * x1)
    return abs(y0 - e0)


@njit(cache=True)
def _quadric_grad(c, x, y, z):
    return (2.0 * c[0] * x + c[3] * y + c[5] * z + c[6],
            2.0 * c[1] * y + c[3] * x + c[4] * z + c[7],
            2.0 * c[2] * z + c[4] * y + c[5] * x + c[8])


@njit(cache=True, inline="always")
def _quartic(b3, b2, b1, b0, t):
    """t^4 + b3 t^3 + b2 t^2 + b1 t + b0 and its derivative."""
    return ((((t + b3) * t + b2) * t + b1) * t + b0,
            ((4.0 * t + 3.0 * b3) * t + 2.0 * b2) * t + b1)


@njit(cache=True, inline="always")
def _cubic(b3, b2, b1, t):
    """The quartic's derivative 4 t^3 + 3 b3 t^2 + 2 b2 t + b1, and its
    own derivative."""
    return (((4.0 * t + 3.0 * b3) * t + 2.0 * b2) * t + b1,
            (12.0 * t + 6.0 * b3) * t + 2.0 * b2)


@njit(cache=True)
def _bracket_root(which, b3, b2, b1, b0, lo, hi, flo):
    """The root of the quartic (which = 0) or of its derivative (1) in
    [lo, hi], where it changes sign once (flo = its value at lo):
    Newton steps, falling back to bisection whenever a step leaves the
    bracket, so it always converges."""
    t = 0.5 * (lo + hi)
    for _ in range(200):
        if which == 0:
            f, df = _quartic(b3, b2, b1, b0, t)
        else:
            f, df = _cubic(b3, b2, b1, t)
        if f == 0.0:
            return t
        if (f > 0.0) == (flo > 0.0):
            lo = t
        else:
            hi = t
        if hi - lo <= 1.0e-13 * (1.0 + abs(t)):
            return 0.5 * (lo + hi)
        tn = t - f / df if df != 0.0 else lo - 1.0
        t = tn if lo < tn < hi else 0.5 * (lo + hi)
    return t


@njit(cache=True)
def _torus_ray(kind, c, x, y, z, u, v, w):
    """First crossing of a ray with a torus: the smallest positive root of
    a quartic (the torus equation squared out).  With the tube inside its
    ring (c < a) the quartic's roots are exactly the torus's crossings.
    A horn or spindle torus (c >= a, v1.6.1) squares in false roots too:
    S = rho^2 + k h^2 + a^2 - c^2 = 2 a rho on the surface, and squaring
    adds S = -2 a rho (the tube's mirror image through the axis), so a
    root counts only where S >= 0.

    Found without a general polynomial solver: the quartic's turning
    points split [0, T] (T bounds every root) into pieces where it is
    monotone; the first piece whose ends differ in sign holds the answer.
    The turning points are the roots of the cubic derivative, found the
    same way between the roots of ITS derivative (a quadratic)."""
    qa, qb, qc, _, _ = _torus_frame(kind, c, x, y, z)
    if kind == X_TORUS:
        ua, ub, uc = v, w, u
    elif kind == Y_TORUS:
        ua, ub, uc = u, w, v
    else:
        ua, ub, uc = u, v, w
    A, k = c[3], (c[5] / c[4]) ** 2
    # rho^2(t) = R0 + 2 R1 t + R2 t^2;  s(t) = rho^2 + k h^2 + A^2 - c^2
    R0, R1, R2 = qa * qa + qb * qb, qa * ua + qb * ub, ua * ua + ub * ub
    S0 = R0 + k * qc * qc + A * A - c[5] * c[5]
    S1 = 2.0 * R1 + 2.0 * k * qc * uc
    S2 = R2 + k * uc * uc                     # > 0 for any direction
    a4, q4 = S2 * S2, 4.0 * A * A
    b3 = 2.0 * S1 * S2 / a4
    b2 = (S1 * S1 + 2.0 * S0 * S2 - q4 * R2) / a4
    b1 = (2.0 * S0 * S1 - 2.0 * q4 * R1) / a4
    b0 = (S0 * S0 - q4 * R0) / a4
    T = 1.0 + max(abs(b3), abs(b2), abs(b1), abs(b0))    # Cauchy's bound
    # turning points of the derivative: 12 t^2 + 6 b3 t + 2 b2 = 0
    e1, e2 = T, T
    disc = 36.0 * b3 * b3 - 96.0 * b2
    if disc > 0.0:
        sq = np.sqrt(disc)
        e1, e2 = (-6.0 * b3 - sq) / 24.0, (-6.0 * b3 + sq) / 24.0
    # the quartic's turning points in (0, T), in order (at most three)
    tp0, tp1, tp2 = T, T, T
    ntp = 0
    lo = 0.0
    for hi in (e1, e2, T):
        if hi <= lo:
            continue
        hi = min(hi, T)
        flo, _ = _cubic(b3, b2, b1, lo)
        fhi, _ = _cubic(b3, b2, b1, hi)
        if flo != 0.0 and (flo > 0.0) != (fhi > 0.0):
            r = _bracket_root(1, b3, b2, b1, b0, lo, hi, flo)
            if ntp == 0:
                tp0 = r
            elif ntp == 1:
                tp1 = r
            else:
                tp2 = r
            ntp += 1
        lo = hi
    # the first monotone piece of [0, T] whose ends differ in sign (and,
    # for a horn or spindle torus, whose root is not a false one)
    lo = 0.0
    flo, _ = _quartic(b3, b2, b1, b0, 0.0)
    for j in range(ntp + 1):
        hi = tp0 if j == 0 else tp1 if j == 1 else tp2 if j == 2 else T
        if j == ntp:
            hi = T
        fhi, _ = _quartic(b3, b2, b1, b0, hi)
        if flo != 0.0 and (flo > 0.0) != (fhi > 0.0):
            r = _bracket_root(0, b3, b2, b1, b0, lo, hi, flo)
            if c[5] < A or (S2 * r + S1) * r + S0 >= 0.0:
                return r
        lo, flo = hi, fhi
    return INF


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
    if kind == QUADRIC:
        return (c[0] * x * x + c[1] * y * y + c[2] * z * z + c[3] * x * y +
                c[4] * y * z + c[5] * x * z + c[6] * x + c[7] * y +
                c[8] * z + c[9])
    if kind >= X_TORUS:
        _, _, _, rho, h = _torus_frame(kind, c, x, y, z)
        return h * h / (c[4] * c[4]) + (rho - c[3]) ** 2 / (c[5] * c[5]) - 1.0
    if kind >= CYLINDER:
        _, _, _, h, rho2 = _any_axis(c, x, y, z)
        if kind == CYLINDER:
            return rho2 - c[3] * c[3]
        return rho2 - c[3] * h * h
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
    if kind == QUADRIC:
        # f(p + t u) = f + t (g . u) + t^2 (u . M u)
        gx, gy, gz = _quadric_grad(c, x, y, z)
        uMu = (c[0] * u * u + c[1] * v * v + c[2] * w * w + c[3] * u * v +
               c[4] * v * w + c[5] * u * w)
        return _smallest_positive_root(uMu, 0.5 * (gx * u + gy * v + gz * w),
                                       value(QUADRIC, c, x, y, z))
    if kind >= X_TORUS:
        return _torus_ray(kind, c, x, y, z, u, v, w)
    if kind >= CYLINDER:
        # with ud = u . d: |q + t u|^2 - (h + t ud)^2 (1 + k), k = 0 for a
        # cylinder (minus r^2) and r2 for a cone
        qx, qy, qz, h, rho2 = _any_axis(c, x, y, z)
        ud = u * c[4] + v * c[5] + w * c[6]
        qu = qx * u + qy * v + qz * w
        if kind == CYLINDER:
            return _smallest_positive_root(1.0 - ud * ud, qu - h * ud,
                                           rho2 - c[3] * c[3])
        k = 1.0 + c[3]
        return _smallest_positive_root(1.0 - k * ud * ud, qu - k * h * ud,
                                       rho2 - c[3] * h * h)
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

    Exact for every kind but the quadric, which returns a lower bound (the
    ball method needs only that: something smaller, never larger).
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
    if kind == QUADRIC:
        # Within distance r of p, f changes by at most |grad f| r + lam r^2
        # (lam = the largest |eigenvalue| of the quadratic part), so f
        # cannot reach zero before r solves lam r^2 + |grad f| r = |f|.
        # Exact for a plane; tight near the surface for any quadric.
        f = abs(value(QUADRIC, c, x, y, z))
        gx, gy, gz = _quadric_grad(c, x, y, z)
        gn = np.sqrt(gx * gx + gy * gy + gz * gz)
        den = gn + np.sqrt(gn * gn + 4.0 * c[10] * f)
        return 2.0 * f / den if den > 0.0 else INF
    if kind >= X_TORUS:
        # the nearest point lies in the half-plane through the axis and p,
        # where the torus is an ellipse (semi-axes c across, b along the
        # axis) centred at rho = a
        # (v1.6.1) For a horn or spindle torus (c >= a) the tube's section
        # in the opposite half-plane, the ellipse centred at rho = -a,
        # reaches into this one; the distance is the smaller of the two
        # (a lower bound: each ellipse counts in full, although only its
        # part on its own side of the axis is surface).
        _, _, _, rho, h = _torus_frame(kind, c, x, y, z)
        if c[4] == c[5]:
            d = abs(np.sqrt((rho - c[3]) ** 2 + h * h) - c[5])
            if c[5] >= c[3]:
                d = min(d, abs(np.sqrt((rho + c[3]) ** 2 + h * h) - c[5]))
            return d
        d = _ellipse_distance(c[5], c[4], rho - c[3], h)
        if c[5] >= c[3]:
            d = min(d, _ellipse_distance(c[5], c[4], rho + c[3], h))
        return d
    if kind == CYLINDER:
        _, _, _, _, rho2 = _any_axis(c, x, y, z)
        return abs(np.sqrt(rho2) - c[3])
    if kind <= Z_CYLINDER:
        a, b, _ = _axis_frame(kind, x, y, z)
        da, db = a - c[0], b - c[1]
        return abs(np.sqrt(da * da + db * db) - c[2])
    # Cone: work in the half-plane through the axis and the point, with
    # rho = distance from the axis and h = offset along it.  The surface
    # there is two rays from the apex, rho = k |h| with k = sqrt(r2).  The
    # distance to the cone is the distance to the nearer ray.
    if kind == CONE:
        _, _, _, h, rho2 = _any_axis(c, x, y, z)
        rho = np.sqrt(rho2)
    else:
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
    if kind == QUADRIC:
        return _quadric_grad(c, x, y, z)
    if kind >= X_TORUS:
        qa, qb, qc, rho, h = _torus_frame(kind, c, x, y, z)
        fr = 2.0 * (rho - c[3]) / (c[5] * c[5] * rho) if rho > 0.0 else 0.0
        ga, gb, gc = fr * qa, fr * qb, 2.0 * h / (c[4] * c[4])
        if kind == X_TORUS:
            return gc, ga, gb
        if kind == Y_TORUS:
            return ga, gc, gb
        return ga, gb, gc
    if kind >= CYLINDER:
        # grad of |q|^2 - (1 + k) h^2 (- r^2): 2 (q - (1 + k) h d)
        qx, qy, qz, h, _ = _any_axis(c, x, y, z)
        f = h if kind == CYLINDER else (1.0 + c[3]) * h
        return (2.0 * (qx - f * c[4]), 2.0 * (qy - f * c[5]),
                2.0 * (qz - f * c[6]))
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
