"""Sampling: noise -> exit state, plus the analytic uncollided branch.

Full sampling pipeline for one entry state (W, H, xi, Omega_in):

  1. Uncollided branch (docs Sec 3.1).  With probability exp(-d), where d is
     the straight-line chord from the entry point to the boundary along
     Omega_in (in mfp), the particle crosses without scattering:
     the exit is analytic (Omega_exit = Omega_in, s = d) and a smooth flow
     cannot represent this Dirac component, so it is sampled exactly here
     and the network never sees it.
  2. Otherwise integrate dx/dt = v_theta(x, t, c) from t=0 to t=1 (Heun,
     fixed steps) starting from x ~ N(0, I), then decode:
       (cos, sin) -> perimeter coordinate p via atan2 (mod 4W),
       (Ox, Oy, Oz) -> renormalised to the unit sphere,
       log(s/W)    -> s = W exp(.), clamped to the physical minimum
                      distance from the entry point to the decoded exit.
"""

import numpy as np
import torch

from .data import encode_conditions


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


@torch.no_grad()
def integrate(model, z, c, steps=25, solver="heun"):
    """Fixed-step integration of the flow ODE.

    Network evaluations per sample (NFE) = steps * cost, with cost 1 for
    Euler, 2 for Heun, 4 for RK4.  NFE is the quantity that matters for
    inference cost, not the step count: the reference work uses RK4 with 12
    steps (48 NFE).

    Under the optimal-transport interpolation the conditional path is a
    straight line with constant velocity, so the learned field is close to
    straight and high-order solvers buy little -- at matched NFE a cheap
    solver with more steps is generally the better trade.
    """
    x = z
    dt = 1.0 / steps
    n = x.shape[0]

    def tt(v):
        return torch.full((n, 1), v, device=x.device)

    for i in range(steps):
        t0 = tt(i * dt)
        if solver == "euler":
            x = x + dt * model(x, t0, c)
        elif solver == "heun":
            v0 = model(x, t0, c)
            v1 = model(x + dt * v0, tt((i + 1) * dt), c)
            x = x + 0.5 * dt * (v0 + v1)
        elif solver == "rk4":
            k1 = model(x, t0, c)
            k2 = model(x + 0.5 * dt * k1, tt((i + 0.5) * dt), c)
            k3 = model(x + 0.5 * dt * k2, tt((i + 0.5) * dt), c)
            k4 = model(x + dt * k3, tt((i + 1) * dt), c)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown solver {solver!r}")
    return x


NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


@torch.no_grad()
def integrate_heun(model, z, c, steps=25):
    """Backwards-compatible alias."""
    return integrate(model, z, c, steps, "heun")


class GMCBoundarySampler:
    """Trained boundary model + normalizers -> physical exit states."""

    def __init__(self, model, ynorm, cnorm, device="cpu", ode_steps=25,
                 solver="heun"):
        self.model = model.to(device).eval()
        self.ynorm, self.cnorm = ynorm, cnorm
        self.device = device
        self.ode_steps = ode_steps
        self.solver = solver

    def sample(self, W, H, xi, oxi, oyi, seed=0, uncollided="analytic"):
        """Sample one exit state per entry state (arrays broadcast together).

        Returns dict: p, dir (n,3), s, face, uncollided mask.
        uncollided: "analytic" = full pipeline (default);
                    "off"      = force every particle through the flow
                                 (for evaluating the learned part alone
                                 against k>0 MC data).
        """
        W, H, xi = np.broadcast_arrays(np.atleast_1d(np.asarray(W, np.float64)),
                                       np.asarray(H, np.float64),
                                       np.asarray(xi, np.float64))
        oxi = np.broadcast_to(np.asarray(oxi, np.float64), W.shape)
        oyi = np.broadcast_to(np.asarray(oyi, np.float64), W.shape)
        n = W.size
        rng = np.random.default_rng(seed)

        d = chord_length(W, H, xi, oxi, oyi)
        if uncollided == "analytic":
            unc = rng.random(n) < np.exp(-d)
        else:
            unc = np.zeros(n, bool)

        p = np.empty(n)
        s = np.empty(n)
        dirs = np.empty((n, 3))

        # ---- analytic uncollided exits ---------------------------------
        if unc.any():
            m = unc
            ex = 0.0 + oxi[m] * d[m]
            ey = xi[m] * H[m] + oyi[m] * d[m]
            # snap to the exited face and encode perimeter coordinate
            hit_x = np.isclose(ex, W[m])
            pm = np.where(hit_x, W[m] + ey,                       # right
                          np.where(oyi[m] > 0,
                                   W[m] + H[m] + (W[m] - ex),     # top
                                   ex))                           # bottom
            p[m] = pm
            s[m] = d[m]
            ozu = np.sqrt(np.maximum(0.0, 1.0 - oxi[m]**2 - oyi[m]**2))
            ozu *= np.where(rng.random(m.sum()) < 0.5, 1.0, -1.0)
            dirs[m, 0], dirs[m, 1], dirs[m, 2] = oxi[m], oyi[m], ozu

        # ---- flow-matched collided exits -------------------------------
        m = ~unc
        if m.any():
            c_raw = encode_conditions(W[m], H[m], xi[m], oxi[m], oyi[m])
            c = torch.from_numpy(self.cnorm.transform(c_raw)).to(self.device)
            g = torch.Generator(device="cpu").manual_seed(int(rng.integers(2**31)))
            z = torch.randn(int(m.sum()), self.ynorm.mean.shape[0], generator=g)
            x = integrate(self.model, z.to(self.device), c,
                          self.ode_steps, self.solver)
            y = self.ynorm.inverse(x.cpu().numpy())

            theta = np.arctan2(y[:, 1], y[:, 0])            # (cos,sin) -> angle
            p[m] = np.mod(theta / (2.0 * np.pi), 1.0) * 4.0 * W[m]
            omega = y[:, 2:5]
            omega /= np.linalg.norm(omega, axis=1, keepdims=True) + 1e-12
            dirs[m] = omega
            # s = W exp(target); the exit point implies a hard lower bound
            # (straight line from entry to exit), enforce it
            sv = W[m] * np.exp(y[:, 5])
            ex_x, ex_y, _ = perimeter_decode(p[m], W[m], H[m])
            smin = np.hypot(ex_x - 0.0, ex_y - xi[m] * H[m])
            s[m] = np.maximum(sv, smin)

        _, _, face = perimeter_decode(p, W, H)
        return {"p": p, "dir": dirs, "s": s, "face": face, "uncollided": unc}
