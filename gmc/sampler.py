"""Cell geometry, ODE integration, and sampling an exit state.

Two branches per particle:

  UNCOLLIDED, with probability exp(-chord).  The particle crosses without
  scattering, so its exit is an exact function of its entry.  That is a
  Dirac delta a smooth flow cannot represent, so it is done analytically
  and the network never sees it.

  COLLIDED.  Integrate dx/dt = v(x, t, c) from noise at t=0 to t=1, then
  decode the six outputs back to physical quantities.

Perimeter coordinate p runs anticlockwise from the bottom-left corner:
bottom [0, W), right [W, W+H), top [W+H, 2W+H), left [2W+H, 2W+2H).
"""

import numpy as np
import torch

from .data import encode_conditions

NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


def chord_length(W, H, xi, oxi, oyi):
    """Straight-line distance from entry (0, xi*H) to the wall along Omega."""
    y = xi * H
    with np.errstate(divide="ignore"):
        ty = np.where(oyi > 0, (H - y) / np.where(oyi > 0, oyi, 1.0),
                      np.where(oyi < 0, -y / np.where(oyi < 0, oyi, 1.0),
                               np.inf))
    return np.minimum(W / oxi, ty)


def perimeter_decode(p, W, H):
    """p -> (x, y) on the boundary and a face index (0 bot, 1 rt, 2 top, 3 lf)."""
    p, W, H = np.broadcast_arrays(p, W, H)
    p = np.mod(p, 2.0 * (W + H))
    x, y = np.empty_like(p), np.empty_like(p)
    face = np.empty(p.shape, np.int8)
    for i, m in enumerate((p < W, (p >= W) & (p < W + H),
                           (p >= W + H) & (p < 2 * W + H), p >= 2 * W + H)):
        if i == 0:
            x[m], y[m] = p[m], 0.0
        elif i == 1:
            x[m], y[m] = W[m], p[m] - W[m]
        elif i == 2:
            x[m], y[m] = 2 * W[m] + H[m] - p[m], H[m]
        else:
            x[m], y[m] = 0.0, 2 * (W[m] + H[m]) - p[m]
        face[m] = i
    return x, y, face


@torch.no_grad()
def integrate(model, z, c, steps, solver):
    """Fixed-step flow ODE solve.  Cost is steps * NFE_PER_STEP[solver]."""
    x, dt = z, 1.0 / steps
    # t only ever takes values on a half-step grid, so build them once
    ts = [torch.full((x.shape[0], 1), i / (2.0 * steps), device=x.device)
          for i in range(2 * steps + 1)]

    for i in range(steps):
        if solver == "euler":
            x = x + dt * model(x, ts[2 * i], c)
        elif solver == "heun":
            v0 = model(x, ts[2 * i], c)
            v1 = model(x + dt * v0, ts[2 * i + 2], c)
            x = x + 0.5 * dt * (v0 + v1)
        elif solver == "rk4":
            k1 = model(x, ts[2 * i], c)
            k2 = model(x + 0.5 * dt * k1, ts[2 * i + 1], c)
            k3 = model(x + 0.5 * dt * k2, ts[2 * i + 1], c)
            k4 = model(x + dt * k3, ts[2 * i + 2], c)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown solver {solver!r}")
    return x


class GMCBoundarySampler:
    """Trained model + normalizers -> physical exit states."""

    def __init__(self, model, ynorm, cnorm, device="cpu", ode_steps=5,
                 solver="heun"):
        self.model = model.to(device).eval()
        self.ynorm, self.cnorm = ynorm, cnorm
        self.device, self.ode_steps, self.solver = device, ode_steps, solver
        self.nfe_per_sample = ode_steps * NFE_PER_STEP[solver]
        self.reset()

    def reset(self):
        """Zero the cost counters that evaluate.py reports."""
        self.stats = {"calls": 0, "particles": 0, "to_network": 0, "nfe": 0,
                      "clamped": 0}

    def sample(self, W, H, xi, oxi, oyi, seed=0, uncollided="analytic"):
        """One exit state per entry state.  uncollided="off" forces the flow."""
        W, H, xi = np.broadcast_arrays(np.atleast_1d(np.asarray(W, float)),
                                       np.asarray(H, float),
                                       np.asarray(xi, float))
        oxi = np.broadcast_to(np.asarray(oxi, float), W.shape)
        oyi = np.broadcast_to(np.asarray(oyi, float), W.shape)
        n = W.size
        rng = np.random.default_rng(seed)

        chord = chord_length(W, H, xi, oxi, oyi)
        unc = (rng.random(n) < np.exp(-chord) if uncollided == "analytic"
               else np.zeros(n, bool))
        p, s, dirs = np.empty(n), np.empty(n), np.empty((n, 3))

        if unc.any():
            m = unc
            ex, ey = oxi[m] * chord[m], xi[m] * H[m] + oyi[m] * chord[m]
            p[m] = np.where(np.isclose(ex, W[m]), W[m] + ey,
                            np.where(oyi[m] > 0, 2 * W[m] + H[m] - ex, ex))
            s[m] = chord[m]
            # Omega_z is fixed in magnitude by |Omega|=1; both signs are equally likely
            oz = np.sqrt(np.maximum(0.0, 1.0 - oxi[m]**2 - oyi[m]**2))
            oz *= np.where(rng.random(m.sum()) < 0.5, 1.0, -1.0)
            dirs[m, 0], dirs[m, 1], dirs[m, 2] = oxi[m], oyi[m], oz

        m = ~unc
        n_flow = int(m.sum())
        if n_flow:
            c = torch.from_numpy(self.cnorm.transform(
                encode_conditions(W[m], H[m], xi[m], oxi[m], oyi[m])))
            g = torch.Generator().manual_seed(int(rng.integers(2**31)))
            z = torch.randn(n_flow, self.ynorm.mean.shape[0], generator=g)
            y = self.ynorm.inverse(integrate(
                self.model, z.to(self.device), c.to(self.device),
                self.ode_steps, self.solver).cpu().numpy())

            p[m] = np.mod(np.arctan2(y[:, 1], y[:, 0]) / (2 * np.pi), 1.0) \
                * 4.0 * W[m]
            omega = y[:, 2:5]
            dirs[m] = omega / (np.linalg.norm(omega, axis=1, keepdims=True)
                               + 1e-12)

            s_raw = W[m] * np.exp(y[:, 5].astype(float))
            ex, ey, _ = perimeter_decode(p[m], W[m], H[m])
            s_min = np.hypot(ex, ey - xi[m] * H[m])   # cannot be shorter
            s[m] = np.maximum(s_raw, s_min)
            self.stats["clamped"] += int((s_raw < s_min).sum())

        self.stats["calls"] += 1
        self.stats["particles"] += n
        self.stats["to_network"] += n_flow
        self.stats["nfe"] += n_flow * self.nfe_per_sample

        _, _, face = perimeter_decode(p, W, H)
        return {"p": p, "dir": dirs, "s": s, "face": face, "uncollided": unc}
