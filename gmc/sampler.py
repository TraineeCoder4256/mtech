"""Noise -> exit state: ODE integration, decoding, and the uncollided branch.

Sampling one exit state for one entry state (W, H, xi, Omega_in) has two
branches.

1. UNCOLLIDED.  With probability exp(-d), where d is the straight-line
   chord from the entry point to the boundary along Omega_in (in mean free
   paths), the particle crosses the cell without scattering at all.  Its
   exit is then an exact function of its entry: same direction, path length
   d.  That is a Dirac delta, which a smooth flow cannot represent, so it
   is sampled analytically here and the network never sees it.

2. COLLIDED.  Integrate dx/dt = v(x, t, c) from t = 0 to t = 1 starting at
   x ~ N(0, I), then decode the six outputs back to physical quantities:

     (cos, sin)   -> perimeter coordinate p, via atan2, modulo 4W
     (Ox, Oy, Oz) -> renormalised onto the unit sphere
     log(s/W)     -> s = W exp(.), then raised to the straight-line
                     distance from the entry point to the decoded exit if
                     it came out below it (a path cannot be shorter than
                     that).  How often that clamp fires is recorded in
                     last_clamp_frac -- it is a defect, not a feature.
"""

import numpy as np
import torch

from .data import encode_conditions
from .geometry import chord_length, perimeter_decode  # noqa: F401  (re-export)

# network evaluations per ODE step, per solver
NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


@torch.no_grad()
def integrate(model, z, c, steps=25, solver="heun"):
    """Fixed-step integration of the flow ODE.

    Inference cost is NFE = steps * NFE_PER_STEP[solver], not the step
    count.  Under the optimal-transport interpolation used in training the
    conditional path is a straight line at constant velocity, so the learned
    field is close to straight and a high-order solver buys little: at
    matched NFE a cheap solver with more steps is usually the better trade.
    """
    x = z
    dt = 1.0 / steps
    n = x.shape[0]
    # the solver only needs t on a fixed grid of half-steps; build them once
    # rather than allocating (and, on a GPU, launching a fill kernel for)
    # one tensor per stage per step
    grid = {}

    def tt(v):
        k = round(v * 2 * steps)
        if k not in grid:
            grid[k] = torch.full((n, 1), k / (2.0 * steps), device=x.device,
                                 dtype=x.dtype)
        return grid[k]

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


class GMCBoundarySampler:
    """Trained boundary model + normalizers -> physical exit states."""

    def __init__(self, model, ynorm, cnorm, device="cpu", ode_steps=25,
                 solver="heun"):
        self.model = model.to(device).eval()
        self.ynorm, self.cnorm = ynorm, cnorm
        self.device = device
        self.ode_steps = ode_steps
        self.solver = solver
        # diagnostics from the most recent sample() call
        self.last_clamp_frac = 0.0
        # running cost counters, so a transport solve can report exactly how
        # much of its wall time went into the network (see reset_counters)
        self.reset_counters()

    def reset_counters(self):
        self.n_calls = 0
        self.n_particles = 0        # entry states asked for
        self.n_flow = 0             # of those, the ones that hit the network
        self.nfe = 0                # network evaluations, the unit of cost

    @property
    def nfe_per_sample(self):
        return self.ode_steps * NFE_PER_STEP[self.solver]

    def sample(self, W, H, xi, oxi, oyi, seed=0, uncollided="analytic"):
        """Sample one exit state per entry state (arrays broadcast together).

        Returns dict: p, dir (n,3), s, face, uncollided mask.
        uncollided="off" forces every particle through the flow, which is
        what you want when comparing against k>0 reference data.
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

        # ---- branch 1: analytic uncollided exits ------------------------
        if unc.any():
            m = unc
            ex = 0.0 + oxi[m] * d[m]
            ey = xi[m] * H[m] + oyi[m] * d[m]
            hit_x = np.isclose(ex, W[m])
            p[m] = np.where(hit_x, W[m] + ey,                     # right
                            np.where(oyi[m] > 0,
                                     W[m] + H[m] + (W[m] - ex),   # top
                                     ex))                         # bottom
            s[m] = d[m]
            # Omega_z is not used by the in-plane trajectory but is part of
            # the exit state, and both signs are equally likely
            ozu = np.sqrt(np.maximum(0.0, 1.0 - oxi[m]**2 - oyi[m]**2))
            ozu *= np.where(rng.random(m.sum()) < 0.5, 1.0, -1.0)
            dirs[m, 0], dirs[m, 1], dirs[m, 2] = oxi[m], oyi[m], ozu

        # ---- branch 2: flow-matched collided exits ----------------------
        m = ~unc
        n_flow = int(m.sum())
        if n_flow:
            c_raw = encode_conditions(W[m], H[m], xi[m], oxi[m], oyi[m])
            c = torch.from_numpy(self.cnorm.transform(c_raw)).to(
                self.device, non_blocking=True)
            g = torch.Generator(device="cpu").manual_seed(
                int(rng.integers(2**31)))
            z = torch.randn(n_flow, self.ynorm.mean.shape[0], generator=g)
            x = integrate(self.model, z.to(self.device), c,
                          self.ode_steps, self.solver)
            y = self.ynorm.inverse(x.cpu().numpy())

            theta = np.arctan2(y[:, 1], y[:, 0])
            p[m] = np.mod(theta / (2.0 * np.pi), 1.0) * 4.0 * W[m]
            omega = y[:, 2:5]
            omega /= np.linalg.norm(omega, axis=1, keepdims=True) + 1e-12
            dirs[m] = omega

            sv = W[m] * np.exp(y[:, 5].astype(np.float64))
            ex_x, ex_y, _ = perimeter_decode(p[m], W[m], H[m])
            smin = np.hypot(ex_x, ex_y - xi[m] * H[m])
            s[m] = np.maximum(sv, smin)
            self.last_clamp_frac = float((sv < smin).mean())
        else:
            self.last_clamp_frac = 0.0

        self.n_calls += 1
        self.n_particles += n
        self.n_flow += n_flow
        self.nfe += n_flow * self.nfe_per_sample

        _, _, face = perimeter_decode(p, W, H)
        return {"p": p, "dir": dirs, "s": s, "face": face, "uncollided": unc}
