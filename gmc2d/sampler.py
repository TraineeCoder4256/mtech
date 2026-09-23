"""Turn noise into an exit state: the physics wrapper shared by every model.

CellSampler does everything around the generative draw, in exactly one copy
that every model family goes through, so that a comparison between families
measures the families and not differences in this code.

Two branches per particle.

  UNCOLLIDED, with probability exp(-chord): the particle crosses without
  scattering, so its exit is an exact function of its entry.  That is a
  Dirac delta a smooth generative model cannot represent, so it is done
  analytically and no model ever sees it.

  COLLIDED: encode and standardise the condition, ask the plugged-in
  Generator for a standardised exit state, then decode the six outputs back
  to (p, Omega, s).

The draw itself -- integrating the flow ODE, or any other family's method --
lives in generators/.  Until 23 Sept 2026 the ODE solve was in this file;
check_refactor.py proves that moving it out changed no output bit.

`Sampler` is kept as the old constructor, so evaluate.py and the two speed
scripts run unchanged.
"""

import numpy as np
import torch

from data import encode_conditions, mean_chord
from generators.cfm import CFM, NFE_PER_STEP, integrate   # noqa: F401  (re-exported)


def chord(W, H, xi, ox, oy):
    """Straight-line distance from entry (0, xi*H) to the wall along Omega."""
    y = xi * H
    with np.errstate(divide="ignore"):
        ty = np.where(oy > 0, (H - y) / np.where(oy > 0, oy, 1.0),
                      np.where(oy < 0, -y / np.where(oy < 0, oy, 1.0), np.inf))
    return np.minimum(W / ox, ty)


def decode_p(p, W, H):
    """Perimeter coordinate -> (x, y) and face (0 bottom, 1 right, 2 top, 3 left)."""
    p, W, H = np.broadcast_arrays(p, W, H)
    p = np.mod(p, 2.0 * (W + H))
    x, y = np.empty_like(p), np.empty_like(p)
    face = np.empty(p.shape, np.int8)
    edges = (p < W, (p >= W) & (p < W + H),
             (p >= W + H) & (p < 2 * W + H), p >= 2 * W + H)
    for i, m in enumerate(edges):
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


class CellSampler:
    """A plugged-in Generator + the shared normalisers -> physical exit states.

    Drop-in for gmc_solve: it offers .sample(), .reset(), .stats and
    .nfe_per_sample, which are the only things the transport loop uses.
    """

    def __init__(self, gen, ynorm, cnorm):
        self.gen = gen
        self.ynorm, self.cnorm = ynorm, cnorm
        self.nfe_per_sample = gen.nfe
        self.reset()

    def reset(self):
        """Zero the cost counters that evaluate.py reports."""
        self.stats = {"calls": 0, "particles": 0, "to_network": 0,
                      "nfe": 0, "clamped": 0}

    def sample(self, W, H, xi, ox_in, oy_in, seed=0, analytic_uncollided=True):
        """One exit state per entry state.  Arrays broadcast together."""
        W, H, xi = np.broadcast_arrays(np.atleast_1d(np.asarray(W, float)),
                                       np.asarray(H, float),
                                       np.asarray(xi, float))
        ox_in = np.broadcast_to(np.asarray(ox_in, float), W.shape)
        oy_in = np.broadcast_to(np.asarray(oy_in, float), W.shape)
        n = W.size
        rng = np.random.default_rng(seed)

        d = chord(W, H, xi, ox_in, oy_in)
        unc = (rng.random(n) < np.exp(-d) if analytic_uncollided
               else np.zeros(n, bool))
        p, s, dirs = np.empty(n), np.empty(n), np.empty((n, 3))

        # ---- analytic uncollided exits ---------------------------------
        if unc.any():
            m = unc
            ex, ey = ox_in[m] * d[m], xi[m] * H[m] + oy_in[m] * d[m]
            p[m] = np.where(np.isclose(ex, W[m]), W[m] + ey,
                            np.where(oy_in[m] > 0, 2 * W[m] + H[m] - ex, ex))
            s[m] = d[m]
            oz = np.sqrt(np.maximum(0.0, 1 - ox_in[m]**2 - oy_in[m]**2))
            oz *= np.where(rng.random(m.sum()) < 0.5, 1.0, -1.0)
            dirs[m, 0], dirs[m, 1], dirs[m, 2] = ox_in[m], oy_in[m], oz

        # ---- flow-matched collided exits -------------------------------
        m = ~unc
        n_flow = int(m.sum())
        if n_flow:
            c = torch.from_numpy(self.cnorm.transform(
                encode_conditions(W[m], H[m], xi[m], ox_in[m], oy_in[m])))
            g = torch.Generator().manual_seed(int(rng.integers(2**31)))
            y = self.gen.sample(c, g, cond={"W": W[m], "H": H[m], "xi": xi[m],
                                            "ox_in": ox_in[m], "oy_in": oy_in[m]})
            if torch.is_tensor(y):
                y = y.numpy()
            y = self.ynorm.inverse(np.asarray(y))

            frac = np.mod(np.arctan2(y[:, 1], y[:, 0]) / (2 * np.pi), 1.0)
            p[m] = frac * 2.0 * (W[m] + H[m])
            omega = y[:, 2:5]
            dirs[m] = omega / (np.linalg.norm(omega, axis=1, keepdims=True) + 1e-12)

            s_raw = mean_chord(W[m], H[m]) * np.exp(y[:, 5].astype(float))
            ex, ey, _ = decode_p(p[m], W[m], H[m])
            s_min = np.hypot(ex, ey - xi[m] * H[m])   # cannot be shorter
            s[m] = np.maximum(s_raw, s_min)
            self.stats["clamped"] += int((s_raw < s_min).sum())

        self.stats["calls"] += 1
        self.stats["particles"] += n
        self.stats["to_network"] += n_flow
        self.stats["nfe"] += n_flow * self.nfe_per_sample

        _, _, face = decode_p(p, W, H)
        return {"p": p, "dir": dirs, "s": s, "face": face, "uncollided": unc}


def Sampler(model, ynorm, cnorm, device="cpu", steps=5, solver="heun"):
    """The pre-pipeline constructor: a bare velocity-field network in, a
    CellSampler around a CFM generator out.  Kept so every script written
    before the split runs unchanged."""
    return CellSampler(CFM(model, steps, solver, device), ynorm, cnorm)
