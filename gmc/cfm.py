"""Conditional flow matching objective (the paper's stated loss) + EMA.

Optimal-transport (rectified) interpolation:

    z ~ N(0, I),  t ~ U(0, 1)
    x_t = (1 - t) z + t y          =>   dx_t/dt = y - z

    L(theta) = E || v_theta(x_t, t, c) - (y - z) ||^2

which is the paper's  L = E||v_theta(x_t, t) - x_dot_t||^2  with the OT path
made explicit.  At inference, integrating dx/dt = v_theta from t=0 (noise)
to t=1 produces a sample of y | c.
"""

import copy

import torch


def cfm_loss(model, y, c):
    """One CFM step on a batch of normalised targets y and conditions c."""
    n = y.shape[0]
    z = torch.randn_like(y)
    t = torch.rand(n, 1, device=y.device)
    x_t = (1.0 - t) * z + t * y
    v_target = y - z
    v_pred = model(x_t, t, c)
    return torch.mean((v_pred - v_target) ** 2)


class EMA:
    """Exponential moving average of model weights; sampling uses the EMA
    copy, which is standard practice for flow/diffusion models and costs
    nothing at train time."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(pm, alpha=1.0 - self.decay)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)
