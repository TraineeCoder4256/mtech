"""Conditional flow matching: how the velocity field is trained.

THE TRICK
---------
We want to sample from p(exit state | cell condition), which we only know
through examples.  Flow matching turns that into a regression problem.

Imagine a straight line from a noise sample z to a real data sample y:

    x(t) = (1 - t) z + t y          t running 0 -> 1

Travel along it and your velocity is constant:

    dx/dt = y - z

So: pick a random pair (z, y), pick a random time t, work out where you
would be (x_t), and train the network to predict the velocity y - z that
belongs there.  That is the entire loss:

    L = E || v(x_t, t, c) - (y - z) ||^2

WHY THAT WORKS
--------------
Any individual x_t sits on many different (z, y) lines, so the network
cannot predict any one of them.  The mean-squared error makes it predict
the AVERAGE velocity over all lines through that point -- and that average
field is exactly the one whose flow carries the noise distribution onto the
data distribution.  We never have to know either density.

So at sampling time we start from noise and integrate v from t=0 to t=1,
and we land on a draw from p(y | c).  See gmc/sampler.py.

The straight-line path is the "optimal transport" or "rectified" choice.
It matters for cost: because the true paths are close to straight, the ODE
needs few steps, and step count is what inference costs.
"""

import copy

import torch


def cfm_loss(model, y, c):
    """One training step's loss, on a batch of targets y and conditions c.

    y and c are already encoded and standardised (see gmc/data.py).
    """
    n = y.shape[0]
    z = torch.randn_like(y)                     # the noise end of the line
    t = torch.rand(n, 1, device=y.device)       # a random point along it
    x_t = (1.0 - t) * z + t * y                 # where we are at time t
    v_target = y - z                            # the velocity along this line
    v_pred = model(x_t, t, c)                   # what the network thinks
    return torch.mean((v_pred - v_target) ** 2)


class EMA:
    """An exponential moving average of the weights, used for sampling.

    The training weights bounce around from batch to batch.  A running
    average of them is smoother and reliably samples better -- standard
    practice for flow and diffusion models, and it costs nothing at train
    time beyond one extra copy of the parameters.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # shadow <- decay * shadow + (1 - decay) * current
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(pm, alpha=1.0 - self.decay)
        # buffers (e.g. the fixed Fourier frequencies) are copied, not
        # averaged -- they are constants, not learned parameters
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)
