"""Family 3: a conditional variational autoencoder (cVAE).

The family closest to published prior art for this exact job: Leonard,
Hoehlein and Westermann (2021, arXiv:2011.03082) used small cVAEs to learn
where a random walk leaves a sphere, for graphics.  Here it learns p(mu, s
| R) in the encoded space of encoding.py, like the other families.

HOW IT DRAWS (one pass, nfe = 1).  Pick a short random code z (LATENT
numbers from a standard Gaussian).  The DECODER, a network told R, turns z
into a small Gaussian bump in (mu, s) space -- a centre and a width for
each output -- and the sample is drawn from that bump.  Different codes
give bumps in different places, and together they cover the whole
distribution.  Only the decoder is used at draw time.

HOW IT TRAINS.  A second network, the ENCODER, looks at a real walk (and
R) and proposes which codes could have produced it.  The decoder is then
asked to reproduce the walk from such a code.  Two terms, summed (the
negative "evidence lower bound", ELBO, in nats per sample):

  * reconstruction: how unlikely the real walk is under the decoder's bump
  * KL: how far the encoder's proposed codes are from a standard Gaussian,
    so that at draw time, when there is no encoder, codes drawn from a
    plain Gaussian land where the decoder expects them.

An analogy: the encoder files each walk under a short index code, the
decoder learns to fetch a walk from its code, and the KL term keeps the
filing system tidy enough that a random code still finds a sensible walk.

The negative ELBO is an UPPER BOUND on the true negative log-likelihood,
so it can be read against the flow's NLL: the flow's value is exact, this
one is at least the cVAE's true value.

KNOWN WEAKNESS, and why it matters here.  Each code makes a Gaussian bump,
so the edges of the distribution are built from the tails of Gaussians;
VAEs tend to blur sharp edges and thin tails.  The weight a ball passes on
in an absorbing medium depends on the short-path tail, so ball_metrics'
weight factor is the number to watch.

SHAPE.  Shared condition embedding (as the other families), encoder of 3
FiLM blocks and decoder of 4, width 64, latent size 2 (the target is
2-D).  The decoder's output layer starts at zero, so the untrained model
draws a standard Gaussian.  Log-variances are clamped to [-12, 6] for
stability.  148,776 weights in all (+1.5% on the flow's 146,554); a draw
uses the embedding and decoder only (85,412).  Tuning trials: none yet
(budget fairness.TUNING_RUNS).
"""

import math

import torch
import torch.nn as nn

from .base import Family
from .encoding import C_DIM, X_DIM
from .nflow import FiLMBlock

WIDTH, ENC_DEPTH, DEC_DEPTH, EMB, LATENT = 64, 3, 4, 32, 2
LOGVAR_MIN, LOGVAR_MAX = -12.0, 6.0
LOG_2PI = math.log(2 * math.pi)


class _Net(nn.Module):
    """FiLM-conditioned MLP: (n, d_in) and an embedding -> (n, d_out)."""

    def __init__(self, d_in, d_out, width, depth, emb, zero_out):
        super().__init__()
        self.proj_in = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, emb)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, d_out)
        if zero_out:
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, emb):
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.proj_out(self.norm(h))


class CVAENet(nn.Module):

    def __init__(self, width=WIDTH, enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH,
                 emb=EMB, latent=LATENT):
        super().__init__()
        self.latent = latent
        self.embed = nn.Sequential(nn.Linear(C_DIM, emb), nn.SiLU(),
                                   nn.Linear(emb, emb), nn.SiLU())
        self.encoder = _Net(X_DIM, 2 * latent, width, enc_depth, emb, False)
        self.decoder = _Net(latent, 2 * X_DIM, width, dec_depth, emb, True)

    def decode(self, z, emb):
        mean, logvar = self.decoder(z, emb).chunk(2, dim=1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def neg_elbo(self, y, c):
        """Per-sample negative ELBO in nats (reparameterised, one z)."""
        emb = self.embed(c)
        qm, qlv = self.encoder(y, emb).chunk(2, dim=1)
        qlv = qlv.clamp(LOGVAR_MIN, LOGVAR_MAX)
        z = qm + torch.exp(0.5 * qlv) * torch.randn_like(qm)
        mean, logvar = self.decode(z, emb)
        rec = 0.5 * ((y - mean) ** 2 * torch.exp(-logvar) + logvar
                     + LOG_2PI).sum(1)
        kl = 0.5 * (qm * qm + torch.exp(qlv) - 1.0 - qlv).sum(1)
        return rec + kl


class CVAE(Family):
    name = "cvae"
    nfe = 1
    loss_name = "-ELBO"
    CONFIG = {"width": WIDTH, "enc_depth": ENC_DEPTH, "dec_depth": DEC_DEPTH,
              "emb": EMB, "latent": LATENT}

    @classmethod
    def build(cls, config):
        return CVAENet(config["width"], config["enc_depth"],
                       config["dec_depth"], config["emb"], config["latent"])

    @staticmethod
    def loss(net, y, c):
        return net.neg_elbo(y, c).mean()

    @classmethod
    def draw(cls, net, c, generator):
        n = c.shape[0]
        z = torch.randn(n, net.latent, generator=generator, device=c.device)
        mean, logvar = net.decode(z, net.embed(c))
        eps = torch.randn(n, X_DIM, generator=generator, device=c.device)
        return mean + torch.exp(0.5 * logvar) * eps
