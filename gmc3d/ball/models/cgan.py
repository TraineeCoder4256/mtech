"""Family 4: a conditional generative adversarial network (cGAN).

The fourth family of the proposal.  It learns p(exit | ball) in the
encoded space of encoding.py, like the others: (mu, s | R) at interface
v1, (cos_n, mu, phi, s | R, g) at interface v2.

HOW IT DRAWS (one pass, nfe = 1).  A GENERATOR network takes the
condition and a handful of random numbers and turns them straight into one
sample.  No likelihood, no steps, no inverse: noise in, sample out.

HOW IT TRAINS.  Two networks play a game.  The DISCRIMINATOR sees real
walks and the generator's fakes, each with its condition, and learns to
tell them apart.  The generator learns to fool it.  When neither can do
better, the fakes are distributed like the real walks.  An analogy: a
forger and an inspector who both get better together; the forger is what
we keep.

  discriminator loss  log-loss of calling reals real and fakes fake
  generator loss      "non-saturating": the generator maximises the
                      chance its fakes are called real
  R1 penalty          the discriminator is also charged for being steep
                      around real walks (gamma / 2 * |grad D|^2).  It is
                      the standard cure for the GAN's best-known failure,
                      the two networks chasing each other in circles
                      instead of settling (Mescheder et al. 2018).

KNOWN WEAKNESS, and why it matters here.  A GAN is never told how likely a
real walk is, only whether a fake looks real, so it may drop rare parts of
the distribution ("mode dropping").  The weight a ball passes on in an
absorbing medium depends on the rare short-path tail, so the weight factor
in ball_metrics / gate2 is the number to watch.  There is also no
validation likelihood: the log shows the discriminator's validation loss
(near log 4 = 1.386 means it can no longer tell real from fake).

TRAINING LOOP.  A GAN does not fit training.fit_loop (two networks, two
optimisers, two losses), so fit() below is its own loop that copies the
shared conventions by hand: the seed first, then the network, then the
EMA and optimisers; one torch.Generator for batch and validation indices;
the same steps, batch, warm-up then cosine decay, EMA decay 0.999 (of the
whole module, so of the generator that draws), validation every val_every
steps on 65,536 held-out rows, and the time cap.  One step = one
discriminator update and one generator update on the same batch.

SETTINGS that are the family's own (the tuning budget, fairness.py):
Adam with betas (0.0, 0.99); generator learning rate 2.5e-4,
discriminator 1e-3 (four times faster), 2 discriminator steps per
generator step, R1 gamma 0.1, noise size 8.  The shared lr of 2e-3 is far
too large for a GAN.  Tuning trials used (3 of 4, each 1,500 steps on the
v1 data, scored by ball_metrics at R = 1.5, 5, 20 with 100k draws):

  1. lr 2e-4 both, gamma 1, 1 D step:     discriminator loss stuck at
     log 4 from step 500; s off +23% / +5% / -32%
  2. lr 1e-3 both, gamma 0.1, 1 D step:   same picture; s +7% / -9% / -35%
  3. G 2.5e-4, D 1e-3, gamma 0.1, 2 D steps (kept): s +19% / 0% / -36%
     at 1,500 steps; a check of the game showed the generator tracking
     the conditional means within about 0.1 standard deviation, with
     the discriminator's signal small but alive, so the full step budget
     with its decaying learning rate is the fair test.

SHAPE.  Shared idea with the others: a condition embedding and FiLM
blocks of width 64.  Generator 4 blocks, discriminator 3 (each with its
own embedding); together about the flow's size (fairness counts both, as
it counts the cVAE's encoder; a draw uses the generator only).
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fairness
from .base import Family
from .cvae import _Net
from .encoding import C_DIM, X_DIM
from .training import EMA, VAL_ROWS

WIDTH, G_DEPTH, D_DEPTH, EMB, NOISE = 64, 4, 3, 32, 8
LR_G, LR_D, BETAS, R1_GAMMA, N_CRITIC = 2.5e-4, 1e-3, (0.0, 0.99), 0.1, 2


def _embed(c_dim, emb):
    return nn.Sequential(nn.Linear(c_dim, emb), nn.SiLU(),
                         nn.Linear(emb, emb), nn.SiLU())


class GANNet(nn.Module):
    """Generator and discriminator in one module (saved together)."""

    def __init__(self, width=WIDTH, g_depth=G_DEPTH, d_depth=D_DEPTH,
                 emb=EMB, noise=NOISE, x_dim=X_DIM, c_dim=C_DIM):
        super().__init__()
        self.noise, self.x_dim = noise, x_dim
        self.g_embed = _embed(c_dim, emb)
        self.gen = _Net(noise, x_dim, width, g_depth, emb, False)
        self.d_embed = _embed(c_dim, emb)
        self.disc = _Net(x_dim, 1, width, d_depth, emb, False)

    def generate(self, z, c):
        return self.gen(z, self.g_embed(c))

    def critic(self, y, c):
        return self.disc(y, self.d_embed(c))[:, 0]

    def generator_params(self):
        return list(self.g_embed.parameters()) + list(self.gen.parameters())

    def discriminator_params(self):
        return list(self.d_embed.parameters()) + list(self.disc.parameters())


def _d_loss(net, y, c, fake):
    """Discriminator log-loss (reals called real, fakes called fake)."""
    return (F.softplus(-net.critic(y, c)).mean()
            + F.softplus(net.critic(fake, c)).mean())


class CGAN(Family):
    name = "cgan"
    nfe = 1
    loss_name = "D loss"
    CONFIG = {"width": WIDTH, "g_depth": G_DEPTH, "d_depth": D_DEPTH,
              "emb": EMB, "noise": NOISE}

    @classmethod
    def build(cls, config):
        return GANNet(config["width"], config["g_depth"], config["d_depth"],
                      config["emb"], config["noise"],
                      config.get("x_dim", X_DIM), config.get("c_dim", C_DIM))

    @staticmethod
    def loss(net, y, c):
        """The validation number: discriminator loss against fresh fakes
        (1.386 = log 4 means it cannot tell them apart)."""
        z = torch.randn(y.shape[0], net.noise, device=y.device)
        return _d_loss(net, y, c, net.generate(z, c))

    @classmethod
    def draw(cls, net, c, generator):
        z = torch.randn(c.shape[0], net.noise, generator=generator,
                        device=c.device)
        return net.generate(z, c)

    @classmethod
    def fit(cls, data, *, seed, steps=None, time_cap=fairness.TIME_CAP,
            log=print, interface=1):
        cfg = cls.config_for(interface)
        tr = dict(cls.TRAIN if interface == 1 else fairness.TRAIN_ANISO)
        if steps:
            tr["steps"] = steps
        steps, batch, warmup = tr["steps"], tr["batch"], tr["warmup"]
        torch.manual_seed(seed)
        y = torch.from_numpy(data["y_train"])
        c = torch.from_numpy(data["c_train"])
        yv, cv = torch.from_numpy(data["y_val"]), torch.from_numpy(data["c_val"])
        net = cls.build(cfg)
        n_params = sum(p.numel() for p in net.parameters())
        ema = EMA(net, tr["ema_decay"])
        opt_g = torch.optim.Adam(net.generator_params(), lr=LR_G,
                                 betas=BETAS)
        opt_d = torch.optim.Adam(net.discriminator_params(), lr=LR_D,
                                 betas=BETAS)
        sched = lambda s: min(1.0, (s + 1) / warmup) * \
            0.5 * (1 + np.cos(np.pi * min(1.0, s / steps)))
        sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, sched)
        sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, sched)
        gen = torch.Generator().manual_seed(seed)
        hist, smooth, t1, done = [], None, time.time(), 0
        for step in range(1, steps + 1):
            for _ in range(N_CRITIC):
                # discriminator step(s), each on its own batch, R1 on reals
                idx = torch.randint(0, y.shape[0], (batch,), generator=gen)
                yb, cb = y[idx], c[idx]
                z = torch.randn(batch, net.noise)
                with torch.no_grad():
                    fake = net.generate(z, cb)
                yr = yb.clone().requires_grad_(True)
                real_logit = net.critic(yr, cb)
                grad, = torch.autograd.grad(real_logit.sum(), yr,
                                            create_graph=True)
                d_loss = (F.softplus(-real_logit).mean()
                          + F.softplus(net.critic(fake, cb)).mean()
                          + 0.5 * R1_GAMMA * (grad * grad).sum(1).mean())
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.discriminator_params(),
                                               1.0)
                opt_d.step()
            # generator step (non-saturating)
            z = torch.randn(batch, net.noise)
            g_loss = F.softplus(-net.critic(net.generate(z, cb), cb)).mean()
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.generator_params(), 1.0)
            opt_g.step()
            sch_g.step()
            sch_d.step()
            ema.update(net)
            smooth = d_loss.item() if smooth is None else \
                0.98 * smooth + 0.02 * d_loss.item()
            done = step
            capped = time.time() - t1 > time_cap
            if step % tr["val_every"] == 0 or step == steps or capped:
                with torch.no_grad():
                    v = torch.randint(0, yv.shape[0], (VAL_ROWS,),
                                      generator=gen)
                    vloss = cls.loss(ema.shadow, yv[v], cv[v]).item()
                hist.append((step, smooth, vloss))
                log(f"step {step:6d}  train D loss {smooth:.4f}  G loss "
                    f"{g_loss.item():.4f}  val D loss (EMA G) {vloss:.4f}  "
                    f"{time.time() - t1:.0f} s")
            if capped:
                log(f"time cap of {time_cap:.0f} s reached at step {step}")
                break
        facts = {"steps": steps, "steps_done": done, "seed": seed,
                 "params": int(n_params), "train_seconds": time.time() - t1,
                 "history": hist, "loss_name": cls.loss_name, "train": tr,
                 "gan": {"lr_g": LR_G, "lr_d": LR_D, "betas": list(BETAS),
                         "r1_gamma": R1_GAMMA, "n_critic": N_CRITIC}}
        return ema.shadow.eval(), cfg, facts
