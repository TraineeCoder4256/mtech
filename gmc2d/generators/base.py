"""The plug-in interface every generative model implements.

Every model tested here does one job: given a cell and an entry condition,
turn random noise into one exit state.  What differs between families --
flow matching, diffusion, a normalizing flow, a VAE, a GAN, a distilled
one-step model -- is only HOW noise becomes the exit state, and how the model
is trained.  Everything around that draw must be identical for every model,
or a comparison measures differences in the surrounding code instead of
differences between models.

So the line is drawn here, and it is thin on purpose:

  IN   c    (n, 5) conditions, already encoded and standardised by data.py
  OUT  y    (n, 6) exit states in the same encoded, standardised space

A model never sees W, H, a perimeter coordinate or a path length.  The
circle encoding of the exit point, log(s / chord), the analytic uncollided
branch and the decode back to physics all live in sampler.CellSampler, in
exactly one copy, shared by every model.  A family cannot quietly handle
them differently, because it never touches them.

What the pipeline asks of a family:

  nfe         network evaluations per sample -- the model's cost class.
              10 for the shipped flow matching (heun x 5), 1 for anything
              that draws in a single pass.  The speed analysis showed the
              cost per network call is what decides the crossover with Monte
              Carlo, so this number goes into every result.
  fit()       train on the shared dataset.  Each family owns its training
              loop, because they genuinely differ (a regression, a
              two-player game, a variational bound); what is fixed is the
              data going in and the checkpoint coming out.
  sample()    the draw.  It is called only for particles that scattered at
              least once -- the uncollided ones never reach a model.
  save/load   one directory per trained model.

`cond` in sample() carries the same conditions in physical units (W, H, xi,
Omega_in).  Learned models must ignore it: it exists for the two REFERENCE
generators (oracle.py, free.py), which are goalposts rather than models.
"""


class Generator:
    name = "base"
    nfe = 0                 # network evaluations per sample
    trainable = True        # False for the reference generators

    def fit(self, data, out_dir, time_cap):
        """Train on data.load_dataset() output; write logs into out_dir.
        Stop early if training exceeds time_cap seconds.  Returns a dict of
        facts about the training run (steps done, wall time, final losses)
        that is saved with the checkpoint."""
        raise NotImplementedError

    def sample(self, c, generator, cond=None):
        """(n, 5) standardised conditions -> (n, 6) standardised targets.

        c is a float32 torch tensor on the CPU; generator is a seeded
        torch.Generator, and every random number the draw uses must come from
        it, so that a run is reproducible from its seed.  Return a torch
        tensor or a NumPy array of shape (n, 6)."""
        raise NotImplementedError

    def save(self, out_dir):
        raise NotImplementedError

    @classmethod
    def load(cls, out_dir, **settings):
        raise NotImplementedError

    def describe(self):
        """Settings that identify this model in a results file."""
        return {"name": self.name, "nfe": self.nfe}
