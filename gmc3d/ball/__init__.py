"""The ball step: the walk inside a ball and the backends that replace it.

    walk.py     the exact walk (what everything here imitates)
    oracle.py   backend: run the exact walk live (the reference)
    table.py    backend: draw from walks pre-computed on a grid of radii
    network.py  backend: a trained generative network (the flow; the
                name older code imports -- the code is in models/)
    models/     the model side: every generative family behind one
                plug-in face, the shared training loop, fairness rules,
                trained-model store, side-by-side comparison (m1.0)
    data.py     training data for every family, made with walk.py
                (v2 rows too: make_aniso, one law per row)
    laws.py     the scattering laws v2 models train on, and the
                held-out test laws (interface v2)
    interface.py  the model interface (inputs, outputs, ranges, version);
                owned by Innovation, read by the model side (v1.6.2)
    profile.py  not a backend: where in a ball its flux lies, for mesh
                tallies (the 'profile' mesh rule, v1.1)

Every backend has the same face, which is all core/transport.py uses:

    name            a label for results
    r_min, r_max    the radii (mean free paths) it can answer for
    sample(R, seeds, idx) -> (mu, s)
                    one draw per entry of R: exit cosine to the outward
                    normal, and path length in mean free paths, for a walk
                    that scatters at least once.  seeds/idx are the waiting
                    particles' random streams, for backends that use them.

and, since v2.1, a backend that also walks balls in anisotropic materials
(only the oracle, until interface v2 has trained models) adds

    aniso = True
    sample_aniso(R, mats, ang, seeds, idx) -> (cos_n, mu, phi, s)
                    a walk in material mats[j], whose scattering law is
                    row mats[j] of ang (core/scattering.py), started along
                    a known direction u0: cos_n of the exit normal to u0,
                    mu as above, phi the exit azimuth about the normal
                    (walk.py), s as above.  Balls in isotropic materials
                    still use sample().
"""
