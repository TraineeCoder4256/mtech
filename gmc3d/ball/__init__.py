"""The ball step: the walk inside a ball and the backends that replace it.

    walk.py     the exact walk (what everything here imitates)
    oracle.py   backend: run the exact walk live (the reference)
    table.py    backend: draw from walks pre-computed on a grid of radii
    network.py  backend: a trained generative network
    data.py     training data for network.py, made with walk.py

Every backend has the same face, which is all core/transport.py uses:

    name            a label for results
    r_min, r_max    the radii (mean free paths) it can answer for
    sample(R, seeds, idx) -> (mu, s)
                    one draw per entry of R: exit cosine to the outward
                    normal, and path length in mean free paths, for a walk
                    that scatters at least once.  seeds/idx are the waiting
                    particles' random streams, for backends that use them.
"""
