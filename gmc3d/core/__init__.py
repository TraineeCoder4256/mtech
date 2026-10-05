"""The solver.  Each file uses only the ones above it in this list.

    surfaces.py        the shapes: value, ray distance, nearest distance
    geometry.py        cells, universes, lattices: locate, distance field
    openmc_import.py   an openmc.Model -> the arrays the solver runs on
    rng.py             one random stream per particle
    tallies.py         scoring and error bars
    transport.py       plain Monte Carlo + ball steps (the main loop)
    output.py          JSON and VTK files
"""
