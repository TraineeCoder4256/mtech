"""Checks that the solver is right, and measurements of what it costs.

    openmc_ref.py   run OpenMC on the same model; read its tallies
    compare.py      ours vs a reference, bin by bin (z-scores)
    analytic.py     problems with exact answers
    validate.py     the Stage 1 validation suite (all problems, all backends)
    ball_metrics.py backend vs exact walk, one ball at a time
    measure.py      how much time balls can save (big-ball share, R*)
    race.py         OpenMC vs our plain MC vs ball runs, timed
    gpu_bench.py    the network's cost per draw on a GPU, projected onto
                    the race
"""
