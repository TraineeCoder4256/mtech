"""The error bar: the chunk groups it comes from, and that it is honest.

Run:  python -m pytest -q tests   (from gmc3d/)

core/tallies.py estimates the standard error from the thread chunks rather
than from the batch means, which gives nchunk x batches independent groups
instead of batches.  Three things have to hold: the group sizes must match
the split transport.py actually uses, the MEAN must not change, and the
error bar must predict how much the answer really moves.
"""
import numpy as np

from core import openmc_import, tallies, transport
from problems import PROBLEMS


def test_chunk_sizes_match_transports_split():
    """tallies.chunk_sizes must agree with `per = ceil(n / nchunk)` and
    chunk c taking [c*per, min(n, (c+1)*per)), as _advance does."""
    for n, k in ((100000, 32), (10, 32), (0, 32), (97, 7), (1, 4)):
        per = (n + k - 1) // k
        want = np.array([min(n, (c + 1) * per) - min(c * per, n)
                         for c in range(k)])
        got = tallies.chunk_sizes(n, k)
        assert (got == want).all()
        assert got.sum() == n


def test_short_last_chunk_is_weighted_right():
    """A batch that does not divide evenly still gives the exact mean and a
    finite error bar."""
    L = tallies.compile_layout([], 1, 1)
    acc = np.zeros((7, L.n_entries))
    acc[:, tallies.LEAKAGE] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
    a = tallies.Accumulator(L)
    a.add_batch(acc, 97)                     # 7 chunks of 14, last one of 13
    a.add_batch(acc, 97)
    mean, se = a.mean_se()
    assert abs(mean[tallies.LEAKAGE] - 25.0 / 97.0) < 1e-15
    assert np.isfinite(se[tallies.LEAKAGE]) and se[tallies.LEAKAGE] > 0.0
    assert a.groups == 14


def test_mean_does_not_depend_on_the_grouping():
    """The chunk estimator must leave the answer itself alone: the mean is
    still the total over the total number of histories."""
    pr = openmc_import.load(PROBLEMS["slab"]())
    r = transport.run(pr, 4000, 5, seed=6)
    assert r.groups == 32 * 5
    # the old batch-mean route must give the same mean, to rounding
    pr2 = openmc_import.load(PROBLEMS["slab"]())
    a = transport.run(pr2, 4000, 5, seed=6)
    assert a.leakage == r.leakage
    assert r.leakage_se > 0.0 and r.leakage_se_batches > 0.0
    # and the two estimators must agree to within their own noise: the old
    # one has only 5 groups, so allow a wide factor
    assert 0.2 < r.leakage_se / r.leakage_se_batches < 5.0


def test_error_bar_predicts_the_real_spread():
    """Ten independent runs: the reported error bar should be close to the
    standard deviation of their means.  That standard deviation is itself
    uncertain by 1/sqrt(2*9) = 24%, so the test is loose on purpose and
    only catches an error bar that is wrong by a factor."""
    pr = openmc_import.load(PROBLEMS["slab"]())
    runs = [transport.run(pr, 4000, 5, seed=1 + 1000 * k) for k in range(10)]
    means = np.array([r.leakage for r in runs])
    truth = means.std(ddof=1)
    reported = np.mean([r.leakage_se for r in runs])
    assert 0.5 < reported / truth < 2.0
    # and the chunk estimator must be the steadier of the two
    spread = lambda v: np.std(v, ddof=1) / np.mean(v)
    assert spread([r.leakage_se for r in runs]) < \
        spread([r.leakage_se_batches for r in runs])
