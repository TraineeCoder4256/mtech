"""v1.6.2: the model interface (ball/interface.py) matches the exact walk."""
import numpy as np

from ball import data, interface as I


def test_version_and_sizes():
    assert I.INTERFACE_VERSION == 1
    assert (I.N_IN, I.N_OUT) == (1, 2)
    assert (I.R_MIN, I.R_MAX) == (1.0, 30.0)


def test_exact_walk_inside_interface():
    d = data.make(20000, I.R_MIN, I.R_MAX, seed=3)
    assert all(v == 0 for v in I.check_outputs(d["R"], d["mu"], d["s"]).values())


def test_check_outputs_flags_bad_draws():
    R = np.array([2.0, 2.0, 2.0, 40.0])
    mu = np.array([0.5, 1.0, 0.5, 0.5])
    s = np.array([3.0, 3.0, 1.5, 50.0])
    c = I.check_outputs(R, mu, s)
    assert c == {"R_out_of_range": 1, "mu_out_of_range": 1,
                 "s_not_above_R": 1, "non_finite": 0}
