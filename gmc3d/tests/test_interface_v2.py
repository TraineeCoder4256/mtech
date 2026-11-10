"""Interface v2 (anisotropic balls): the spec, the training-law family and
the data generator agree with the exact walk."""
import numpy as np

from ball import data, interface as I, laws as LW
from core import scattering as SC


def test_v1_untouched_and_v2_sizes():
    assert I.INTERFACE_VERSION == 1 and (I.N_IN, I.N_OUT) == (1, 2)
    assert I.ANISO_INTERFACE_VERSION == 2
    assert (I.ANISO_N_IN, I.ANISO_N_OUT) == (1 + I.N_MOMENTS, 4)
    assert I.N_MOMENTS == LW.N_MOMENTS
    assert (I.G1_MIN, I.G1_MAX) == (LW.G_MIN, LW.G_MAX)


def test_family_covers_the_range_and_every_shape():
    laws, g = LW.family(3000, seed=2)
    assert g.shape == (3000, I.N_MOMENTS)
    assert g[:, 0].min() < -0.45 and g[:, 0].max() > 0.9
    assert np.all((g[:, 0] >= LW.G_MIN) & (g[:, 0] <= LW.G_MAX))
    assert np.all(np.abs(g) <= 1.0 + 1e-9)
    kinds = {int(k) for k, _ in laws}
    assert kinds == {SC.TAB, SC.HIST}


def test_held_out_laws_are_not_in_the_pool():
    laws, g = LW.family(3_000, seed=1)
    for name, law in LW.held_out().items():
        m = LW.moments(law)
        assert np.min(np.abs(g - m).max(axis=1)) > 1e-6, name


def test_rows_inside_interface_and_reproducible():
    d = data.make_aniso(5000, I.R_MIN, I.R_MAX, seed=4, pool=300)
    assert all(v == 0 for v in I.check_aniso_outputs(
        d["R"], d["cos_n"], d["mu"], d["phi"], d["s"]).values())
    assert d["g_pool"].shape == (300, I.N_MOMENTS)
    laws, g = LW.family(300, seed=4)
    assert np.array_equal(d["g_pool"], g) and d["law"].max() < 300
    assert data.data_hash(d) == data.data_hash(
        data.make_aniso(5000, I.R_MIN, I.R_MAX, seed=4, pool=300))
