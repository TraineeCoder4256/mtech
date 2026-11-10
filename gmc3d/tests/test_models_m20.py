"""The model side at interface v2 (m2.0, anisotropic balls).

Fast: a few thousand real anisotropic walks from a pool of 40 laws, a few
training steps, no OpenMC.  Every family in the registry is tested, so a
new family is tested at v2 by being added.
"""

import numpy as np
import pytest
import torch

from ball import data as D
from ball import interface as I
from ball import laws as LW
from ball.models import REGISTRY, encoding as E, fairness, get, pipeline, \
    store
from ball.models.backend import law_moments
from ball.models.nflow import masks
from core import scattering as SC

V2 = I.ANISO_INTERFACE_VERSION


class _Iso:
    name = "iso-stub"

    def sample(self, R, seeds=None, idx=None):
        return np.full(len(R), 0.5), 2.0 * np.asarray(R)


def _data(n=3000, seed=3):
    raw = D.make_aniso(n, I.R_MIN, I.R_MAX, seed=seed, pool=40)
    iso = pipeline.iso_rows(400, I.R_MIN, I.R_MAX, seed=seed, cache=False)
    d = E.prepare_aniso([(raw, n - 500), (iso, 300)], n - 200, I.R_MIN,
                        I.R_MAX, [D.data_hash(raw), D.data_hash(iso)])
    d["seed"] = seed
    return d


def test_masks():
    assert masks(2, 4) == [([0], [1]), ([1], [0])] * 2
    m = masks(4, 6)
    bent = sum((b for b, _ in m), [])
    assert all(bent.count(i) == 3 for i in range(4))
    assert all(sorted(b + k) == [0, 1, 2, 3] for b, k in m)


def test_encoding_round_trip():
    g = np.random.default_rng(1)
    R = np.exp(g.uniform(0, np.log(30), 1000))
    cn, mu = g.uniform(-0.99, 0.99, 1000), g.uniform(0.01, 0.99, 1000)
    phi, s = g.uniform(0.01, 3.1, 1000), R * (1 + g.exponential(1, 1000))
    back = E.decode_aniso(R, E.encode_aniso(R, cn, mu, phi, s))
    for a, b in zip((cn, mu, phi, s), back):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12)


def test_law_moments_from_packed_rows():
    held = list(LW.held_out().values())
    ang = SC.pack(held)
    for row, law in zip(ang, held):
        assert np.allclose(law_moments(row), LW.moments(law), atol=1e-12)


@pytest.mark.parametrize("family", list(REGISTRY))
def test_v2_size_within_budget(family):
    F = get(family)
    n = sum(p.numel() for p in F.build(F.config_for(V2)).parameters())
    assert fairness.check_size(n, V2) is None


@pytest.mark.parametrize("family", list(REGISTRY))
def test_v2_train_save_load_draw(family, tmp_path):
    F = get(family)
    d = _data()
    net, cfg, facts = F.fit(d, seed=0, steps=20, log=lambda *a: None,
                            interface=V2)
    out = tmp_path / store.folder(family, 0, V2)
    store.save(out, F, cfg, net, d, facts, shared=False, interface=V2)
    be = pipeline.load(family, 0, out_dir=out, interface=V2, iso=_Iso())
    assert be.aniso and be.describe()["interface"] == V2
    ang = SC.pack(list(LW.held_out().values()))
    R = np.linspace(1.0, 30.0, 700)
    mats = np.arange(700) % ang.shape[0]
    cn, mu, phi, s = be.sample_aniso(R, mats, ang, None, np.arange(700))
    assert not any(I.check_aniso_outputs(R, cn, mu, phi, s).values())
    mu1, s1 = be.sample(R[:5])                  # isotropic: the v1 model
    assert np.all(mu1 == 0.5)
    # a v2 model is never loaded as v1, nor a v1 one as v2
    with pytest.raises(store.NotSupported):
        store.read(out)
    with pytest.raises(store.NotSupported):
        pipeline.load(family, 0, out_dir=out)
