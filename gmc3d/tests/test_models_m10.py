"""The model side (ball/models/, m1.0): every registered family plugs in.

Fast: tiny data, a few training steps, no OpenMC.  Each test runs over
every family in the registry, so a new family is tested by being added.
"""

import numpy as np
import pytest
import torch

from ball import interface as I
from ball.models import REGISTRY, encoding, fairness, get, pipeline, store
from ball.models.encoding import C_DIM, X_DIM


def _data(n=512, seed=3):
    g = np.random.default_rng(seed)
    R = np.exp(g.uniform(np.log(I.R_MIN), np.log(I.R_MAX), n))
    mu = g.uniform(0.01, 0.99, n)
    s = R * (1.0 + g.exponential(1.0, n))
    raw = {"R": R, "mu": mu, "s": s, "k": np.ones(n)}
    d = encoding.prepare(raw, n - 128, I.R_MIN, I.R_MAX)
    d["seed"] = seed
    return d


@pytest.mark.parametrize("family", list(REGISTRY))
def test_family_face(family):
    F = get(family)
    assert F.name == family and F.nfe >= 1
    net = F.build(dict(F.CONFIG))
    c = torch.zeros(7, C_DIM)
    y = torch.zeros(7, X_DIM)
    loss = F.loss(net, y, c)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    a = F.draw(net.eval(), c, torch.Generator().manual_seed(5))
    b = F.draw(net, c, torch.Generator().manual_seed(5))
    assert a.shape == (7, X_DIM) and torch.equal(a, b)


@pytest.mark.parametrize("family", list(REGISTRY))
def test_fairness_budget(family):
    assert fairness.check_size(pipeline.count_params(family)) is None


@pytest.mark.parametrize("family", list(REGISTRY))
def test_train_save_load_draw(family, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SHARED_DIR", tmp_path / "shared")
    F, d = get(family), _data()
    net, cfg, facts = F.fit(d, seed=0, steps=3, log=lambda *a: None)
    assert facts["steps_done"] == 3
    out = tmp_path / store.folder(family, 0)
    m = store.save(out, F, cfg, net, d, facts)
    assert m["interface"] == I.INTERFACE_VERSION and m["family"] == family
    assert (tmp_path / "shared" / out.name / "manifest.json").exists()
    be = pipeline.load(family, 0, out_dir=out)
    R = np.array([1.5, 5.0, 29.0])
    mu, s = be.sample(R)
    assert not any(I.check_outputs(R, mu, s).values())


def test_encoding_round_trip():
    R = np.array([1.0, 3.0, 30.0])
    mu = np.array([0.2, 0.5, 0.9])
    s = np.array([1.5, 7.0, 400.0])
    c, y = encoding.encode(R, mu, s)
    mu2, s2 = encoding.decode(R, y)
    assert np.allclose(mu2, mu) and np.allclose(s2, s)


def test_other_interface_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SHARED_DIR", tmp_path / "shared")
    F, d = get("nflow"), _data()
    net, cfg, facts = F.fit(d, seed=0, steps=1, log=lambda *a: None)
    store.save(tmp_path / "m", F, cfg, net, d, facts, shared=False)
    monkeypatch.setattr(I, "INTERFACE_VERSION", I.INTERFACE_VERSION + 1)
    with pytest.raises(store.NotSupported):
        store.read(tmp_path / "m")
