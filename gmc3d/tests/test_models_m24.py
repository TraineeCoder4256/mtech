"""Drawing at speed (m2.4, ball/models/fast.py).

Without a GPU: the default engine is the original draw, bit for bit; the
noise/draw_from split gives the same numbers as draw(); compile on the CPU
stays within float32 rounding; GPU-only options refuse on the CPU; the
bucket ladder pads by at most sqrt(2).  With a GPU the CUDA tests also run
every option on every family.
"""

import math

import numpy as np
import pytest
import torch

from ball.models import REGISTRY, get
from ball.models.fast import Engine, bucket

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="no GPU on this machine")
DIMS = {1: (1, 2), 2: (13, 4)}


def _net(fam, interface):
    F = get(fam)
    torch.manual_seed(0)
    return F, F.build(F.config_for(interface))


def _c(n, c_dim, seed=0):
    return np.random.default_rng(seed).normal(size=(n, c_dim)) \
        .astype(np.float32)


@pytest.mark.parametrize("fam", sorted(REGISTRY))
@pytest.mark.parametrize("interface", [1, 2])
def test_split_draw_is_the_old_draw(fam, interface):
    """draw == draw_from(noise) on the same generator."""
    F, net = _net(fam, interface)
    c_dim, x_dim = DIMS[interface]
    c = torch.from_numpy(_c(300, c_dim))
    a = F.draw(net, c, torch.Generator().manual_seed(3))
    noise = F.noise(net, 300, torch.Generator().manual_seed(3), "cpu")
    b = F.draw_from(net, c, *noise)
    assert torch.equal(a, b) and a.shape == (300, x_dim)


@pytest.mark.parametrize("fam", sorted(REGISTRY))
def test_plain_engine_is_bit_identical(fam):
    F, net = _net(fam, 2)
    e = Engine(F, net, 13, 4, seed=9)
    c = _c(500, 13)
    ref = F.draw(net, torch.from_numpy(c),
                 torch.Generator().manual_seed(9)).numpy().astype(np.float64)
    assert e.plain and np.array_equal(e(c), ref)


def test_compile_on_cpu_matches():
    F, net = _net("cvae", 2)
    c = _c(2000, 13)
    a = Engine(F, net, 13, 4, seed=1)(c)
    b = Engine(F, net, 13, 4, seed=1, compile=True)(c)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_gpu_only_options_refuse_on_cpu():
    F, net = _net("cvae", 1)
    for kw in ({"graphs": True}, {"precision": "fp16"}):
        with pytest.raises(ValueError):
            Engine(F, net, 1, 2, **kw)
    with pytest.raises(ValueError):
        Engine(F, net, 1, 2, precision="bf16")


def test_bucket_ladder():
    assert [bucket(n) for n in (1, 2, 3, 4, 5, 7, 9, 17, 1000)] == \
        [1, 2, 3, 4, 6, 8, 12, 23, 1024]
    for n in range(1, 5000, 37):
        m = bucket(n)
        assert n <= m <= math.ceil(n * math.sqrt(2)) + 1
        assert bucket(m) == m


@needs_cuda
@pytest.mark.parametrize("fam", sorted(REGISTRY))
@pytest.mark.parametrize("opt", [{}, {"compile": True}, {"graphs": True},
                                 {"precision": "fp16"},
                                 {"graphs": True, "precision": "fp16"}])
def test_cuda_options(fam, opt):
    F, net = _net(fam, 2)
    e = Engine(F, net, 13, 4, device="cuda", **opt)
    e.warmup((1000, 3000))
    for n in (1, 999, 3000, 5000):
        y = e(_c(n, 13))
        assert y.shape == (n, 4) and np.isfinite(y).all()
    if opt.get("graphs"):
        assert e.recorded


@needs_cuda
@pytest.mark.parametrize("fam", sorted(REGISTRY))
def test_cuda_fp16_close_to_fp32(fam):
    F, net = _net(fam, 2)
    c = _c(4000, 13)
    a = Engine(F, net, 13, 4, device="cuda", seed=2)(c)
    b = Engine(F, net, 13, 4, device="cuda", seed=2, precision="fp16")(c)
    assert np.abs(a - b).max() < 0.1 * (np.abs(a).max() + 1)


def _trained(fam, interface, **kw):
    from ball import models
    try:
        iso = models.load("nflow", 0, name="network", **kw) \
            if interface == 2 else None
        return models.load(fam, 0, interface=interface, iso=iso, **kw)
    except FileNotFoundError:
        pytest.skip("no trained models on this machine")


def _laws():
    from core import scattering as SC
    from ball import laws as LW
    ang = SC.pack(list(LW.held_out().values())[:3])
    return ang


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_submit_collect_same_as_sample(device):
    """Two tickets in flight give what two sample() calls give."""
    R1 = np.linspace(1, 30, 3000)
    R2 = np.linspace(2, 9, 1700)
    a, b = _trained("cvae", 1, device=device), _trained("cvae", 1,
                                                         device=device)
    want = [a.sample(R1), a.sample(R2)]
    t1, t2 = b.submit(R1), b.submit(R2)
    got = [b.collect(t2), b.collect(t1)][::-1]
    for w, g in zip(want, got):
        assert all(np.array_equal(x, y) for x, y in zip(w, g))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_submit_aniso_same_as_sample_aniso(device):
    ang = _laws()
    R = np.linspace(1, 20, 2500)
    mats = (np.arange(R.size) % 3).astype(np.int64)
    a, b = _trained("cvae", 2, device=device), _trained("cvae", 2,
                                                         device=device)
    want = a.sample_aniso(R, mats, ang)
    got = b.collect(b.submit_aniso(R, mats, ang))
    assert all(np.array_equal(x, y) for x, y in zip(want, got))
    w_iso, g_iso = a.sample(R), b.collect(b.submit(R))
    assert all(np.array_equal(x, y) for x, y in zip(w_iso, g_iso))


@needs_cuda
def test_three_in_flight_refused():
    F, net = _net("cvae", 2)
    e = Engine(F, net, 13, 4, device="cuda")
    t1, t2 = e.submit(_c(100, 13)), e.submit(_c(100, 13))
    with pytest.raises(RuntimeError):
        e.submit(_c(100, 13))
    e.collect(t1)
    e.collect(e.submit(_c(100, 13)))
    e.collect(t2)



def test_sample_aniso_refuses_isotropic_rows():
    from core import scattering as SC
    from ball import laws as LW
    b = _trained("cvae", 2)
    ang = SC.pack([list(LW.held_out().values())[0], (SC.ISO, None)])
    with pytest.raises(ValueError):
        b.sample_aniso(np.array([3.0, 4.0]), np.array([0, 1]), ang)
