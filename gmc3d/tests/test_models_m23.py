"""Training on a device (m2.3).

On a machine without a GPU these check that the device code path is the
same arithmetic as the original CPU loop (weights equal to the last bit),
and that the speed options refuse or fall back cleanly.  On a machine with
a GPU the CUDA tests also run: every family trains a few steps on the GPU
in each precision and comes back as a CPU model that draws.
"""

import numpy as np
import pytest
import torch

from ball.models import REGISTRY, get, training

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="no GPU on this machine")


def _data(x_dim, c_dim, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, c_dim)).astype(np.float32)
    y = (rng.normal(size=(n, x_dim)) + 0.3 * c[:, :1]).astype(np.float32)
    return {"y_train": y, "c_train": c, "y_val": y[:500], "c_val": c[:500]}


def _fit(fam, interface=1, steps=6, **kw):
    x, c = (2, 1) if interface == 1 else (4, 13)
    net, cfg, facts = get(fam).fit(_data(x, c), seed=5, steps=steps,
                                   log=lambda *a: None, interface=interface,
                                   **kw)
    return net, facts


@pytest.mark.parametrize("fam", sorted(REGISTRY))
@pytest.mark.parametrize("interface", [1, 2])
def test_device_path_is_the_cpu_loop(fam, interface, monkeypatch):
    """The GPU loop run on the CPU gives the original loop's weights."""
    plain, _ = _fit(fam, interface)
    monkeypatch.setattr(training.Device, "plain",
                        property(lambda self: False))
    dev, facts = _fit(fam, interface)
    for k, v in plain.state_dict().items():
        assert torch.equal(v, dev.state_dict()[k]), k
    if fam != "cgan":
        assert facts["device"] == "cpu"


def test_options_are_checked():
    with pytest.raises(ValueError):
        training.Device("cpu", precision="bf16")
    if not CUDA:
        with pytest.raises(RuntimeError):
            training.Device("cuda")
    said = []
    dv = training.Device("cpu", precision="fp16", log=said.append)
    assert not dv.amp and said           # fp16 needs a GPU: says so, fp32


def test_gan_refuses_fp16_quietly():
    said = []
    get("cgan").fit(_data(2, 1), seed=0, steps=2, log=said.append,
                    precision="fp16")
    assert any("fp16" in s for s in said)


@needs_cuda
@pytest.mark.parametrize("fam", sorted(REGISTRY))
@pytest.mark.parametrize("precision", ["fp32", "tf32", "fp16"])
def test_cuda_training(fam, precision):
    net, facts = _fit(fam, 2, steps=20, device="cuda", precision=precision)
    assert next(net.parameters()).device.type == "cpu"
    assert facts["device"].startswith("cuda") and facts["rows_on_gpu"]
    assert all(np.isfinite(h[1]) and np.isfinite(h[2])
               for h in facts["history"])
    c = torch.zeros(64, 13)
    y = get(fam).draw(net, c, torch.Generator().manual_seed(0))
    assert y.shape == (64, 4) and torch.isfinite(y).all()


@needs_cuda
def test_cuda_training_compiled():
    net, facts = _fit("cvae", 2, steps=20, device="cuda", compile=True)
    assert facts["compile"] and np.isfinite(facts["history"][-1][2])
