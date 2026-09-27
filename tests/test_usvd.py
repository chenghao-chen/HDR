"""
Contract tests for the unrolled denoiser (USVD).

Two classes of thing are pinned here. First the shapes and the tuple
contract, so USVD stays interchangeable with the MoE and FiLM models in the
trainers and the evaluation harness. Second, and more importantly, that no
parameter is born without a gradient: this project has twice shipped a
model whose final layer was initialised to exactly zero, which silently
zeroes the gradient of everything feeding it. USVD has two such layers by
construction (the fuse projection and the FiLM head), so the regression
guard belongs here rather than in a comment.
"""
from __future__ import annotations

import pytest
import torch

from HDR_model_hybrid_Teacher import build_denoiser

pytest.importorskip("einops", reason="USVD needs einops")


def _kwargs(**over):
    kw = dict(in_channels=4, out_channels=4, upsample=False,
              num_iters=2, d_model=64, patch_size=8, top_k=4)
    kw.update(over)
    return kw


def _inputs(b=2, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(b, 4, h, w, generator=g)
    return x, torch.rand(b, 1, h, w, generator=g)


def test_raw_mode_preserves_shape():
    model = build_denoiser("usvd", **_kwargs()).eval()
    x, snr = _inputs()
    with torch.no_grad():
        out, experts, gates = model(x, snr)
    assert out.shape == x.shape
    assert experts.shape == (x.shape[0], 1, *x.shape[1:])
    assert gates.shape == (x.shape[0], 1, *x.shape[2:])


def test_jdd_mode_upsamples_to_sensor_resolution():
    model = build_denoiser("usvd", **_kwargs(out_channels=3, upsample=True)).eval()
    x, snr = _inputs()
    with torch.no_grad():
        out, experts, gates = model(x, snr)
    b, _, h, w = x.shape
    assert out.shape == (b, 3, 2 * h, 2 * w)
    assert experts.shape == (b, 1, 3, 2 * h, 2 * w)
    assert gates.shape == (b, 1, 2 * h, 2 * w)


@pytest.mark.parametrize("h,w", [(32, 48), (24, 24), (40, 32)])
def test_non_multiple_of_patch_size_is_handled(h, w):
    """Non-aligned inputs are padded internally and cropped back."""
    model = build_denoiser("usvd", **_kwargs()).eval()
    x, snr = _inputs(1, h, w)
    with torch.no_grad():
        out, _, _ = model(x, snr)
    assert out.shape == x.shape


def test_every_parameter_receives_gradient():
    model = build_denoiser("usvd", **_kwargs()).train()
    x, snr = _inputs()
    out, _, _ = model(x, snr)
    out.square().mean().backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.norm()) == 0.0)]
    assert not dead, f"parameters with no gradient at init: {dead}"


def test_step_sizes_stay_positive():
    """alpha is stored raw and softplus'd, so no optimiser step can flip it."""
    import torch.nn.functional as F
    model = build_denoiser("usvd", **_kwargs(num_iters=4))
    with torch.no_grad():
        model.core.alpha_raw.fill_(-50.0)
    assert bool((F.softplus(model.core.alpha_raw) > 0).all())


def test_sigma_map_is_strictly_positive():
    """W = 1/sigma^2 divides by this, so a zero would produce infinities."""
    model = build_denoiser("usvd", **_kwargs()).eval()
    x, _ = _inputs()
    with torch.no_grad():
        sigma = model.core.noise_estimator(x)
    assert bool((sigma > 0).all())
    assert bool(torch.isfinite(sigma).all())


def test_shared_weights_ties_every_iteration():
    shared = build_denoiser("usvd", **_kwargs(num_iters=4, share_weights=True))
    first = shared.core.denoisers[0]
    assert all(d is first for d in shared.core.denoisers)
    unshared = build_denoiser("usvd", **_kwargs(num_iters=4, share_weights=False))
    assert sum(p.numel() for p in unshared.parameters()) > \
           sum(p.numel() for p in shared.parameters())


def test_chunked_attention_matches_unchunked():
    """chunk_size is a memory knob and must not change the result."""
    torch.manual_seed(0)
    big = build_denoiser("usvd", **_kwargs(chunk_size=4096)).eval()
    small = build_denoiser("usvd", **_kwargs(chunk_size=3)).eval()
    small.load_state_dict(big.state_dict())
    x, snr = _inputs(1, 32, 32, seed=3)
    with torch.no_grad():
        a, _, _ = big(x, snr)
        b, _, _ = small(x, snr)
    assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_round_trip_rebuilds_the_same_model():
    import os
    import tempfile
    from hdr_eval.checkpoints import load_model_from_checkpoint

    kw = _kwargs()
    model = build_denoiser("usvd", **kw)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "usvd.pth")
        torch.save({"model_state_dict": model.state_dict(), "model_kwargs": kw,
                    "mode": "usvd", "num_experts": 1}, path)
        loaded = load_model_from_checkpoint(path, torch.device("cpu"))
        loaded = loaded[0] if isinstance(loaded, tuple) else loaded
    assert sum(p.numel() for p in loaded.parameters()) == \
           sum(p.numel() for p in model.parameters())


def test_trunk_kwargs_are_accepted_and_ignored():
    """The shared trainers pass trunk hyperparameters to every mode."""
    model = build_denoiser("usvd", **_kwargs(), dim=32, num_blocks=[4, 4, 4, 4],
                           heads=[1, 2, 4, 8], se_reduction=8, expert_blocks=2)
    assert "dim" in model._ignored_kwargs
    x, snr = _inputs()
    with torch.no_grad():
        out, _, _ = model.eval()(x, snr)
    assert out.shape == x.shape


def test_eval_mode_clamps_to_unit_range():
    model = build_denoiser("usvd", **_kwargs()).eval()
    x, snr = _inputs()
    with torch.no_grad():
        out, experts, _ = model(x, snr)
    assert float(out.max()) <= 1.0
    assert float(experts.max()) <= 1.0
    assert float(out.min()) > 0.0


def test_gradient_checkpointing_matches_and_still_trains():
    """checkpoint=True is a memory knob: same values, same trainable params.

    It is off by default, so without this the path would only ever run in a
    job -- which is where it first failed, on a missing import.
    """
    torch.manual_seed(0)
    plain = build_denoiser("usvd", **_kwargs(num_iters=2)).train()
    ckpt = build_denoiser("usvd", **_kwargs(num_iters=2, checkpoint=True)).train()
    ckpt.load_state_dict(plain.state_dict())

    x, snr = _inputs(1, 32, 32, seed=7)
    a, _, _ = plain(x, snr)
    b, _, _ = ckpt(x, snr)
    assert torch.allclose(a, b, atol=1e-6)

    b.square().mean().backward()
    dead = [n for n, p in ckpt.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.norm()) == 0.0)]
    assert not dead, f"no gradient under checkpointing: {dead}"
