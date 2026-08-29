"""
Unit tests for the building blocks in HDR_model_hybrid_Teacher.py.

Scope: the *components*, not the full networks (TransUNet_Teacher_HDR and
MoEDenoiser are covered elsewhere):

    estimate_local_snr_map   the canonical routing signal, imported by BOTH
                             train_A100_MoE_two_phase.py and
                             test_dual_MoE_two_phase.py — if its semantics
                             drift, train/eval routing silently diverges.
    SqueezeExcite            channel recalibration gate.
    ResidualConvBlock        the CNN workhorse (+ SEResidualBlock alias).
    HeavyExposhare           cross-Bayer-plane feature mixing.
    NoiseGate                the MoE router (zero-init -> uniform routing).
    ExpertHead               the per-expert 4x upsampling decoder head.

Everything runs on CPU at tiny sizes.
"""

import pytest
import torch
import torch.nn.functional as F

from helpers import (
    assert_finite,
    assert_in_range,
    assert_shape,
    packed_bayer,
    packed_bayer_3d,
)

from HDR_model_hybrid_Teacher import (
    ExpertHead,
    HeavyExposhare,
    NoiseGate,
    ResidualConvBlock,
    SEResidualBlock,
    SqueezeExcite,
    estimate_local_snr_map,
)


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — conftest/helpers are shared)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_snr_map(x, window_size=5, eps=1e-5):
    """
    Slow, obviously-correct re-implementation of estimate_local_snr_map for a
    batched 4-D input. Written with explicit zero padding and an explicit
    /window_size**2 divisor, i.e. it reproduces avg_pool2d's default
    count_include_pad=True semantics (border windows are averaged over the
    full window including the zero pad, which biases the border means down).
    Only used to cross-check the real implementation on a tiny tensor.
    """
    B, C, H, W = x.shape
    pad = window_size // 2
    xp = F.pad(x, (pad, pad, pad, pad))
    n = float(window_size * window_size)
    out_h = H + 2 * pad - window_size + 1
    out_w = W + 2 * pad - window_size + 1
    snr = torch.zeros(B, C, out_h, out_w, dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            win = xp[:, :, i:i + window_size, j:j + window_size]
            mean = win.sum(dim=(2, 3)) / n
            sq_mean = (win * win).sum(dim=(2, 3)) / n
            var = torch.clamp(sq_mean - mean * mean, min=0.0)
            snr[:, :, i, j] = mean / torch.sqrt(var + eps)
    snr = snr.mean(dim=1, keepdim=True)
    return snr / (snr.amax(dim=(2, 3), keepdim=True) + eps)


def _split_contrast_batch(h=32, w=32):
    """
    Batch of two packed tensors with wildly different contrast:
      image 0: nearly flat (tiny noise)  -> very large raw mean/std ratios
      image 1: strongly textured         -> small raw mean/std ratios
    Used to prove the normalisation is PER IMAGE, not per batch.
    """
    g = torch.Generator().manual_seed(7)
    flat = 0.9 + 0.001 * torch.rand((1, 4, h, w), generator=g)
    rough = (0.5 + 0.4 * (torch.rand((1, 4, h, w), generator=g) - 0.5) * 2).clamp(0, 1)
    return torch.cat([flat, rough], dim=0)


def _zero_out(conv):
    with torch.no_grad():
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()


# ═════════════════════════════════════════════════════════════════════════════
# estimate_local_snr_map — the canonical routing signal
# ═════════════════════════════════════════════════════════════════════════════

def test_snr_map_batched_shape_range_and_unit_max():
    """
    [B,4,H,W] -> [B,1,H,W], values in [0,1], and the per-image max sits at 1.0.

    This is the exact tensor handed to NoiseGate alongside the packed Bayer
    input (they are concatenated on dim=1), so both the channel count of 1 and
    the preserved spatial size are load-bearing. The max is 1.0 only up to the
    eps guard in `spatial_snr / (image_max + eps)` — a deliberate
    divide-by-zero guard that costs ~5e-6 of headroom, not a defect.
    """
    x = packed_bayer(batch=2, h=32, w=32, seed=3)
    snr = estimate_local_snr_map(x, window_size=5)

    assert_shape(snr, (2, 1, 32, 32), "snr_map")
    assert snr.dtype == x.dtype
    assert_finite(snr, "snr_map")
    assert_in_range(snr, 0.0, 1.0, "snr_map", atol=0.0)
    per_image_max = snr.amax(dim=(1, 2, 3))
    assert torch.allclose(per_image_max, torch.ones(2), atol=1e-4), per_image_max


def test_snr_map_unbatched_three_dim_branch():
    """
    The dim()==3 branch: [C,H,W] -> [1,H,W] (no phantom batch axis left over).

    MobileHDRDataset yields unbatched [4,h,w] tensors, and
    test_dual_MoE_two_phase's per-patch inference path feeds single images
    straight in, so the unbatched contract has real callers.
    """
    x3 = packed_bayer_3d(h=24, w=16, seed=5)
    snr3 = estimate_local_snr_map(x3, window_size=5)

    assert snr3.dim() == 3
    assert_shape(snr3, (1, 24, 16), "snr_map_3d")
    assert_finite(snr3, "snr_map_3d")
    assert_in_range(snr3, 0.0, 1.0, "snr_map_3d", atol=0.0)

    # Same numbers as the batched call on the same content.
    snr4 = estimate_local_snr_map(x3.unsqueeze(0), window_size=5)
    assert torch.allclose(snr3, snr4[0], atol=0.0)


@pytest.mark.parametrize("window_size", [1, 3, 5, 7, 9])
def test_snr_map_odd_window_preserves_spatial_dims(window_size):
    """
    Every odd window keeps H,W exactly (pad = w//2 on both sides).

    window_size=1 is the degenerate edge: local variance is identically 0, so
    the whole map collapses onto x/max(x) via the eps floor — it must still be
    finite and in range rather than exploding.
    """
    x = packed_bayer(batch=1, h=20, w=28, seed=11)
    snr = estimate_local_snr_map(x, window_size=window_size)
    assert_shape(snr, (1, 1, 20, 28), f"snr_map(ws={window_size})")
    assert_finite(snr, "snr_map")
    assert_in_range(snr, 0.0, 1.0, "snr_map", atol=0.0)


@pytest.mark.xfail(
    reason="BUG: even window_size returns H+1,W+1 — pad=w//2 under-pads by one, "
           "breaking the documented [B,C,H,W]->[B,1,H,W] contract",
    strict=False,
)
@pytest.mark.parametrize("window_size", [2, 4, 6])
def test_snr_map_even_window_preserves_spatial_dims(window_size):
    """
    The documented contract is x:[B,C,H,W] -> [B,1,H,W] for any window_size,
    but `pad = window_size // 2` is one short of what an even kernel needs, so
    avg_pool2d emits H + 2*(w//2) - w + 1 = H+1 rows.

    Why it matters: the returned map is concatenated with the packed Bayer
    input inside NoiseGate (torch.cat on dim=1). An off-by-one map therefore
    hard-crashes the router — the function should either pad asymmetrically or
    reject even windows, not silently change the spatial size.
    """
    x = packed_bayer(batch=1, h=16, w=16, seed=13)
    snr = estimate_local_snr_map(x, window_size=window_size)
    assert_shape(snr, (1, 1, 16, 16), f"snr_map(ws={window_size})")


def test_snr_map_matches_independent_reference():
    """
    Pin the exact numerics against a from-scratch reimplementation.

    train and test import this one function precisely so the routing signal
    cannot diverge; that guarantee is only worth something if the formula
    itself is nailed down — mean/std (not variance, not std/mean), averaged
    over channels, count_include_pad=True border handling, then per-image
    max normalisation with the eps in the denominator.
    """
    x = packed_bayer(batch=2, h=8, w=8, seed=17)
    got = estimate_local_snr_map(x, window_size=5)
    want = _reference_snr_map(x, window_size=5)
    assert_shape(got, want.shape, "snr_map")
    assert torch.allclose(got, want, atol=2e-5, rtol=1e-4), \
        float((got - want).abs().max())


def test_snr_map_scores_smooth_region_above_noisy_region():
    """
    A low-noise (smooth) region must score HIGHER than a high-noise region of
    the same mean level. This is the whole point of the signal: the gate uses
    it to route clean pixels and noisy pixels to different experts, and the
    sign of this relationship is what makes "high SNR" mean "clean".
    """
    h = w = 32
    x = torch.full((1, 4, h, w), 0.5)
    g = torch.Generator().manual_seed(19)
    noise = (torch.rand((1, 4, h, w // 2), generator=g) - 0.5) * 0.4
    x[:, :, :, w // 2:] += noise            # right half is noisy, same mean

    snr = estimate_local_snr_map(x, window_size=5)
    # Sample well inside each half so the zero-padded border and the seam
    # between the halves cannot contaminate the comparison.
    smooth = snr[0, 0, 8:24, 4:12]
    noisy = snr[0, 0, 8:24, 20:28]

    assert float(smooth.mean()) > float(noisy.mean())
    # The gap is not marginal: flat -> std == sqrt(eps), noisy -> std ~ 0.1.
    assert float(smooth.min()) > float(noisy.max())


def test_snr_map_constant_and_zero_images_stay_finite():
    """
    The eps path: a perfectly constant image has zero local variance, and an
    all-zero image additionally has a zero per-image max — both would divide
    by zero without `+ eps`. Real HDR patches contain saturated (constant)
    and fully-black regions, so this is hit in practice, and a single NaN in
    the SNR map poisons the gate logits for the whole batch.
    """
    for value in (0.0, 0.5, 1.0):
        x = torch.full((2, 4, 16, 16), value)
        snr = estimate_local_snr_map(x, window_size=5)
        assert_shape(snr, (2, 1, 16, 16), f"snr_map(const={value})")
        assert_finite(snr, f"snr_map(const={value})")
        assert_in_range(snr, 0.0, 1.0, f"snr_map(const={value})", atol=0.0)

    # All-zero image: the numerator is 0 everywhere, so the map is exactly 0
    # (not NaN) — there is no "max == 1" to be had.
    zeros = estimate_local_snr_map(torch.zeros(1, 4, 16, 16), window_size=5)
    assert float(zeros.abs().max()) == 0.0


def test_snr_map_normalises_each_batch_element_independently():
    """
    With B>1 the normaliser is amax(dim=(2,3), keepdim=True), i.e. per image.

    Build a batch whose two images have wildly different contrast and check
    that BOTH reach a max of 1.0 — a batch-wide max would leave the flat image
    at ~1.0 and crush the textured one to near 0, making the router's notion
    of "high SNR" depend on whatever else happened to share the batch.
    """
    x = _split_contrast_batch(h=32, w=32)
    snr = estimate_local_snr_map(x, window_size=5)

    per_image_max = snr.amax(dim=(1, 2, 3))
    assert torch.allclose(per_image_max, torch.ones(2), atol=1e-4), per_image_max
    # Raw contrast really is wildly different, so this is a meaningful check.
    raw_ratio = float(x[0].std()) / max(float(x[1].std()), 1e-12)
    assert raw_ratio < 0.01

    # Batched result == each image processed alone.
    for i in range(x.shape[0]):
        alone = estimate_local_snr_map(x[i:i + 1], window_size=5)
        assert torch.allclose(snr[i:i + 1], alone, atol=0.0), \
            f"image {i} depends on its batch neighbours"


def test_snr_map_does_not_mutate_its_input():
    """
    The training loop computes the SNR map from the same `x` it then feeds to
    the model, so any in-place write here would corrupt the network input.
    """
    x = packed_bayer(batch=2, h=16, w=16, seed=23)
    before = x.clone()
    estimate_local_snr_map(x, window_size=5)
    assert torch.equal(x, before)


@pytest.mark.gpu
def test_snr_map_is_device_agnostic(gpu_device):
    """
    The map is computed on-device inside both the train loop and the eval
    script; GPU and CPU must agree, and the result must stay on x's device.

    Runs against whichever accelerator the host has (CUDA on Polaris, XPU on
    Aurora) rather than calling .cuda(), which would fail outright on Aurora
    and hide a real device-placement regression there.
    """
    x = packed_bayer(batch=1, h=32, w=32, seed=29)
    cpu = estimate_local_snr_map(x, window_size=5)
    gpu = estimate_local_snr_map(x.to(gpu_device), window_size=5)
    assert gpu.device.type == gpu_device.type
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# SqueezeExcite
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim,h,w", [(8, 16, 16), (16, 8, 12), (64, 4, 4)])
def test_squeeze_excite_preserves_shape(dim, h, w):
    """SE is a pure channel gate: it must never change the tensor shape."""
    se = SqueezeExcite(dim, reduction=8)
    x = torch.randn(2, dim, h, w)
    out = se(x)
    assert_shape(out, (2, dim, h, w), "se_out")
    assert_finite(out, "se_out")


@pytest.mark.parametrize("dim,reduction,expected_hidden", [
    (8, 8, 4),      # 8//8 == 1 -> floored up to 4
    (16, 8, 4),     # 16//8 == 2 -> floored up to 4
    (32, 8, 4),     # 32//8 == 4 -> exactly the floor
    (64, 8, 8),     # above the floor
    (8, 2, 4),      # 8//2 == 4
    (8, 32, 4),     # 8//32 == 0 -> would be a 0-channel conv without the floor
])
def test_squeeze_excite_hidden_is_at_least_four(dim, reduction, expected_hidden):
    """
    hidden = max(dim // reduction, 4).

    The floor is what lets the tiny test/ablation architectures (dim=8 with
    the default reduction=8) build at all: without it the bottleneck conv
    would be asked for 1 — or, at reduction=32, 0 — output channels.
    """
    se = SqueezeExcite(dim, reduction=reduction)
    squeeze, expand = se.gate[1], se.gate[3]
    assert squeeze.out_channels == expected_hidden
    assert squeeze.in_channels == dim
    assert expand.in_channels == expected_hidden
    assert expand.out_channels == dim
    assert expected_hidden >= 4


def test_squeeze_excite_gate_is_sigmoid_bounded():
    """
    The gate ends in Sigmoid, so every multiplier lies in (0,1): the output
    can only ever shrink a channel, never amplify or flip it. That bound is
    why SE can be dropped into a residual trunk without rescaling anything.
    """
    se = SqueezeExcite(16, reduction=8)
    with torch.no_grad():                     # push the gate off its init
        for m in se.gate:
            if isinstance(m, torch.nn.Conv2d):
                m.weight.normal_(0.0, 2.0)
                m.bias.normal_(0.0, 2.0)

    x = torch.randn(3, 16, 8, 8) * 5.0
    out = se(x)

    assert (out.abs() <= x.abs() + 1e-6).all(), "SE amplified its input"
    nz = x.abs() > 1e-3
    assert (torch.sign(out[nz]) == torch.sign(x[nz])).all(), "SE flipped a sign"
    # And the gate is strictly non-degenerate: nothing is fully zeroed.
    assert float(out.abs().max()) > 0.0


def test_squeeze_excite_keeps_spatially_constant_input_constant():
    """
    Global AdaptiveAvgPool2d(1) means the gate is one scalar per (sample,
    channel). A spatially flat input therefore stays flat — the block adds no
    spatial structure of its own, which is what keeps it safe inside a
    denoiser (it cannot invent texture).
    """
    dim = 8
    se = SqueezeExcite(dim, reduction=8)
    per_channel = torch.randn(2, dim, 1, 1)
    x = per_channel.expand(2, dim, 6, 6).contiguous()

    out = se(x)
    spread = out.amax(dim=(2, 3)) - out.amin(dim=(2, 3))
    assert float(spread.abs().max()) < 1e-6, "SE broke spatial constancy"

    # The surviving value is x scaled by a per-(sample,channel) gate in (0,1).
    ratio = out[:, :, 0, 0] / per_channel[:, :, 0, 0]
    assert ((ratio > 0.0) & (ratio < 1.0)).all()


# ═════════════════════════════════════════════════════════════════════════════
# ResidualConvBlock / SEResidualBlock
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim", [8, 16, 64])
@pytest.mark.parametrize("se_reduction", [None, 8])
def test_residual_conv_block_preserves_shape(dim, se_reduction):
    """
    Shape in == shape out for every width the trunk uses (dim, dim*4, dim*8,
    dim*2 ...). The U-Net concatenates block outputs with skip tensors, so a
    single channel or pixel of drift here breaks the whole decoder.
    """
    block = ResidualConvBlock(dim, se_reduction)
    x = torch.randn(2, dim, 8, 12)
    out = block(x)
    assert_shape(out, (2, dim, 8, 12), "resblock_out")
    assert_finite(out, "resblock_out")


def test_residual_block_se_reduction_none_is_identity_and_legacy_loadable():
    """
    se_reduction=None must give `.se = nn.Identity` and a state_dict with
    exactly the two legacy conv weights — that is the documented promise that
    pre-SE checkpoints load unchanged. An int must give a real SqueezeExcite
    with extra parameters, which (correctly) makes the two variants
    non-interchangeable under strict loading.
    """
    legacy = ResidualConvBlock(8, None)
    assert isinstance(legacy.se, torch.nn.Identity)
    assert sorted(legacy.state_dict().keys()) == ["conv1.weight", "conv2.weight"]
    # conv1/conv2 are bias-free in the legacy block; keep it that way.
    assert legacy.conv1.bias is None and legacy.conv2.bias is None

    # A legacy checkpoint round-trips strictly into a legacy block.
    fresh = ResidualConvBlock(8, None)
    fresh.load_state_dict(legacy.state_dict(), strict=True)
    assert torch.equal(fresh.conv1.weight, legacy.conv1.weight)

    se_block = ResidualConvBlock(8, 8)
    assert isinstance(se_block.se, SqueezeExcite)
    assert sorted(se_block.state_dict().keys()) == [
        "conv1.weight", "conv2.weight",
        "se.gate.1.bias", "se.gate.1.weight",
        "se.gate.3.bias", "se.gate.3.weight",
    ]
    with pytest.raises(RuntimeError):
        se_block.load_state_dict(legacy.state_dict(), strict=True)


def test_residual_block_zero_falls_back_to_identity_se():
    """
    The guard is `if se_reduction`, so 0 (and not just None) selects the
    legacy path. Worth pinning because 0 would otherwise be a ZeroDivisionError
    inside SqueezeExcite's `dim // reduction`.
    """
    block = ResidualConvBlock(8, 0)
    assert isinstance(block.se, torch.nn.Identity)


def test_residual_block_applies_post_activation_not_bare_identity():
    """
    The block is `GELU(res + f(x))`, not `res + f(x)`: zeroing the second conv
    leaves GELU(x), not x. Pinning this stops a future "identity-init the
    residual branch" refactor from silently assuming the block can pass its
    input through untouched (it cannot — negatives get squashed).
    """
    block = ResidualConvBlock(8, None)
    _zero_out(block.conv2)
    x = torch.randn(1, 8, 8, 8)
    out = block(x)
    assert torch.allclose(out, F.gelu(x), atol=1e-6)
    assert not torch.allclose(out, x, atol=1e-3), "block behaved as a bare identity"


def test_seresidualblock_is_an_alias_of_residual_conv_block():
    """
    SEResidualBlock is a module-level alias, not a subclass. Checkpoints store
    parameter paths, so the two names must remain the same class or old
    state_dicts stop matching.
    """
    assert SEResidualBlock is ResidualConvBlock
    assert isinstance(SEResidualBlock(8, 8), ResidualConvBlock)


@pytest.mark.parametrize("se_reduction", [None, 8])
def test_residual_block_gradients_reach_every_parameter(se_reduction):
    """
    Every parameter — including the SE gate's squeeze/expand convs, which sit
    behind a global pool and a sigmoid — must receive a non-zero gradient.
    A dead SE branch would train silently as a no-op.
    """
    block = ResidualConvBlock(8, se_reduction)
    x = torch.rand(2, 8, 8, 8)
    block(x).pow(2).mean().backward()

    named = list(block.named_parameters())
    assert named, "block has no parameters"
    for name, p in named:
        assert p.grad is not None, f"{name} got no gradient"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} gradient is all zeros"


# ═════════════════════════════════════════════════════════════════════════════
# HeavyExposhare
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim", [8, 32])
def test_heavy_exposhare_preserves_shape(dim):
    """
    Used inline in the encoder at dim and dim*4; it must return exactly what
    it was given so the following PixelUnshuffle stage still lines up.
    """
    block = HeavyExposhare(dim)
    x = torch.randn(2, dim, 8, 8)
    out = block(x)
    assert_shape(out, (2, dim, 8, 8), "exposhare_out")
    assert_finite(out, "exposhare_out")


@pytest.mark.parametrize("dim", [8, 32])
def test_heavy_exposhare_internal_channel_widths(dim):
    """
    The documented widening is dim -> dim*2 -> dim*2 -> dim, with 3x3 convs
    for the two expanding layers and a 1x1 projection back down. Checkpoint
    tensor shapes depend on this exactly, so it is part of the contract.
    """
    block = HeavyExposhare(dim)
    assert (block.conv1.in_channels, block.conv1.out_channels) == (dim, dim * 2)
    assert (block.conv2.in_channels, block.conv2.out_channels) == (dim * 2, dim * 2)
    assert (block.conv3.in_channels, block.conv3.out_channels) == (dim * 2, dim)
    assert block.conv1.kernel_size == (3, 3) and block.conv1.padding == (1, 1)
    assert block.conv2.kernel_size == (3, 3) and block.conv2.padding == (1, 1)
    assert block.conv3.kernel_size == (1, 1)


def test_heavy_exposhare_is_a_pure_residual():
    """
    Unlike ResidualConvBlock there is NO post-activation: the return is
    `res + z`. Zeroing the output projection must therefore reproduce the
    input bit-for-bit, which is what makes this block safe to stack (it can
    learn to do nothing) and lets negative features survive it.
    """
    block = HeavyExposhare(8)
    _zero_out(block.conv3)
    x = torch.randn(2, 8, 8, 8)
    assert torch.equal(block(x), x)


def test_heavy_exposhare_mixes_across_channels():
    """
    Its job is exposure sharing BETWEEN the Bayer planes, so the output at a
    given channel must depend on the other channels. Perturb one input
    channel and check the response leaks into the others — a depthwise/
    per-channel regression would show up here.
    """
    block = HeavyExposhare(8)
    x = torch.zeros(1, 8, 6, 6)
    base = block(x)
    x2 = x.clone()
    x2[:, 0] = 1.0
    moved = (block(x2) - base).abs().amax(dim=(0, 2, 3))
    other_channels = moved[1:]
    assert float(other_channels.max()) > 1e-6, "no cross-channel mixing"


# ═════════════════════════════════════════════════════════════════════════════
# NoiseGate
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("num_experts", [1, 2, 3, 5])
def test_noise_gate_shape_at_packed_bayer_resolution(num_experts):
    """
    (x [B,4,h,w], snr [B,1,h,w]) -> [B,K,h,w].

    The gate deliberately runs at PACKED Bayer resolution; MoEDenoiser
    bilinearly upsamples the result 2x to sensor resolution. If the gate
    returned sensor-res weights the blend would be done twice over.
    """
    gate = NoiseGate(num_experts)
    x = packed_bayer(batch=2, h=16, w=24, seed=31)
    snr = estimate_local_snr_map(x, window_size=5)
    out = gate(x, snr)
    assert_shape(out, (2, num_experts, 16, 24), "gates")
    assert_finite(out, "gates")
    assert_in_range(out, 0.0, 1.0, "gates", atol=0.0)


@pytest.mark.parametrize("num_experts", [1, 2, 3, 5])
def test_noise_gate_is_exactly_uniform_at_initialisation(num_experts):
    """
    The final 1x1 conv is zero-initialised (weight AND bias), so at step 0
    every logit is 0 and softmax gives EXACTLY 1/K at every pixel.

    This is a documented training-stability property: the MoE starts as a
    plain average of its experts, so no expert is starved before it has
    learned anything. Asserting it exactly (not approximately) is the point —
    a non-zero bias init would break it while still "looking uniform".
    """
    gate = NoiseGate(num_experts)
    assert float(gate.net[-1].weight.abs().max()) == 0.0
    assert float(gate.net[-1].bias.abs().max()) == 0.0

    x = packed_bayer(batch=2, h=16, w=16, seed=37)
    snr = estimate_local_snr_map(x, window_size=5)
    out = gate(x, snr)

    expected = torch.full_like(out, 1.0 / num_experts)
    assert torch.equal(out, expected), \
        f"init routing not exactly uniform: {torch.unique(out)}"

    # Uniform for *any* input, including degenerate ones.
    flat = torch.full((1, 4, 16, 16), 0.5)
    out_flat = gate(flat, torch.zeros(1, 1, 16, 16))
    assert torch.equal(out_flat, torch.full_like(out_flat, 1.0 / num_experts))


@pytest.mark.parametrize("num_experts", [2, 3])
def test_noise_gate_softmax_sums_to_one_per_pixel(num_experts):
    """
    After training moves the head off zero, the weights must still form a
    per-pixel probability simplex — MoEDenoiser relies on sum(gates)==1 for
    the blend to be a convex combination (and hence stay in [0,1] when the
    experts do).
    """
    gate = NoiseGate(num_experts)
    with torch.no_grad():                     # simulate a trained gate
        gate.net[-1].weight.normal_(0.0, 1.0)
        gate.net[-1].bias.normal_(0.0, 1.0)

    x = packed_bayer(batch=2, h=16, w=16, seed=41)
    snr = estimate_local_snr_map(x, window_size=5)
    out = gate(x, snr)

    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        float((sums - 1.0).abs().max())
    # And routing is genuinely input-dependent, not a constant vector.
    assert float(out.amax(dim=(2, 3)).sub(out.amin(dim=(2, 3))).max()) > 1e-4


@pytest.mark.parametrize("num_experts", [1, 2, 3, 8])
def test_noise_gate_in_channels_is_always_five(num_experts):
    """
    in_channels == 5 (4 BGGR planes + 1 SNR plane) regardless of K, and only
    the LAST conv's width tracks K. MoEDenoiser passes in_channels=5
    explicitly with a comment that it is independent of out_channels; if the
    stem width ever tracked K or the RGB channel count, the cat() in forward
    would break.
    """
    gate = NoiseGate(num_experts)
    assert gate.net[0].in_channels == 5
    assert gate.net[-1].out_channels == num_experts
    assert gate.net[0].out_channels == gate.net[2].in_channels == 16   # hidden default


def test_noise_gate_reads_snr_as_the_final_input_channel():
    """
    forward does `torch.cat([x, snr_map], dim=1)`, so the SNR plane must be
    channel index 4. Verified structurally: wire the stem to look ONLY at
    channel 4 and confirm the output then ignores x entirely but still
    responds to snr. If the order were ever flipped the router would silently
    treat the blue plane as the SNR signal.
    """
    gate = NoiseGate(3)
    with torch.no_grad():
        gate.net[0].weight.zero_()
        gate.net[0].weight[:, 4] = 1.0        # only the SNR channel is read
        gate.net[0].bias.zero_()
        gate.net[-1].weight.normal_(0.0, 1.0)  # un-zero the head so it varies
        gate.net[-1].bias.zero_()

    snr = torch.rand(1, 1, 8, 8)
    x_a = packed_bayer(batch=1, h=8, w=8, seed=43)
    x_b = packed_bayer(batch=1, h=8, w=8, seed=44)

    out_a, out_b = gate(x_a, snr), gate(x_b, snr)
    assert torch.equal(out_a, out_b), "gate output depends on x through channel 4"

    other_snr = torch.rand(1, 1, 8, 8)
    assert not torch.allclose(gate(x_a, other_snr), out_a, atol=1e-6), \
        "gate ignored the SNR channel"


def test_noise_gate_head_gradient_is_nonzero_at_init():
    """
    The zero-init head is dormant, not dead: at step 0 the *body* of the gate
    receives exactly zero gradient (d logits / d hidden == net[-1].weight == 0),
    but net[-1].weight and net[-1].bias do get non-zero gradients — so one
    optimiser step is enough for the router to leave uniform routing and wake
    the body up.

    Pinning both halves matters: an all-zero head gradient would freeze the
    MoE as a permanent ensemble average, and a non-zero body gradient here
    would mean the head was not actually zero-initialised.
    """
    gate = NoiseGate(3)
    x = packed_bayer(batch=2, h=16, w=16, seed=47)
    snr = estimate_local_snr_map(x, window_size=5)
    # A non-symmetric target so the uniform output is genuinely wrong.
    target = torch.zeros(2, 3, 16, 16)
    target[:, 0] = 1.0
    F.mse_loss(gate(x, snr), target).backward()

    head = gate.net[-1]
    assert float(head.weight.grad.abs().sum()) > 0.0, "zero-init head cannot learn"
    assert float(head.bias.grad.abs().sum()) > 0.0, "zero-init head bias cannot learn"

    for name, p in gate.named_parameters():
        if name.startswith("net.4."):         # the head itself
            continue
        assert float(p.grad.abs().sum()) == 0.0, \
            f"{name} got a gradient through a zero-initialised head"


def test_noise_gate_gradients_reach_every_parameter_once_head_is_nonzero():
    """
    Once the head has moved off zero (i.e. from training step 2 onwards) every
    parameter in the gate must receive a non-zero gradient, otherwise part of
    the router would never train and routing would stay input-blind.
    """
    gate = NoiseGate(3)
    with torch.no_grad():                     # simulate one taken step
        gate.net[-1].weight.normal_(0.0, 0.5)
        gate.net[-1].bias.normal_(0.0, 0.5)

    x = packed_bayer(batch=2, h=16, w=16, seed=47)
    snr = estimate_local_snr_map(x, window_size=5)
    target = torch.zeros(2, 3, 16, 16)
    target[:, 0] = 1.0
    F.mse_loss(gate(x, snr), target).backward()

    for name, p in gate.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} gradient is all zeros"


# ═════════════════════════════════════════════════════════════════════════════
# ExpertHead
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim,h_in,w_in", [(8, 4, 4), (8, 8, 6), (32, 4, 4)])
def test_expert_head_upsamples_four_times(dim, h_in, w_in):
    """
    in_dim = dim*2 trunk features at half packed-Bayer resolution ->
    [B, out_channels, 4*h_in, 4*w_in] via two PixelShuffle(2) stages
    (half-res -> packed Bayer res -> sensor res).

    That 4x is the whole demosaicing upsample; getting it wrong by one stage
    would silently emit a half-resolution "sensor" image.
    """
    in_dim = dim * 2
    head = ExpertHead(in_dim, out_channels=3, num_blocks=2, se_reduction=8)
    with torch.no_grad():                     # un-zero so shapes are meaningful
        head.proj_out.weight.normal_(0.0, 0.1)
    feat = torch.randn(2, in_dim, h_in, w_in)
    out = head(feat)
    assert_shape(out, (2, 3, 4 * h_in, 4 * w_in), "expert_out")
    assert_finite(out, "expert_out")


@pytest.mark.parametrize("out_channels", [1, 3, 4])
def test_expert_head_honours_out_channels(out_channels):
    """
    proj_out must map r -> out_channels*4 so the second PixelShuffle(2) yields
    out_channels planes. build_denoiser forwards out_channels through, so a
    non-RGB head (e.g. mono ablation) has to work.
    """
    head = ExpertHead(16, out_channels=out_channels, num_blocks=1, se_reduction=8)
    assert head.proj_out.out_channels == out_channels * 4
    out = head(torch.randn(1, 16, 4, 4))
    assert_shape(out, (1, out_channels, 16, 16), "expert_out")


@pytest.mark.parametrize("in_dim,expected_r", [(16, 4), (32, 8), (64, 16)])
def test_expert_head_r_is_in_dim_over_four(in_dim, expected_r):
    """
    r = in_dim // 4 is fixed by the first PixelShuffle(2) (channels/4), and
    the two refinement convs plus proj_out's input must all agree with it.
    At the default dim=32 this is 64//4 = 16 = dim//2, matching the teacher's
    refinement width.
    """
    head = ExpertHead(in_dim, out_channels=3, num_blocks=1, se_reduction=8)
    assert head.refine1.in_channels == head.refine1.out_channels == expected_r
    assert head.refine2.in_channels == head.refine2.out_channels == expected_r
    assert head.proj_out.in_channels == expected_r
    assert head.blocks[0].conv1.in_channels == in_dim


def test_expert_head_outputs_exactly_zero_at_initialisation():
    """
    proj_out is zero-initialised (weight AND bias), so a freshly built head
    returns EXACTLY 0 for any input.

    Combined with the uniform gate this means the MoE's initial prediction is
    exactly zero rather than noise — the documented warm-start. Note the raw
    head output is 0; MoEDenoiser is what clamps it up to _CLAMP_EPS.
    """
    head = ExpertHead(16, out_channels=3, num_blocks=2, se_reduction=8)
    assert float(head.proj_out.weight.abs().max()) == 0.0
    assert float(head.proj_out.bias.abs().max()) == 0.0

    for feat in (torch.randn(2, 16, 4, 4), torch.randn(2, 16, 4, 4) * 100.0):
        out = head(feat)
        assert float(out.abs().max()) == 0.0, "fresh expert head is not exactly zero"


def test_expert_head_pixelshuffle_places_channels_at_the_right_subpixels():
    """
    The final PixelShuffle(2) reads proj_out channel k as
    (colour = k//4, dy = (k%4)//2, dx = k%2). Pin that layout: a permutation
    here would scramble RGB across the 2x2 sensor cell and show up as a
    checkerboard colour artefact rather than an error.
    """
    head = ExpertHead(16, out_channels=3, num_blocks=1, se_reduction=8)
    with torch.no_grad():
        head.proj_out.bias.copy_(torch.arange(12, dtype=torch.float32))
    out = head(torch.randn(1, 16, 2, 2))     # weight is still zero -> bias only
    assert_shape(out, (1, 3, 8, 8), "expert_out")

    for colour in range(3):
        for dy in range(2):
            for dx in range(2):
                sub = out[0, colour, dy::2, dx::2]
                expected = float(colour * 4 + dy * 2 + dx)
                assert torch.allclose(sub, torch.full_like(sub, expected)), \
                    f"subpixel ({colour},{dy},{dx}) got {sub.flatten()[0]}, " \
                    f"expected {expected}"


def test_expert_head_gradients_reach_every_parameter():
    """
    With proj_out at exactly zero, gradients into the residual blocks and
    refinement convs flow only through proj_out's *weight* gradient; verify
    nothing upstream is dead once the head is perturbed, so the trunk keeps
    receiving a learning signal from every expert.
    """
    head = ExpertHead(16, out_channels=3, num_blocks=2, se_reduction=8)
    with torch.no_grad():
        head.proj_out.weight.normal_(0.0, 0.1)
        head.proj_out.bias.normal_(0.0, 0.1)

    feat = torch.rand(2, 16, 4, 4, requires_grad=True)
    head(feat).pow(2).mean().backward()

    for name, p in head.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} gradient is all zeros"
    assert feat.grad is not None and float(feat.grad.abs().sum()) > 0.0, \
        "no gradient reaches the shared trunk features"
