"""
Unit tests for blocks_Restormer.py — LayerNorm, MDTA, GDFN, RestormerBlock.

These four modules are the transformer bottleneck of both TransUNet_Teacher_HDR
and the MoE/Single denoisers (`self.latent = nn.Sequential(*[RestormerBlock(
dim=dim*16, num_heads=heads[3]) ...])`).  They are pure feature-map operators:
they take and return [B, C, H, W] and must never change the shape, because the
decoder's PixelShuffle stages and the skip connections assume it.

The properties pinned down here are the ones a "harmless" refactor breaks
silently:
  * LayerNorm normalises ACROSS CHANNELS per pixel (dim=1), with a biased
    variance and eps INSIDE the sqrt (the only thing standing between a
    flat patch and a NaN).
  * MDTA is *transposed* (channel) attention: its attention map is
    [B, heads, C//heads, C//heads] and independent of H and W.  If that ever
    became spatial ([B, heads, HW, HW]) both memory use and the
    teacher/student distillation target would change.
  * q and k are L2-normalised along the TOKEN dim before the product.
  * GDFN's hidden width is int(dim * mlp_ratio) (truncated, not rounded) and
    its gate is gelu(first half) * (second half), in that order.
  * RestormerBlock is a pre-norm double-residual block and silently ignores
    the Swin-era window_size/shift_size kwargs, as its docstring promises.
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks_Restormer import GDFN, MDTA, LayerNorm, RestormerBlock
from helpers import assert_finite, assert_shape


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — conftest/helpers are shared)
# ─────────────────────────────────────────────────────────────────────────────

def _feat(b=2, c=8, h=6, w=6, seed=0, scale=1.0, offset=0.0):
    """Deterministic random feature map [b, c, h, w] — the latent-tensor shape."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn((b, c, h, w), generator=g) * scale + offset


def _mdta_reference(mdta, x, normalize_dim=-1, gate_swap=False):
    """
    Re-implementation of MDTA.forward from its own submodules.

    `normalize_dim` lets a test discriminate "L2-normalise along the token dim"
    (-1, what the source does) from "along the channel dim" (-2).
    """
    b, c, h, w = x.shape
    nh = mdta.num_heads
    qkv = mdta.qkv_dwconv(mdta.qkv(x))
    q, k, v = qkv.chunk(3, dim=1)
    q = q.reshape(b, nh, c // nh, h * w)
    k = k.reshape(b, nh, c // nh, h * w)
    v = v.reshape(b, nh, c // nh, h * w)
    q = F.normalize(q, dim=normalize_dim)
    k = F.normalize(k, dim=normalize_dim)
    attn = ((q @ k.transpose(-2, -1)) * mdta.temperature).softmax(dim=-1)
    out = (attn @ v).reshape(b, c, h, w)
    return mdta.project_out(out), attn


def _gdfn_reference(gdfn, x, swap=False):
    """gelu(x1) * x2 (source order) or gelu(x2) * x1 (swapped, for contrast)."""
    hidden = gdfn.dwconv(gdfn.project_in(x))
    x1, x2 = hidden.chunk(2, dim=1)
    gated = F.gelu(x2) * x1 if swap else F.gelu(x1) * x2
    return gdfn.project_out(gated)


# =============================================================================
# LayerNorm
# =============================================================================

def test_layernorm_normalises_across_the_channel_dim():
    """
    With the affine params at their identity values the output must have zero
    mean and unit variance along dim=1 (channels) for EVERY pixel.

    This is the whole point of Restormer's LayerNorm: it is a per-pixel channel
    norm, not a spatial norm and not a BatchNorm.  A refactor to `mean((1,2,3))`
    would still train, but the bottleneck statistics — and therefore every
    checkpoint — would change meaning.
    """
    ln = LayerNorm(8)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.zero_()
    x = _feat(b=2, c=8, h=5, w=7, scale=5.0, offset=3.0)

    y = ln(x)

    assert_shape(y, x.shape, "layernorm output")
    assert_finite(y, "layernorm output")
    assert y.mean(dim=1).abs().max() < 1e-5, "channel mean is not 0"
    assert (y.var(dim=1, unbiased=False) - 1.0).abs().max() < 1e-4, \
        "channel variance is not 1"


def test_layernorm_is_independent_per_pixel():
    """
    Changing one pixel must not change any other pixel's output.

    Normalising over the spatial dims (a very easy typo: `mean((2,3))`) would
    couple all pixels together, which would leak information across a patch
    boundary during the tiled inference in test_dual_MoE_two_phase.infer_patches.
    """
    ln = LayerNorm(6)
    x = _feat(b=1, c=6, h=4, w=4, seed=1)
    y_ref = ln(x)

    x_perturbed = x.clone()
    x_perturbed[0, :, 0, 0] += 100.0
    y_new = ln(x_perturbed)

    # Every pixel except (0, 0) is bit-identical.
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[0, 0] = False
    assert torch.equal(y_new[0, :, mask], y_ref[0, :, mask]), \
        "LayerNorm coupled unrelated pixels"
    # And the perturbed pixel really did change.
    assert not torch.allclose(y_new[0, :, 0, 0], y_ref[0, :, 0, 0])


def test_layernorm_weight_and_bias_are_learnable_params_of_shape_1c11():
    """
    weight/bias must be nn.Parameters of shape [1, C, 1, 1], initialised to
    ones/zeros, and must appear in the state_dict under exactly those names.

    The shape is what lets them broadcast over NCHW; the names are baked into
    every saved checkpoint (`latent.0.norm1.weight`), so renaming them silently
    breaks `load_state_dict(strict=True)` in load_model_from_checkpoint.
    """
    ln = LayerNorm(12)

    for name in ("weight", "bias"):
        p = getattr(ln, name)
        assert isinstance(p, nn.Parameter), f"{name} is not an nn.Parameter"
        assert p.requires_grad, f"{name} is frozen"
        assert_shape(p, (1, 12, 1, 1), f"layernorm {name}")

    assert torch.equal(ln.weight, torch.ones(1, 12, 1, 1))
    assert torch.equal(ln.bias, torch.zeros(1, 12, 1, 1))
    assert set(ln.state_dict().keys()) == {"weight", "bias"}
    assert ln.eps == pytest.approx(1e-6)


def test_layernorm_applies_affine_after_normalisation():
    """
    Output must equal weight * normalised + bias (scale then shift), per channel.

    Applying the affine before the normalisation would cancel it out entirely,
    turning the learnable gain into a no-op — a bug that is invisible in the
    loss curve but silently removes parameters from the model.
    """
    ln = LayerNorm(4)
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([2.0, -1.0, 0.5, 3.0]).view(1, 4, 1, 1))
        ln.bias.copy_(torch.tensor([1.0, 0.0, -2.0, 0.25]).view(1, 4, 1, 1))
    x = _feat(b=1, c=4, h=3, w=3, seed=2, scale=2.0)

    normed = (x - x.mean(1, keepdim=True)) / \
        x.var(1, keepdim=True, unbiased=False).add(ln.eps).sqrt()
    expected = normed * ln.weight + ln.bias

    torch.testing.assert_close(ln(x), expected, rtol=1e-6, atol=1e-6)


def test_layernorm_uses_the_biased_variance():
    """
    The variance must be the biased one (unbiased=False, divide by C).

    With C=4 the unbiased estimate is 4/3 larger, i.e. the normalised features
    would be ~13% smaller — enough to shift the bottleneck's operating point and
    invalidate a pretrained checkpoint.  Pinning it against BOTH variants makes
    the test fail if the flag is ever dropped (unbiased defaults to True).
    """
    ln = LayerNorm(4)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.zero_()
    x = _feat(b=1, c=4, h=3, w=3, seed=3, scale=3.0)
    mean = x.mean(1, keepdim=True)

    biased = (x - mean) / x.var(1, keepdim=True, unbiased=False).add(ln.eps).sqrt()
    unbiased = (x - mean) / x.var(1, keepdim=True, unbiased=True).add(ln.eps).sqrt()

    y = ln(x)
    torch.testing.assert_close(y, biased, rtol=1e-6, atol=1e-6)
    assert (y - unbiased).abs().max() > 1e-3, \
        "biased and unbiased variance are indistinguishable — test is toothless"


def test_layernorm_eps_prevents_nan_on_a_channel_constant_input():
    """
    A pixel whose channels are all equal has variance exactly 0.  eps lives
    INSIDE the sqrt, so 0 / sqrt(0 + 1e-6) = 0 and the output is exactly `bias`.

    Without eps this is 0/0 = NaN.  Flat regions are not hypothetical here:
    HDR_Mobile_dataset min/max-normalises each tile, and a saturated or fully
    black tile produces exactly this input.  One NaN in the bottleneck poisons
    the whole batch's gradients.
    """
    ln = LayerNorm(4)
    with torch.no_grad():
        ln.bias.copy_(torch.tensor([0.5, -0.5, 1.0, 0.0]).view(1, 4, 1, 1))
    # Value chosen so mean(4 identical float32 values) is exact -> x-mean == 0.
    x = torch.full((2, 4, 3, 3), 0.5)

    y = ln(x)

    assert_finite(y, "layernorm output on a flat input")
    assert torch.equal(y, ln.bias.expand_as(y)), \
        "flat input should collapse to exactly the bias"

    # Same story for a random-but-channel-constant input: finite, ~= bias.
    # (Only ~= because 1/sqrt(eps) amplifies float rounding in the mean by 1e3.)
    x2 = _feat(b=2, c=4, h=3, w=3, seed=4)[:, :1].expand(2, 4, 3, 3).contiguous()
    y2 = ln(x2)
    assert_finite(y2, "layernorm output on a channel-constant input")
    assert (y2 - ln.bias).abs().max() < 1e-2


def test_layernorm_gradients_flow_to_weight_bias_and_input():
    """
    Backward must reach the input and both affine params with finite,
    non-zero gradients — the block is trained end to end, and a detached
    normalisation would freeze the bottleneck without any error message.
    """
    ln = LayerNorm(6)
    x = _feat(b=2, c=6, h=4, w=4, seed=5, scale=2.0).requires_grad_(True)

    ln(x).pow(2).mean().backward()

    for name, p in ln.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert_finite(p.grad, f"grad of {name}")
        assert p.grad.abs().max() > 0, f"grad for {name} is all zeros"
    assert x.grad is not None
    assert_finite(x.grad, "grad wrt input")
    assert x.grad.abs().max() > 0


@pytest.mark.parametrize("shape", [(1, 8, 1, 1), (1, 8, 3, 5), (4, 8, 8, 8), (2, 8, 16, 9)])
def test_layernorm_preserves_shape(shape):
    """Shape in == shape out, including 1x1 and non-square/odd spatial sizes."""
    ln = LayerNorm(shape[1])
    y = ln(torch.randn(*shape))
    assert_shape(y, shape, "layernorm output")
    assert_finite(y, "layernorm output")


def test_layernorm_needs_a_batch_dim_and_silently_rank_promotes_3d_input():
    """
    Documents ACTUAL behaviour on an unbatched [C, H, W] tensor: no error is
    raised, but dim=1 is then H, so the module normalises over ROWS and the
    [1,C,1,1] affine broadcasts the result up to [1, C, H, W].

    Worth pinning because the dataset yields unbatched [4, h, w] tensors: if a
    caller ever hands an unbatched feature map to the bottleneck it will get
    silently wrong numbers and a silently promoted rank instead of a crash.
    """
    ln = LayerNorm(4)
    x3 = _feat(b=1, c=4, h=5, w=5, seed=6)[0]  # [4, 5, 5]

    y = ln(x3)

    assert tuple(y.shape) == (1, 4, 5, 5), "3D input is no longer rank-promoted"
    # It is NOT channel-normalised (that is the point of the warning).
    assert y.mean(dim=1).abs().max() > 1e-3
    # It *is* normalised over dim=1 of the input, i.e. the ROW axis H, so the
    # promoted [1, C, H, W] result has zero mean along H, not along C.
    assert y[0].mean(dim=1).abs().max() < 1e-5


# =============================================================================
# MDTA
# =============================================================================

@pytest.mark.parametrize("shape,heads", [
    ((1, 8, 1, 1), 2),
    ((1, 8, 3, 5), 1),
    ((2, 8, 6, 6), 2),
    ((2, 16, 8, 8), 4),
    ((1, 8, 16, 9), 8),
])
def test_mdta_preserves_shape(shape, heads):
    """
    MDTA must be shape-preserving for any batch size, head count dividing C and
    any spatial size (1x1, odd, non-square).  It sits inside a residual add, so
    any shape change is an immediate crash in RestormerBlock.
    """
    mdta = MDTA(shape[1], heads)
    y = mdta(torch.randn(*shape))
    assert_shape(y, shape, "MDTA output")
    assert_finite(y, "MDTA output")


def test_mdta_submodule_shapes_and_qkv_chunking():
    """
    qkv projects C -> 3C, the depth-wise conv keeps 3C with groups=3C (so q, k
    and v never mix spatially), and project_out maps C -> C.  chunk(3, dim=1)
    must therefore split into three exactly-C-channel blocks.

    A wrong `groups` here turns the "Multi-Dconv" into a dense 3Cx3C conv:
    still runs, ~C times more FLOPs, and estimate_flops in the eval script
    would under-report it.
    """
    dim, heads = 8, 2
    mdta = MDTA(dim, heads)

    assert (mdta.qkv.in_channels, mdta.qkv.out_channels) == (dim, 3 * dim)
    assert mdta.qkv.kernel_size == (1, 1)
    assert mdta.qkv_dwconv.in_channels == 3 * dim
    assert mdta.qkv_dwconv.out_channels == 3 * dim
    assert mdta.qkv_dwconv.groups == 3 * dim, "qkv_dwconv is not depth-wise"
    assert mdta.qkv_dwconv.kernel_size == (3, 3)
    assert mdta.qkv_dwconv.padding == (1, 1), "padding must keep H, W"
    assert (mdta.project_out.in_channels, mdta.project_out.out_channels) == (dim, dim)

    x = _feat(b=1, c=dim, h=5, w=5, seed=7)
    parts = mdta.qkv_dwconv(mdta.qkv(x)).chunk(3, dim=1)
    assert len(parts) == 3
    for p in parts:
        assert_shape(p, (1, dim, 5, 5), "qkv chunk")


@pytest.mark.parametrize("h,w", [(4, 4), (8, 8), (16, 16), (3, 7)])
def test_mdta_attention_map_is_channel_attention_not_spatial(h, w):
    """
    The returned map must be [B, heads, C//heads, C//heads] — a CHANNEL
    covariance — and its size must not depend on H or W at all.

    This is the defining property of "transposed" attention and the reason the
    bottleneck is affordable; it is also the tensor the comment earmarks as the
    teacher/student distillation target, so a switch to spatial attention would
    both blow up memory (HW x HW) and silently change that target.
    """
    b, dim, heads = 2, 16, 4
    mdta = MDTA(dim, heads)

    out, attn = mdta(torch.randn(b, dim, h, w), return_attn=True)

    assert_shape(out, (b, dim, h, w), "MDTA output")
    assert_shape(attn, (b, heads, dim // heads, dim // heads), "MDTA attention")
    assert attn.shape[-1] != h * w or (h * w) == dim // heads, \
        "attention looks spatial ([B, heads, HW, HW])"
    assert_finite(attn, "MDTA attention")


def test_mdta_attention_rows_are_a_softmax_over_the_last_dim():
    """
    Every attention row must be non-negative and sum to exactly 1 along dim=-1.

    softmax over the wrong dim (-2) still produces a plausible-looking matrix
    and trains, but the rows no longer form a convex combination of value
    channels, so `attn @ v` stops being an averaging operator.
    """
    mdta = MDTA(16, 4)
    _, attn = mdta(_feat(b=3, c=16, h=6, w=6, seed=8), return_attn=True)

    assert float(attn.min()) >= 0.0
    assert float(attn.max()) <= 1.0
    row_sums = attn.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums),
                               rtol=1e-5, atol=1e-5)
    # ...and NOT normalised along the other axis, which would be the typo.
    col_sums = attn.sum(dim=-2)
    assert (col_sums - 1.0).abs().max() > 1e-4, \
        "attention is doubly stochastic — cannot tell the softmax dim apart"


def test_mdta_temperature_is_a_learnable_parameter_of_shape_heads11():
    """
    temperature must be an nn.Parameter of shape [num_heads, 1, 1] initialised
    to ones, so it broadcasts over [B, heads, C/h, C/h] and gives each head its
    own learned sharpness.  A scalar or a [heads] shape would broadcast onto the
    wrong axis (or not at all).
    """
    mdta = MDTA(16, 4)
    t = mdta.temperature
    assert isinstance(t, nn.Parameter) and t.requires_grad
    assert_shape(t, (4, 1, 1), "temperature")
    assert torch.equal(t, torch.ones(4, 1, 1))
    assert "temperature" in mdta.state_dict()


def test_mdta_temperature_scales_the_logits_before_the_softmax():
    """
    Raising temperature must sharpen the attention (row max -> 1); driving it to
    0 must flatten it to a uniform 1/(C/heads).

    That ordering (multiply, then softmax) is what makes the parameter useful; a
    post-softmax multiplication would break the row-sum-to-1 invariant instead.
    """
    dim, heads = 16, 4
    n = dim // heads
    mdta = MDTA(dim, heads).eval()
    x = _feat(b=1, c=dim, h=6, w=6, seed=9)

    with torch.no_grad():
        mdta.temperature.fill_(0.0)
        _, flat = mdta(x, return_attn=True)
        mdta.temperature.fill_(1e4)
        _, sharp = mdta(x, return_attn=True)

    torch.testing.assert_close(flat, torch.full_like(flat, 1.0 / n),
                               rtol=1e-5, atol=1e-5)
    assert float(sharp.max(dim=-1).values.min()) > 0.99, \
        "large temperature did not sharpen the attention"
    # Row sums survive either extreme.
    for a in (flat, sharp):
        torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)),
                                   rtol=1e-5, atol=1e-5)


def test_mdta_q_and_k_are_l2_normalised_along_the_token_dim():
    """
    q and k are unit-normalised along dim=-1 (the H*W token axis) before
    q @ k^T, which is what keeps the logits bounded in [-1, 1] * temperature.

    Normalising along the channel axis instead (dim=-2) is a one-character
    change that still runs and still produces a valid softmax — so the test
    asserts the module matches the token-dim reference AND measurably differs
    from the channel-dim one.
    """
    dim, heads = 16, 4
    mdta = MDTA(dim, heads).eval()
    x = _feat(b=2, c=dim, h=5, w=5, seed=10)

    with torch.no_grad():
        out, attn = mdta(x, return_attn=True)
        out_tok, attn_tok = _mdta_reference(mdta, x, normalize_dim=-1)
        out_ch, attn_ch = _mdta_reference(mdta, x, normalize_dim=-2)

    torch.testing.assert_close(attn, attn_tok, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out, out_tok, rtol=1e-6, atol=1e-6)
    assert (attn - attn_ch).abs().max() > 1e-3, \
        "token-dim and channel-dim normalisation are indistinguishable here"

    # The normalised rows really are unit vectors over the token axis.
    with torch.no_grad():
        q, k, _ = mdta.qkv_dwconv(mdta.qkv(x)).chunk(3, dim=1)
    q = F.normalize(q.reshape(2, heads, dim // heads, 25), dim=-1)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(2, heads, dim // heads),
                               rtol=1e-5, atol=1e-5)


def test_mdta_forward_matches_an_explicit_reference():
    """
    Full-forward equivalence with a hand-written reference built from the
    module's own submodules: qkv -> dwconv -> chunk -> head reshape -> normalise
    -> temperature -> softmax -> attn @ v -> reshape -> project_out.

    This pins the *order* of every step at once, including that the head split
    is a plain contiguous reshape (heads take consecutive channel groups, the
    einops 'b (head c) h w' convention) and that project_out comes last.
    """
    mdta = MDTA(16, 4).eval()
    x = _feat(b=2, c=16, h=6, w=6, seed=11, scale=2.0)

    with torch.no_grad():
        out, attn = mdta(x, return_attn=True)
        ref_out, ref_attn = _mdta_reference(mdta, x)

    torch.testing.assert_close(out, ref_out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(attn, ref_attn, rtol=1e-6, atol=1e-6)


def test_mdta_return_attn_does_not_change_the_output_tensor():
    """
    The tensor returned with return_attn=True must be bit-identical to the one
    returned with return_attn=False.  The flag is a pure debug/distillation tap;
    if it perturbed the forward pass, distillation runs would not match
    inference runs.
    """
    mdta = MDTA(8, 2).eval()
    x = _feat(b=2, c=8, h=6, w=6, seed=12)

    with torch.no_grad():
        plain = mdta(x)
        out, attn = mdta(x, return_attn=True)

    assert isinstance(plain, torch.Tensor), "return_attn=False must return a bare tensor"
    assert torch.equal(plain, out)
    assert attn.shape == (2, 2, 4, 4)


def test_mdta_requires_num_heads_to_divide_dim():
    """
    Documents ACTUAL behaviour when num_heads does not divide dim: the
    constructor accepts it silently and the failure only surfaces at forward
    time, as a RuntimeError from the head reshape.

    Worth pinning: the models build the bottleneck as
    RestormerBlock(dim=dim*16, num_heads=heads[3]), so a bad `heads` entry in a
    config is not caught at build time — it dies in the middle of the first
    training step.  The reshape can never silently succeed with the wrong data
    (numel mismatch), which is the important safety property.
    """
    bad = MDTA(8, 3)          # constructor does NOT validate
    assert bad.num_heads == 3
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        bad(torch.randn(1, 8, 4, 4))

    # num_heads > dim degenerates to c // heads == 0 and also raises.
    too_many = MDTA(4, 8)
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        too_many(torch.randn(1, 4, 4, 4))


def test_mdta_with_num_heads_equal_dim_gives_degenerate_unit_attention():
    """
    heads == dim leaves one channel per head, so the attention matrix is 1x1 and
    softmax over a single element is exactly 1: MDTA collapses to
    project_out(v).  A useful degenerate-config sanity check — it must not NaN,
    and it must not silently reshape into something else.
    """
    dim = 8
    mdta = MDTA(dim, dim).eval()
    x = _feat(b=1, c=dim, h=4, w=4, seed=13)

    with torch.no_grad():
        out, attn = mdta(x, return_attn=True)
        _, _, v = mdta.qkv_dwconv(mdta.qkv(x)).chunk(3, dim=1)
        expected = mdta.project_out(v)

    assert_shape(attn, (1, dim, 1, 1), "degenerate attention")
    assert torch.equal(attn, torch.ones_like(attn))
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_mdta_is_independent_across_batch_elements():
    """
    Attention is computed per sample, so forwarding a batch must equal
    forwarding each element alone.  Any cross-sample leak (e.g. a stray
    reshape that folds B into the head axis) would make validation PSNR depend
    on the batch composition.
    """
    mdta = MDTA(8, 2).eval()
    x = _feat(b=3, c=8, h=6, w=6, seed=14)

    with torch.no_grad():
        batched = mdta(x)
        one_by_one = torch.cat([mdta(x[i:i + 1]) for i in range(3)], dim=0)

    torch.testing.assert_close(batched, one_by_one, rtol=1e-5, atol=1e-6)


def test_mdta_gradients_reach_every_parameter_including_temperature():
    """
    Every MDTA parameter — temperature included — must receive a finite,
    non-zero gradient.  temperature only gets gradient through the pre-softmax
    multiply; if the softmax were applied first it would still be a Parameter
    but would stop learning, which no shape test would catch.
    """
    mdta = MDTA(8, 2)
    x = _feat(b=2, c=8, h=6, w=6, seed=15).requires_grad_(True)

    mdta(x).pow(2).mean().backward()

    names = {n for n, _ in mdta.named_parameters()}
    assert "temperature" in names
    for name, p in mdta.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert_finite(p.grad, f"grad of {name}")
        assert p.grad.abs().max() > 0, f"grad for {name} is all zeros"
    assert x.grad is not None and x.grad.abs().max() > 0


# =============================================================================
# GDFN
# =============================================================================

@pytest.mark.parametrize("dim,ratio,expected_hidden", [
    (8, 2.66, 21),    # int(21.28) -> truncated, NOT rounded to 21/22 by round()
    (6, 2.66, 15),    # int(15.96) -> 15, whereas round() would give 16
    (8, 1.0, 8),
    (8, 0.5, 4),
    (8, 2.0, 16),
    (16, 2.5, 40),
    (128, 2.66, 340),  # the real bottleneck width: dim*16 for dim=8
])
def test_gdfn_hidden_features_is_truncated_int_of_dim_times_ratio(dim, ratio, expected_hidden):
    """
    hidden_features must be int(dim * mlp_ratio) (truncation), project_in must
    emit 2*hidden (the two gate halves), the depth-wise conv must keep 2*hidden
    with groups=2*hidden, and project_out must map hidden -> dim.

    The truncation matters for checkpoint compatibility: with mlp_ratio=2.66,
    round() instead of int() changes the width for many dims (6 -> 16 vs 15) and
    every `latent.*.ffn.*` tensor in a saved checkpoint stops matching.
    """
    gdfn = GDFN(dim, ratio)

    assert expected_hidden == int(dim * ratio)
    assert gdfn.project_in.in_channels == dim
    assert gdfn.project_in.out_channels == 2 * expected_hidden
    assert gdfn.project_in.kernel_size == (1, 1)
    assert gdfn.dwconv.in_channels == 2 * expected_hidden
    assert gdfn.dwconv.out_channels == 2 * expected_hidden
    assert gdfn.dwconv.groups == 2 * expected_hidden, "dwconv is not depth-wise"
    assert gdfn.dwconv.kernel_size == (3, 3)
    assert gdfn.dwconv.padding == (1, 1)
    assert gdfn.project_out.in_channels == expected_hidden
    assert gdfn.project_out.out_channels == dim


def test_gdfn_default_mlp_ratio_is_restormer_266():
    """The default expansion factor must stay 2.66 (Restormer's published value)."""
    assert GDFN(8).project_out.in_channels == int(8 * 2.66) == 21


@pytest.mark.parametrize("shape", [(1, 8, 1, 1), (1, 8, 3, 5), (2, 8, 8, 8), (2, 8, 9, 16)])
def test_gdfn_preserves_shape(shape):
    """GDFN sits inside a residual add, so it must return exactly its input shape."""
    gdfn = GDFN(shape[1])
    y = gdfn(torch.randn(*shape))
    assert_shape(y, shape, "GDFN output")
    assert_finite(y, "GDFN output")


def test_gdfn_gate_is_gelu_of_the_first_half_times_the_second_half():
    """
    The gate must be gelu(x1) * x2 with x1 the FIRST chunk — matched against a
    reference, and contrasted with the swapped gelu(x2) * x1.

    Both orders run and both are shape-correct, so only a differential test can
    catch a swap.  The order decides which half carries the non-linearity, i.e.
    which half a trained checkpoint's weights belong to.
    """
    gdfn = GDFN(8, 2.66).eval()
    x = _feat(b=2, c=8, h=5, w=5, seed=16, scale=2.0)

    with torch.no_grad():
        out = gdfn(x)
        ref = _gdfn_reference(gdfn, x, swap=False)
        swapped = _gdfn_reference(gdfn, x, swap=True)

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    assert (out - swapped).abs().max() > 1e-4, \
        "the two gate orders are indistinguishable here — test is toothless"


def test_gdfn_gate_is_multiplicative_so_a_zeroed_half_zeroes_the_output():
    """
    Zeroing the depth-wise weights of the second half forces x2 == 0, and a
    multiplicative gate must then output exactly 0 (bias=False everywhere).

    This pins the `*` — an additive `gelu(x1) + x2` would leave a large residual
    signal here, and would quietly turn the "Gated" FFN into a plain one.
    """
    gdfn = GDFN(8, 2.66).eval()
    hidden = gdfn.project_out.in_channels
    with torch.no_grad():
        gdfn.dwconv.weight[hidden:].zero_()      # kill x2
    x = _feat(b=1, c=8, h=4, w=4, seed=17)

    with torch.no_grad():
        out = gdfn(x)

    assert torch.equal(out, torch.zeros_like(out)), \
        "output survived a zeroed gate half — the gate is not multiplicative"


def test_gdfn_gradients_reach_every_parameter():
    """All three convs must train; a detached branch would silently freeze the FFN."""
    gdfn = GDFN(8, 2.66)
    x = _feat(b=2, c=8, h=5, w=5, seed=18).requires_grad_(True)

    gdfn(x).pow(2).mean().backward()

    for name, p in gdfn.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert_finite(p.grad, f"grad of {name}")
        assert p.grad.abs().max() > 0, f"grad for {name} is all zeros"
    assert x.grad is not None and x.grad.abs().max() > 0


# =============================================================================
# RestormerBlock
# =============================================================================

def test_block_submodules_are_prenorm_attn_prenorm_ffn():
    """
    The block must be exactly norm1 -> attn (MDTA) -> norm2 -> ffn (GDFN), each
    sized for `dim`, and the parameter names must stay put: checkpoints store
    them as `latent.<i>.norm1.weight`, `latent.<i>.attn.temperature`, ...
    """
    dim, heads = 16, 4
    blk = RestormerBlock(dim=dim, num_heads=heads)

    assert isinstance(blk.norm1, LayerNorm) and isinstance(blk.norm2, LayerNorm)
    assert isinstance(blk.attn, MDTA) and isinstance(blk.ffn, GDFN)
    assert_shape(blk.norm1.weight, (1, dim, 1, 1), "norm1.weight")
    assert_shape(blk.norm2.weight, (1, dim, 1, 1), "norm2.weight")
    assert blk.attn.num_heads == heads
    assert blk.ffn.project_out.out_channels == dim

    assert {n for n, _ in blk.named_parameters()} == {
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        "attn.temperature", "attn.qkv.weight", "attn.qkv_dwconv.weight",
        "attn.project_out.weight",
        "ffn.project_in.weight", "ffn.dwconv.weight", "ffn.project_out.weight",
    }


def test_block_matches_the_double_residual_reference():
    """
    Output must be exactly  t = x + attn(norm1(x));  t + ffn(norm2(t)).

    Note the second residual branches off `t`, not `x` — a classic transcription
    slip (`x + ffn(norm2(t))`) that changes the function while keeping every
    shape valid.  The reference makes the difference observable.
    """
    blk = RestormerBlock(dim=16, num_heads=4).eval()
    x = _feat(b=2, c=16, h=8, w=8, seed=19, scale=2.0)

    with torch.no_grad():
        t = x + blk.attn(blk.norm1(x))
        expected = t + blk.ffn(blk.norm2(t))
        wrong_residual = x + blk.ffn(blk.norm2(t))
        out = blk(x)

    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
    assert (out - wrong_residual).abs().max() > 1e-4, \
        "cannot distinguish the second residual's source — test is toothless"


def test_block_output_differs_from_input_but_is_a_bounded_perturbation():
    """
    A freshly initialised block must actually do something (output != input) yet
    stay a residual perturbation rather than an unbounded blow-up.

    An identity-at-init block would mean the sublayers are dead (e.g. an
    accidental zero init); an explosion would mean the residual is missing.
    """
    blk = RestormerBlock(dim=16, num_heads=4).eval()
    x = _feat(b=2, c=16, h=8, w=8, seed=20)

    with torch.no_grad():
        out = blk(x)

    assert_shape(out, x.shape, "block output")
    assert_finite(out, "block output")
    assert not torch.allclose(out, x), "block behaves as the identity at init"
    assert (out - x).abs().max() < 50.0 * x.abs().max(), \
        "residual perturbation is unreasonably large"


def test_block_with_zeroed_sublayer_projections_is_the_exact_identity():
    """
    Zeroing only attn.project_out and ffn.project_out must make the block a
    BIT-EXACT identity, because both sublayers end in those 1x1 convs and both
    are wrapped in a plain additive residual.

    This is the strongest possible statement of "the residual path is a clean
    `x + f(x)`": any extra scaling, an in-place norm on the residual stream, or
    a missing skip would show up as a non-zero difference.
    """
    blk = RestormerBlock(dim=8, num_heads=2).eval()
    with torch.no_grad():
        blk.attn.project_out.weight.zero_()
        blk.ffn.project_out.weight.zero_()
    x = _feat(b=2, c=8, h=8, w=8, seed=21, scale=3.0, offset=1.0)

    with torch.no_grad():
        out = blk(x)

    assert torch.equal(out, x), "zeroed sublayers did not return the input exactly"


def test_block_return_attn_returns_the_tensor_and_the_channel_attention_map():
    """
    return_attn=True must give (tensor, attn) with the tensor bit-identical to
    the plain forward and attn == the MDTA map for norm1(x), shaped
    [B, heads, C//heads, C//heads].

    The comment marks this map as the teacher/student "common space"; if the
    flag perturbed the output or returned a post-FFN map, the distillation
    target would not correspond to the deployed forward pass.
    """
    b, dim, heads = 2, 16, 4
    blk = RestormerBlock(dim=dim, num_heads=heads).eval()
    x = _feat(b=b, c=dim, h=6, w=6, seed=22)

    with torch.no_grad():
        plain = blk(x)
        result = blk(x, return_attn=True)

    assert isinstance(result, tuple) and len(result) == 2
    out, attn = result
    assert_shape(out, (b, dim, 6, 6), "block output")
    assert_shape(attn, (b, heads, dim // heads, dim // heads), "block attention")
    assert torch.equal(out, plain), "return_attn changed the output tensor"

    with torch.no_grad():
        _, expected_attn = blk.attn(blk.norm1(x), return_attn=True)
    torch.testing.assert_close(attn, expected_attn, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(attn.sum(-1), torch.ones_like(attn.sum(-1)),
                               rtol=1e-5, atol=1e-5)


def test_block_accepts_and_ignores_window_size_and_shift_size():
    """
    The docstring advertises the block as a drop-in for SwinTransformerBlock
    with window_size/shift_size ignored.  Passing them (plus other Swin kwargs)
    must not raise, must not store attributes, and must not change a single
    parameter or output value versus the plain construction under the same seed.
    """
    x = _feat(b=1, c=8, h=8, w=8, seed=23)

    torch.manual_seed(99)
    plain = RestormerBlock(dim=8, num_heads=2).eval()
    torch.manual_seed(99)
    swin_style = RestormerBlock(dim=8, num_heads=2, window_size=8, shift_size=4,
                                input_resolution=(32, 32), qkv_bias=True,
                                drop_path=0.1).eval()

    for attr in ("window_size", "shift_size", "input_resolution", "drop_path"):
        assert not hasattr(swin_style, attr), f"{attr} was unexpectedly stored"

    sd_a, sd_b = plain.state_dict(), swin_style.state_dict()
    assert sd_a.keys() == sd_b.keys()
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    with torch.no_grad():
        torch.testing.assert_close(plain(x), swin_style(x), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 8, 1, 1), (1, 8, 3, 5), (2, 8, 8, 8), (1, 8, 16, 9)])
def test_block_preserves_shape_for_odd_and_tiny_spatial_sizes(shape):
    """
    The bottleneck runs on latents whose size is the packed size / 8, so small
    and non-square maps are normal.  Shape must be preserved with no window
    padding requirement (the whole point of being window-free).
    """
    blk = RestormerBlock(dim=shape[1], num_heads=2).eval()
    with torch.no_grad():
        out = blk(torch.randn(*shape))
    assert_shape(out, shape, "block output")
    assert_finite(out, "block output")


def test_block_is_identical_in_train_and_eval_mode():
    """
    There is no dropout, no BatchNorm and no stochastic depth anywhere, so
    train() and eval() must produce bit-identical outputs.

    The training script flips modes between phases and the eval script relies on
    eval-mode inference matching what was trained; a future dropout added here
    would break that silently.
    """
    blk = RestormerBlock(dim=8, num_heads=2)
    x = _feat(b=2, c=8, h=8, w=8, seed=24)

    blk.train()
    with torch.no_grad():
        y_train = blk(x)
    blk.eval()
    with torch.no_grad():
        y_eval = blk(x)

    assert torch.equal(y_train, y_eval), "train/eval mode changed the output"
    assert not any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.BatchNorm2d))
                   for m in blk.modules()), "unexpected stochastic/stateful layer"


def test_block_is_deterministic_for_the_same_seed_and_input():
    """
    Same seed -> same parameters -> same output, and repeated forwards on one
    instance are bit-identical.  Reproducibility is what makes a PSNR
    regression in the training log meaningful.
    """
    torch.manual_seed(555)
    a = RestormerBlock(dim=8, num_heads=2).eval()
    torch.manual_seed(555)
    b = RestormerBlock(dim=8, num_heads=2).eval()
    x = _feat(b=2, c=8, h=8, w=8, seed=25)

    with torch.no_grad():
        ya1, ya2, yb = a(x), a(x), b(x)

    assert torch.equal(ya1, ya2), "two forwards of one block disagree"
    assert torch.equal(ya1, yb), "same-seed blocks disagree"


def test_block_is_independent_across_batch_elements():
    """
    Everything in the block (channel LayerNorm, depth-wise convs, per-sample
    attention) is per-sample, so batching must not change the result.  The
    training script pads batches to the max size (collate_pad_to_max), so a
    batch-coupled bottleneck would make results depend on padding partners.
    """
    blk = RestormerBlock(dim=8, num_heads=2).eval()
    x = _feat(b=3, c=8, h=8, w=8, seed=26)

    with torch.no_grad():
        batched = blk(x)
        singles = torch.cat([blk(x[i:i + 1]) for i in range(3)], dim=0)

    torch.testing.assert_close(batched, singles, rtol=1e-5, atol=1e-6)


def test_block_gradients_reach_every_parameter_and_the_input():
    """
    A single backward must produce finite, non-zero gradients for ALL 11
    parameters of the block and for the input (so the encoder upstream trains).

    A dead parameter here is invisible in the loss but wastes capacity; a dead
    input gradient would cut the encoder off from the loss entirely.
    """
    blk = RestormerBlock(dim=16, num_heads=4)
    x = _feat(b=2, c=16, h=8, w=8, seed=27).requires_grad_(True)

    blk(x).pow(2).mean().backward()

    params = dict(blk.named_parameters())
    assert len(params) == 11, f"unexpected parameter set: {sorted(params)}"
    dead = [n for n, p in params.items()
            if p.grad is None or not torch.isfinite(p.grad).all()
            or float(p.grad.abs().max()) == 0.0]
    assert dead == [], f"parameters with missing/zero/non-finite grads: {dead}"
    assert x.grad is not None
    assert_finite(x.grad, "grad wrt input")
    assert float(x.grad.abs().max()) > 0


def test_block_gradients_flow_through_the_return_attn_path_too():
    """
    The return_attn branch is a separate code path; its output must still be
    differentiable back to every parameter (distillation losses backprop through
    both the feature output and the attention map).
    """
    blk = RestormerBlock(dim=8, num_heads=2)
    x = _feat(b=1, c=8, h=6, w=6, seed=28).requires_grad_(True)

    out, attn = blk(x, return_attn=True)
    (out.pow(2).mean() + attn.pow(2).mean()).backward()

    dead = [n for n, p in blk.named_parameters()
            if p.grad is None or float(p.grad.abs().max()) == 0.0]
    assert dead == [], f"parameters with missing/zero grads: {dead}"
    assert x.grad is not None and float(x.grad.abs().max()) > 0


def test_block_state_dict_round_trips_through_disk(tmp_path):
    """
    Saving and strictly reloading the state_dict must reproduce the exact
    outputs.  The two-phase training script resumes from checkpoints of the
    whole model, and `latent.*` keys come from this block — a key mismatch or a
    non-persistent buffer here breaks resume with strict=True.
    """
    blk = RestormerBlock(dim=8, num_heads=2).eval()
    x = _feat(b=1, c=8, h=8, w=8, seed=29)
    with torch.no_grad():
        before = blk(x)

    path = tmp_path / "block.pt"
    torch.save(blk.state_dict(), path)

    fresh = RestormerBlock(dim=8, num_heads=2).eval()
    with torch.no_grad():
        assert not torch.allclose(fresh(x), before), "fresh block already matches"
    missing, unexpected = fresh.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True)
    assert not missing and not unexpected

    with torch.no_grad():
        torch.testing.assert_close(fresh(x), before, rtol=0, atol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_block_bias_flag_reaches_both_sublayers(bias):
    """
    `bias` must be forwarded to MDTA and GDFN (RestormerBlock passes it
    positionally), so every conv gets or loses its bias together.

    A mis-ordered positional argument would silently land in `mlp_ratio` and
    resize the FFN, which is exactly the kind of slip that only surfaces as a
    checkpoint shape mismatch weeks later.
    """
    blk = RestormerBlock(dim=8, num_heads=2, bias=bias)

    conv_names = ["attn.qkv", "attn.qkv_dwconv", "attn.project_out",
                  "ffn.project_in", "ffn.dwconv", "ffn.project_out"]
    mods = dict(blk.named_modules())
    for name in conv_names:
        conv = mods[name]
        if bias:
            assert conv.bias is not None, f"{name} should have a bias"
        else:
            assert conv.bias is None, f"{name} should not have a bias"

    # mlp_ratio must be untouched by the bias flag.
    assert blk.ffn.project_out.in_channels == int(8 * 2.66) == 21

    x = _feat(b=1, c=8, h=8, w=8, seed=30)
    with torch.no_grad():
        assert_finite(blk.eval()(x), "block output")


def test_block_stack_at_the_real_bottleneck_width_preserves_shape():
    """
    Exercise the exact configuration the models build:
    nn.Sequential of RestormerBlock(dim=dim*16, num_heads=heads[3]) with the
    tiny architecture (dim=8 -> 128 latent channels, heads=1), on a latent the
    size a 32x32 packed input produces (32 / 8 = 4).

    A stack must remain shape-preserving so that up_shuffle_3_2 and the skip
    concatenation downstream still line up.
    """
    latent_dim = 8 * 16
    stack = nn.Sequential(*[RestormerBlock(dim=latent_dim, num_heads=1)
                            for _ in range(2)]).eval()
    x = _feat(b=1, c=latent_dim, h=4, w=4, seed=31)

    with torch.no_grad():
        out = stack(x)

    assert_shape(out, (1, latent_dim, 4, 4), "latent stack output")
    assert_finite(out, "latent stack output")
    assert not torch.allclose(out, x)
    # Per-block attention at this width is 128x128 per head, not 16x16 spatial.
    with torch.no_grad():
        _, attn = stack[0](x, return_attn=True)
    assert_shape(attn, (1, 1, latent_dim, latent_dim), "latent attention")


def test_block_handles_a_non_contiguous_input():
    """
    Feature maps arriving from a transpose/permute are non-contiguous; the head
    reshape inside MDTA must cope (reshape, not view).  The decoder feeds the
    bottleneck from PixelUnshuffle output, and D4 augmentation
    (make_d4_transform) transposes tensors upstream, so non-contiguity is real.
    """
    blk = RestormerBlock(dim=8, num_heads=2).eval()
    base = _feat(b=1, c=8, h=6, w=10, seed=32)
    x = base.transpose(2, 3)          # [1, 8, 10, 6], non-contiguous
    assert not x.is_contiguous()

    with torch.no_grad():
        out = blk(x)
        out_contig = blk(x.contiguous())

    assert_shape(out, (1, 8, 10, 6), "block output")
    torch.testing.assert_close(out, out_contig, rtol=1e-6, atol=1e-6)


def test_block_deepcopy_is_an_independent_module():
    """
    copy.deepcopy of a block must be functionally identical yet share no
    parameter storage — the pattern used when an EMA/teacher copy is made.
    A shared parameter would let the student's optimizer silently update the
    teacher.
    """
    blk = RestormerBlock(dim=8, num_heads=2).eval()
    clone = copy.deepcopy(blk).eval()
    x = _feat(b=1, c=8, h=8, w=8, seed=33)

    with torch.no_grad():
        torch.testing.assert_close(blk(x), clone(x), rtol=0, atol=0)
        clone.attn.project_out.weight.add_(1.0)
        assert not torch.allclose(blk(x), clone(x)), "parameters are shared"


@pytest.mark.gpu
def test_block_runs_under_gpu_autocast_without_nan(gpu_device):
    """
    Training runs under torch.autocast(bfloat16) on A100s (Polaris) and on
    Intel Max 1550s (Aurora); the softmax and the 1/sqrt(var+eps) in
    LayerNorm are the two spots where reduced precision can produce NaN/Inf,
    and the two backends do not round identically. Auto-skips on a login node.
    """
    dev = gpu_device
    blk = RestormerBlock(dim=16, num_heads=4).to(dev).eval()
    x = _feat(b=2, c=16, h=8, w=8, seed=34, scale=4.0).to(dev)

    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
        out, attn = blk(x, return_attn=True)

    assert_shape(out, (2, 16, 8, 8), "block output")
    assert_finite(out.float(), "autocast block output")
    assert_finite(attn.float(), "autocast attention")
    torch.testing.assert_close(attn.float().sum(-1),
                               torch.ones(2, 4, 4, device=dev),
                               rtol=1e-2, atol=1e-2)
