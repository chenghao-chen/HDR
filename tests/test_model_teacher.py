"""
Contract tests for TransUNet_Teacher_HDR (HDR_model_hybrid_Teacher.py).

The teacher is the reference joint denoising + demosaicing network and the
template every other model in the repo copies: MoEDenoiser reuses its trunk
verbatim, DualSNRDenoiser instantiates two of them, SingleDenoiser wraps one,
and the eval script's tiled inference assumes its exact resolution contract.

What is pinned here, and why each item is load-bearing:

  * the shape contract  [B, 4, h, w] -> [B, 3, 2h, 2w]  (h, w PACKED dims):
    the internal 2x upsampling IS the demosaicing step, so an off-by-one or a
    transposed axis silently misaligns every loss, PSNR and saved JPEG;
  * the h, w % 8 == 0 requirement (three PixelUnshuffle(2) encoder stages) —
    illegal sizes must fail loudly, never return a wrong-sized image, because
    infer_patches/infer_full would happily stitch the result;
  * the asymmetric clamp: always >= _CLAMP_EPS, <= 1.0 only in eval();
  * the PixelShuffle semantics of the RGB head (which to_rgb channel lands on
    which output channel and which intra-cell offset);
  * return_maps exposing the encoder/decoder ladder at the implied resolutions,
    which is what the distillation loss selects feature maps from;
  * se_reduction=None staying binary compatible with pre-SE checkpoints;
  * gradient reachability from the output back to patch_embed, the Restormer
    latent and to_rgb.

Everything runs on CPU with the tiny (dim=8) architecture and packed sizes <= 40.
"""

import pytest
import torch
import torch.nn as nn

from HDR_model_hybrid_Teacher import (
    _CLAMP_EPS,
    SqueezeExcite,
    TransUNet_Teacher_HDR,
)

from helpers import assert_finite, assert_shape, packed_bayer, snr_map_for


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — tests/helpers.py is shared)
# ─────────────────────────────────────────────────────────────────────────────

def _teacher(tiny_kwargs, **override):
    """Tiny teacher; `override` patches the tiny_kwargs dict (e.g. se_reduction)."""
    kw = dict(tiny_kwargs)
    kw.update(override)
    return TransUNet_Teacher_HDR(**kw)


def _drive_to_rgb(model, bias):
    """
    Force the raw (pre-clamp) network output to a known constant pattern.

    to_rgb is a 1x1 conv followed only by PixelShuffle(2) (a pure permutation),
    so zeroing its weight and setting its bias makes the pre-clamp image exactly
    predictable: every pixel of the sub-image fed by to_rgb output channel k
    equals bias[k].  `bias` may be a scalar or a per-channel sequence.
    """
    with torch.no_grad():
        model.to_rgb.weight.zero_()
        if isinstance(bias, (int, float)):
            model.to_rgb.bias.fill_(float(bias))
        else:
            model.to_rgb.bias.copy_(torch.tensor(bias, dtype=model.to_rgb.bias.dtype))
    return model


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core shape contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("batch,h,w", [
    (1, 8, 8),      # smallest legal packed size — the latent collapses to 1x1
    (1, 32, 32),    # the canonical training tile
    (2, 16, 16),    # B > 1
    (3, 8, 16),     # B > 1 and non-square
    (1, 32, 40),    # non-square, w > h
    (2, 24, 8),     # non-square, h > w
])
def test_output_shape_is_double_the_packed_resolution(tiny_kwargs, batch, h, w):
    """
    [B, 4, h, w] -> [B, 3, 2h, 2w] for square, non-square and batched inputs.

    The 2x factor is the demosaicing: the network, not an external GBTF pass,
    produces sensor resolution.  Non-square sizes are included because the
    dataset's crops and the eval script's edge tiles are frequently h != w, and
    a swapped H/W in any reshape would only show up there.
    """
    model = _teacher(tiny_kwargs).eval()
    x = packed_bayer(batch, h, w, seed=batch * 100 + h)
    with torch.no_grad():
        out = model(x)
    assert_shape(out, (batch, 3, 2 * h, 2 * w), "teacher output")
    assert_finite(out, "teacher output")
    assert out.dtype == x.dtype, "forward changed the tensor dtype"


def test_output_is_not_spatially_degenerate(tiny_kwargs):
    """
    A randomly initialised teacher must produce a spatially varying image.

    Every other test here compares shapes or clamps; this one guards the case
    where the head is effectively dead (all pixels pinned to the clamp floor),
    which would make PSNR-based tests pass trivially on a broken network.
    """
    model = _teacher(tiny_kwargs).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 32, 32, seed=3))
    assert float(out.std()) > 0.0, "output is constant — the RGB head is dead"
    assert float(out.max()) > _CLAMP_EPS, "every pixel sits on the clamp floor"


@pytest.mark.parametrize("out_channels", [1, 3, 4])
def test_out_channels_argument_sets_the_output_channel_count(tiny_kwargs, out_channels):
    """
    out_channels sizes to_rgb (-> out_channels*4, then PixelShuffle(2)), so the
    output channel count must equal it exactly while the 2x upsampling is
    unchanged.  Pins that the head is parameterised rather than hardcoded to 3.
    """
    model = _teacher(tiny_kwargs, out_channels=out_channels).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 16, 16, seed=7))
    assert_shape(out, (1, out_channels, 32, 32), f"output(out_channels={out_channels})")


def test_batch_items_are_processed_independently(tiny_kwargs):
    """
    Forwarding a batch must equal forwarding each sample on its own.

    The network contains SqueezeExcite (a global average pool) and a
    channel-wise LayerNorm; either would leak statistics across the batch if it
    reduced over the wrong dim.  Training batches are built by
    collate_pad_to_max, so cross-sample leakage would make a sample's output
    depend on its neighbours' padding.
    """
    model = _teacher(tiny_kwargs).eval()
    x = packed_bayer(3, 16, 16, seed=11)
    with torch.no_grad():
        batched = model(x)
        per_item = torch.cat([model(x[i:i + 1]) for i in range(x.shape[0])], dim=0)
    assert torch.allclose(batched, per_item, atol=1e-6), \
        f"batch leakage: max |diff| = {float((batched - per_item).abs().max())}"


def test_forward_is_deterministic_and_leaves_its_input_alone(tiny_kwargs):
    """
    Two identical forwards give bit-identical results, and the input tensor is
    untouched.  There is no dropout or BatchNorm in the teacher, so any
    nondeterminism would be a defect; an in-place input mutation would corrupt
    the dataset tensor that the training loop also uses for the SNR map and the
    loss target.
    """
    model = _teacher(tiny_kwargs).eval()
    x = packed_bayer(1, 16, 16, seed=13)
    reference = x.clone()
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b), "forward is not deterministic in eval mode"
    assert torch.equal(x, reference), "forward mutated its input in place"


# ─────────────────────────────────────────────────────────────────────────────
# 2. The h, w % 8 == 0 requirement
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("h,w,axis", [
    (12, 12, "height"),   # divisible by 4, not 8 -> dies at the 3rd unshuffle
    (20, 20, "height"),   # divisible by 4, not 8
    (16, 12, "width"),    # only w illegal
    (12, 8, "height"),    # only h illegal
    (4, 4, "height"),     # divisible by 4, not 8
    (9, 16, "height"),    # odd -> dies at the very first (Bayer) unshuffle
])
def test_packed_dims_not_divisible_by_eight_fail_loudly(tiny_kwargs, h, w, axis):
    """
    Documented behaviour for an illegal packed size.

    The docstring requires h, w % 8 == 0 (three PixelUnshuffle(2) stages).
    TransUNet_Teacher_HDR adds no explicit guard, so the failure surfaces as
    torch's own RuntimeError from pixel_unshuffle, e.g. for h=12:

        pixel_unshuffle expects height to be divisible by downscale_factor,
        but input.size(-2)=3 is not divisible by 2

    That is acceptable: it names the op, the offending axis and the rule, so it
    is diagnosable.  What matters for this project is that it RAISES — a
    silently wrong-sized output would be a real bug, because infer_full pads to
    a multiple and would stitch the result without noticing.  (Note the message
    quotes the failing INTERMEDIATE size, 3, not the caller's packed dim, 12.)
    """
    model = _teacher(tiny_kwargs).eval()
    x = packed_bayer(1, h, w, seed=17)
    with pytest.raises(RuntimeError) as excinfo:
        with torch.no_grad():
            model(x)
    msg = str(excinfo.value)
    assert "pixel_unshuffle" in msg, f"unhelpful error for {h}x{w}: {msg}"
    assert "divisible" in msg, f"unhelpful error for {h}x{w}: {msg}"
    assert axis in msg, f"error blames the wrong axis for {h}x{w}: {msg}"


@pytest.mark.parametrize("h,w", [(8, 8), (16, 24), (32, 32)])
def test_every_multiple_of_eight_is_accepted(tiny_kwargs, h, w):
    """
    The complement of the test above: the documented-legal sizes really work,
    including the degenerate h=w=8 case where the latent is a single 1x1 token
    (MDTA then L2-normalises over a length-1 spatial axis, which must not NaN).
    """
    model = _teacher(tiny_kwargs).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, h, w, seed=19))
    assert_shape(out, (1, 3, 2 * h, 2 * w), f"output({h}x{w})")
    assert_finite(out, f"output({h}x{w})")


def test_unbatched_input_is_rejected(tiny_kwargs):
    """
    The dataset hands out unbatched [4, h, w] tensors and estimate_local_snr_map
    accepts them, so calling the teacher without a batch dim is an easy mistake.
    Documented behaviour: it raises (the Restormer LayerNorm's (1, C, 1, 1)
    parameters broadcast a 3-D tensor up to 4-D, and the mismatch is caught at
    the first decoder skip concat).  The message is obscure, but nothing silent
    happens — that is what this test guarantees.
    """
    model = _teacher(tiny_kwargs).eval()
    with pytest.raises(RuntimeError, match="same number of dimensions"):
        with torch.no_grad():
            model(packed_bayer(1, 16, 16, seed=21)[0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Clamp semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_clamp_eps_is_one_lsb_of_a_20_bit_container():
    """_CLAMP_EPS is 1/(2**20 - 1); the mu-law tone mapping and the log-domain
    losses divide by the output, so the exact floor is part of the contract."""
    assert _CLAMP_EPS == 1.0 / (2 ** 20 - 1)


def test_train_mode_lets_the_output_exceed_one(tiny_kwargs):
    """
    In train() the output is clamped BELOW only.  HDR targets exceed 1.0 after
    normalisation, and an upper clamp during training would zero the gradient of
    every over-bright pixel, so this asymmetry is deliberate and must hold.
    """
    model = _drive_to_rgb(_teacher(tiny_kwargs), 5.0).train()
    out = model(packed_bayer(1, 16, 16, seed=23))
    assert float(out.max()) == pytest.approx(5.0, rel=1e-6), \
        f"train-mode output was clamped to {float(out.max())}"


def test_eval_mode_clamps_the_output_at_one(tiny_kwargs):
    """
    In eval() the same driven network must saturate at exactly 1.0: the eval
    script writes 8-bit JPEGs and computes PSNR against [0, 1] ground truth, so
    an unclamped inference output would wrap or blow up the metric.
    """
    model = _drive_to_rgb(_teacher(tiny_kwargs), 5.0).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 16, 16, seed=23))
    assert float(out.max()) == pytest.approx(1.0, rel=1e-6)
    assert float(out.min()) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_lower_clamp_applies_in_both_modes(tiny_kwargs, mode):
    """
    The >= _CLAMP_EPS floor is unconditional — a strictly positive output is
    what makes the log/mu-law tone mapping safe.  Driving the raw output to -5
    must therefore yield exactly _CLAMP_EPS in train() as well as eval().
    """
    model = _drive_to_rgb(_teacher(tiny_kwargs), -5.0)
    getattr(model, mode)()
    with torch.set_grad_enabled(mode == "train"):
        out = model(packed_bayer(1, 16, 16, seed=29))
    assert float(out.min()) == pytest.approx(_CLAMP_EPS, rel=1e-9)
    assert float(out.max()) == pytest.approx(_CLAMP_EPS, rel=1e-9)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_randomly_initialised_output_never_dips_below_the_floor(tiny_kwargs, mode):
    """
    The same floor on an ordinary (undriven) forward: the guarantee must not
    depend on the weights, because the trainer's very first steps run with a
    random head and still feed the output to a log-domain loss.
    """
    model = _teacher(tiny_kwargs)
    getattr(model, mode)()
    x = packed_bayer(2, 16, 16, seed=31)
    with torch.set_grad_enabled(mode == "train"):
        out = model(x)
    assert float(out.min()) >= _CLAMP_EPS - 1e-12, \
        f"{mode} output min {float(out.min())} < _CLAMP_EPS"


def test_eval_equals_train_clamped_at_one_on_a_driven_head(tiny_kwargs):
    """
    The ONLY difference between the two modes is the upper clamp.  Driven above
    1.0 so the clamp really bites, eval(x) must equal train(x).clamp(max=1).
    """
    model = _drive_to_rgb(_teacher(tiny_kwargs), 2.5)
    x = packed_bayer(1, 16, 16, seed=37)
    model.train()
    train_out = model(x).detach()
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert float(train_out.max()) > 1.0, "test setup failed to exceed 1.0"
    assert torch.equal(eval_out, train_out.clamp(max=1.0))


def test_eval_equals_train_on_an_ordinary_forward(tiny_kwargs):
    """
    Same claim without driving the head: no dropout, no BatchNorm, no running
    statistics anywhere, so switching mode must not perturb an in-range output.
    This is what lets the trainer trust that validation PSNR measures exactly
    the function it optimised.
    """
    model = _teacher(tiny_kwargs)
    x = packed_bayer(1, 16, 16, seed=39)
    model.train()
    with torch.no_grad():
        train_out = model(x)
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert torch.equal(eval_out, train_out.clamp(max=1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 4. The RGB head: PixelShuffle channel / offset semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_to_rgb_channel_groups_map_to_output_channels_in_order(tiny_kwargs):
    """
    upshuffle_rgb = PixelShuffle(2) consumes to_rgb's out_channels*4 channels in
    contiguous groups of 4: channels 0-3 become output channel 0 (R), 4-7 become
    1 (G), 8-11 become 2 (B).  Anyone widening the head or reordering that conv
    has to preserve this grouping or the saved images come out colour-swapped —
    exactly the failure mode the packed-Bayer channel-offset contract exists to
    prevent on the input side.
    """
    model = _teacher(tiny_kwargs)
    _drive_to_rgb(model, [0.1] * 4 + [0.5] * 4 + [0.9] * 4).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 16, 16, seed=41))
    for ch, value in enumerate((0.1, 0.5, 0.9)):
        plane = out[:, ch]
        assert float(plane.min()) == pytest.approx(value, abs=1e-6)
        assert float(plane.max()) == pytest.approx(value, abs=1e-6)


def test_to_rgb_subchannels_map_to_the_four_intra_cell_offsets(tiny_kwargs):
    """
    Within one output channel, PixelShuffle(2) sends to_rgb sub-channel k to
    intra-cell offset (k//2, k%2) of the 2x-upsampled image — the same rule the
    packed BGGR input follows.  Pinning it documents which quarter of the
    output each head channel is responsible for, which is what a future
    sub-pixel-aware loss or a per-offset residual head would depend on.
    """
    model = _teacher(tiny_kwargs)
    _drive_to_rgb(model, [0.1, 0.2, 0.3, 0.4] + [0.5] * 8).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 16, 16, seed=43))
    red = out[0, 0]
    expected = {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.4}
    for (dy, dx), value in expected.items():
        quarter = red[dy::2, dx::2]
        assert float(quarter.min()) == pytest.approx(value, abs=1e-6), \
            f"offset ({dy},{dx}) is not fed by to_rgb channel {dy * 2 + dx}"
        assert float(quarter.max()) == pytest.approx(value, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 5. return_maps: the encoder / decoder ladder
# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_MAP_KEYS = {
    "x_level1", "x_level2", "x_latent", "w_level2", "w_level1", "w_level0_refined",
}


def test_return_maps_exposes_exactly_the_documented_keys(tiny_kwargs):
    """
    Distillation selects feature maps by name, and the lookup happens in the
    student loss far away from here, so a renamed or dropped key would silently
    remove a loss term rather than crash.
    """
    model = _teacher(tiny_kwargs).eval()
    with torch.no_grad():
        out, maps = model(packed_bayer(1, 32, 32, seed=45), return_maps=True)
    assert isinstance(maps, dict)
    assert set(maps) == _EXPECTED_MAP_KEYS, \
        f"map keys differ: {sorted(set(maps) ^ _EXPECTED_MAP_KEYS)}"
    assert_shape(out, (1, 3, 64, 64), "output")


@pytest.mark.parametrize("h,w", [(32, 32), (16, 24), (8, 8)])
def test_return_maps_resolutions_and_widths_match_the_architecture(tiny_kwargs, h, w):
    """
    Pins the whole U-Net ladder in one place:

      x_level1          dim      @ h/2, w/2   (Bayer unshuffle + patch_embed)
      x_level2          dim*4    @ h/4, w/4
      x_latent          dim*16   @ h/8, w/8   (Restormer bottleneck)
      w_level2          dim*4    @ h/4, w/4   (after reduce_chan_level_2)
      w_level1          dim*2    @ h/2, w/2   (skip concat with x_level1)
      w_level0_refined  dim//2   @ h,   w     (packed Bayer res, pre-to_rgb)

    A wrong stride or shuffle anywhere in the encoder or decoder shows up here
    instead of being absorbed by the final output shape, and the widths are the
    numbers a student network has to match to distil from these maps.
    """
    dim = tiny_kwargs["dim"]
    model = _teacher(tiny_kwargs).eval()
    with torch.no_grad():
        _, maps = model(packed_bayer(1, h, w, seed=47), return_maps=True)

    expected = {
        "x_level1":         (1, dim,      h // 2, w // 2),
        "x_level2":         (1, dim * 4,  h // 4, w // 4),
        "x_latent":         (1, dim * 16, h // 8, w // 8),
        "w_level2":         (1, dim * 4,  h // 4, w // 4),
        "w_level1":         (1, dim * 2,  h // 2, w // 2),
        "w_level0_refined": (1, dim // 2, h,      w),
    }
    for key, shape in expected.items():
        assert_shape(maps[key], shape, key)
        assert_finite(maps[key], key)


def test_return_maps_output_is_identical_to_the_plain_forward(tiny_kwargs):
    """
    return_maps must be a pure observation flag: the image it returns has to
    match the flagless forward bit for bit, or distillation and inference would
    be optimising two different functions.
    """
    model = _teacher(tiny_kwargs).eval()
    x = packed_bayer(1, 16, 16, seed=49)
    with torch.no_grad():
        plain = model(x)
        with_maps, _ = model(x, return_maps=True)
    assert torch.equal(plain, with_maps)


def test_return_maps_default_returns_a_bare_tensor(tiny_kwargs):
    """Without the flag the forward returns a Tensor, not a 2-tuple — every
    call site in the train and eval scripts unpacks it as a single tensor."""
    model = _teacher(tiny_kwargs).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 16, 16, seed=51))
    assert isinstance(out, torch.Tensor)


def test_returned_maps_stay_attached_to_the_autograd_graph(tiny_kwargs):
    """
    Distillation backprops through these maps, so they must not be detached
    copies: a backward from x_latent alone has to reach patch_embed.
    """
    model = _teacher(tiny_kwargs).train()
    _, maps = model(packed_bayer(1, 16, 16, seed=53), return_maps=True)
    assert maps["x_latent"].requires_grad, "x_latent is detached from the graph"
    maps["x_latent"].pow(2).mean().backward()
    grad = model.patch_embed.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0, \
        "backward through x_latent did not reach patch_embed"


def test_second_positional_argument_is_return_maps_not_an_snr_map(tiny_kwargs):
    """
    The wrappers' signature is model(x, snr_map); the bare teacher's is
    model(x, return_maps).  Passing an SNR map positionally is a plausible
    mistake, and this pins that it fails loudly (torch refuses to take the
    truth value of a multi-element tensor) instead of quietly returning the
    (output, maps) tuple that the caller would then treat as an image.
    """
    model = _teacher(tiny_kwargs).eval()
    x = packed_bayer(1, 16, 16, seed=55)
    snr = snr_map_for(x)
    with pytest.raises(RuntimeError, match="Boolean value of Tensor"):
        with torch.no_grad():
            model(x, snr)


# ─────────────────────────────────────────────────────────────────────────────
# 6. se_reduction: legacy checkpoint compatibility
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("se_reduction,expected", [
    (None, nn.Identity),
    (0, nn.Identity),      # falsy -> legacy path as well
    (8, SqueezeExcite),
    (2, SqueezeExcite),
])
def test_se_reduction_selects_the_residual_block_variant(tiny_kwargs, se_reduction,
                                                        expected):
    """
    se_reduction is threaded into every ResidualConvBlock in the model.  The
    structural check guarantees the legacy configuration gets a true no-op
    module (not a differently-initialised SE), which is what keeps pre-SE
    checkpoints loadable at all.
    """
    model = _teacher(tiny_kwargs, se_reduction=se_reduction)
    stacks = (model.encoder_level_1, model.encoder_level_2, model.decoder_level_2,
              model.decoder_level_1, model.decoder_level_0)
    for stack in stacks:
        for block in stack:
            assert isinstance(block.se, expected), \
                f"se_reduction={se_reduction} produced {type(block.se).__name__}"


def test_legacy_state_dict_keys_are_a_subset_of_the_se_keys(tiny_kwargs):
    """
    Old checkpoints were saved from the se_reduction=None architecture, so every
    legacy key must still exist in the SE model and the SE config may only ADD
    keys — all of them under `.se.gate.`.  That is precisely the condition under
    which a legacy checkpoint can be loaded with strict=False and the only
    complaints are the freshly initialised SE gates.
    """
    legacy = _teacher(tiny_kwargs, se_reduction=None)
    se = _teacher(tiny_kwargs, se_reduction=8)
    legacy_keys, se_keys = set(legacy.state_dict()), set(se.state_dict())

    assert legacy_keys <= se_keys, \
        f"legacy-only keys are missing from the SE model: {sorted(legacy_keys - se_keys)}"
    extra = se_keys - legacy_keys
    assert extra, "se_reduction=8 added no parameters at all"
    offenders = sorted(k for k in extra if ".se.gate." not in k)
    assert not offenders, f"SE config added non-SE keys: {offenders}"
    assert _n_params(se) > _n_params(legacy)


def test_legacy_state_dict_into_an_se_model_fails_strict_but_loads_non_strict(tiny_kwargs):
    """
    The two configurations must NOT be silently interchangeable: strict=True has
    to reject a legacy checkpoint so a half-initialised SE model can never be
    mistaken for a restored one, while strict=False reports exactly the missing
    SE gates and nothing unexpected.
    """
    legacy_sd = _teacher(tiny_kwargs, se_reduction=None).state_dict()
    se = _teacher(tiny_kwargs, se_reduction=8)

    with pytest.raises(RuntimeError, match="Missing key"):
        se.load_state_dict(legacy_sd, strict=True)

    report = se.load_state_dict(legacy_sd, strict=False)
    assert report.unexpected_keys == []
    assert report.missing_keys, "expected the SE gates to be reported missing"
    assert all(".se.gate." in k for k in report.missing_keys)


def test_se_state_dict_into_a_legacy_model_fails_strict_with_unexpected_keys(tiny_kwargs):
    """
    The other direction: an SE checkpoint must not load strictly into a legacy
    model, because the SE gates would be dropped and inference would silently
    differ from training.  strict=True has to name them as unexpected.
    """
    se_sd = _teacher(tiny_kwargs, se_reduction=8).state_dict()
    legacy = _teacher(tiny_kwargs, se_reduction=None)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        legacy.load_state_dict(se_sd, strict=True)

    report = legacy.load_state_dict(se_sd, strict=False)
    assert report.missing_keys == []
    assert all(".se.gate." in k for k in report.unexpected_keys)


def test_legacy_to_legacy_round_trip_is_exact(tiny_kwargs):
    """
    The other half of the compatibility promise: a legacy state_dict loaded
    strictly into another legacy model reproduces its outputs bit for bit.  The
    final assertion checks the two random inits really did differ, so a no-op
    load could not have faked the result.
    """
    src = _teacher(tiny_kwargs, se_reduction=None).eval()
    dst = _teacher(tiny_kwargs, se_reduction=None).eval()
    x = packed_bayer(1, 16, 16, seed=57)
    with torch.no_grad():
        before = dst(x)
    dst.load_state_dict(src.state_dict(), strict=True)
    with torch.no_grad():
        after = dst(x)
        assert torch.equal(src(x), after), "legacy round trip changed the output"
    assert not torch.equal(before, after), \
        "the two random inits were identical — this test proved nothing"


def test_state_dict_file_round_trip_reproduces_identical_outputs(tiny_kwargs, tmp_path):
    """
    The resume-from-checkpoint guarantee, through a real file: save the SE
    model's state_dict, load it strictly into a fresh instance and require
    bit-identical outputs (checked on a non-square batch, since that is what
    the padded training batches look like).
    """
    src = _teacher(tiny_kwargs, se_reduction=8).eval()
    path = tmp_path / "teacher_sd.pt"
    torch.save(src.state_dict(), str(path))

    dst = _teacher(tiny_kwargs, se_reduction=8).eval()
    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    report = dst.load_state_dict(loaded, strict=True)
    assert report.missing_keys == [] and report.unexpected_keys == []

    x = packed_bayer(2, 16, 24, seed=59)
    with torch.no_grad():
        assert torch.equal(src(x), dst(x)), "state_dict round trip changed the output"


@pytest.mark.parametrize("se_reduction", [None, 8])
def test_both_se_configurations_honour_the_output_contract(tiny_kwargs, se_reduction):
    """se_reduction may change the parameters but never the tensor contract."""
    model = _teacher(tiny_kwargs, se_reduction=se_reduction).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 16, 16, seed=61))
    assert_shape(out, (1, 3, 32, 32), f"output(se_reduction={se_reduction})")
    assert_finite(out, "output")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Constructor arguments
# ─────────────────────────────────────────────────────────────────────────────

def test_num_blocks_and_refinement_blocks_control_the_stack_depths(tiny_kwargs):
    """
    num_blocks indexes the stacks as [enc1 and dec1, enc2, dec2, latent] and
    num_refinement_blocks sizes decoder_level_0.  Checkpoints store both in
    model_kwargs, so a re-indexing here would make every saved weight file
    unloadable while still constructing happily.
    """
    model = _teacher(tiny_kwargs, num_blocks=[2, 3, 4, 5], num_refinement_blocks=3)
    assert len(model.encoder_level_1) == 2
    assert len(model.decoder_level_1) == 2      # shares num_blocks[0]
    assert len(model.encoder_level_2) == 3
    assert len(model.decoder_level_2) == 4
    assert len(model.latent) == 5
    assert len(model.decoder_level_0) == 3
    assert model.dim == tiny_kwargs["dim"]


def test_only_the_last_heads_entry_reaches_the_model(tiny_kwargs):
    """
    The teacher has exactly one transformer stage (the latent), so heads[3] is
    the only entry consumed and heads[0:3] are inert.  Worth pinning both ways:
    the latent's head count really does follow heads[3], and a mis-set heads[0]
    will NOT be caught by any shape error.
    """
    a = _teacher(tiny_kwargs, heads=[1, 1, 1, 4])
    b = _teacher(tiny_kwargs, heads=[8, 2, 1, 4])
    assert a.latent[0].attn.num_heads == 4
    assert b.latent[0].attn.num_heads == 4
    assert set(a.state_dict()) == set(b.state_dict())
    assert _n_params(a) == _n_params(b)


def test_a_deeper_configuration_still_honours_the_shape_contract(tiny_kwargs):
    """A non-uniform num_blocks config must not disturb the resolution ladder —
    the depths and the resolutions are independent knobs."""
    model = _teacher(tiny_kwargs, num_blocks=[2, 1, 2, 2],
                     num_refinement_blocks=2).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 16, 16, seed=63))
    assert_shape(out, (1, 3, 32, 32), "output")
    assert_finite(out, "output")


def test_odd_dim_constructs_but_cannot_run(tiny_kwargs):
    """
    up_shuffle_1_0 turns dim*2 channels into dim//2, which only works for an
    even dim.  An odd dim builds without complaint and then dies in the first
    forward — documented here so nobody reads the silent construction as
    support for odd widths.  Fails loudly, so not a correctness hazard.
    """
    model = _teacher(tiny_kwargs, dim=9).eval()
    with pytest.raises(RuntimeError, match="pixel_shuffle"):
        with torch.no_grad():
            model(packed_bayer(1, 8, 8, seed=65))


@pytest.mark.parametrize("dim", [4, 6, 10])
def test_even_dims_other_than_the_default_work(tiny_kwargs, dim):
    """
    Any even dim must produce the same contract; dim is swept by the ablation
    configs in the submit scripts, so this is not a hypothetical.
    """
    model = _teacher(tiny_kwargs, dim=dim).eval()
    with torch.no_grad():
        out = model(packed_bayer(1, 8, 8, seed=67))
    assert_shape(out, (1, 3, 16, 16), f"output(dim={dim})")
    assert_finite(out, "output")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Parameter count of the documented default configuration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.xfail(
    reason="BUG: the class docstring pairs dim=32/num_blocks=(4,4,4,6)/"
           "heads=(1,2,4,8) with ~19.5M parameters, but that config builds "
           "26.92M (+38%); ~19.5M matches num_blocks=(4,4,4,4)",
    strict=False,
)
def test_documented_default_config_has_about_19_5m_parameters():
    """
    The class docstring advertises "dim=32, num_blocks=[4,4,4,6],
    heads=[1,2,4,8] -> ~19.5M parameters".  That figure is what the memory and
    FLOPs budget in the submit scripts is sized against, so it should hold to
    within a generous +/-15%.  Construction only — no forward pass at this size.
    """
    model = TransUNet_Teacher_HDR(dim=32, num_blocks=(4, 4, 4, 6), heads=(1, 2, 4, 8))
    n_params = _n_params(model)
    target = 19.5e6
    assert 0.85 * target <= n_params <= 1.15 * target, \
        f"default config has {n_params / 1e6:.2f}M params, docstring says ~19.5M"


@pytest.mark.slow
def test_the_config_the_training_scripts_actually_use_is_about_19_5m():
    """
    Diagnosis for the xfail above: train_A100_MoE_two_phase.py and
    test_dual_MoE_two_phase.py both pass num_blocks=[4,4,4,4], and THAT config
    lands inside the documented +/-15% band.  The docstring's parameter count and
    its num_blocks therefore describe two different models, and the class
    default is the heavier one.
    """
    model = TransUNet_Teacher_HDR(dim=32, num_blocks=(4, 4, 4, 4), heads=(1, 2, 4, 8))
    n_params = _n_params(model)
    target = 19.5e6
    assert 0.85 * target <= n_params <= 1.15 * target, \
        f"num_blocks=(4,4,4,4) gives {n_params / 1e6:.2f}M params"


@pytest.mark.slow
@pytest.mark.parametrize("num_blocks,expected", [
    ((4, 4, 4, 6), 26_919_176),   # the class default
    ((4, 4, 4, 4), 20_560_276),   # what the train/eval scripts pass
])
def test_parameter_counts_of_the_shipped_configs_are_locked(num_blocks, expected):
    """
    Exact counts for the two configurations that exist in the repo.  An
    unintended architecture change would also invalidate every stored
    checkpoint, so it is worth catching here even while the docstring's ~19.5M
    figure stays wrong.
    """
    model = TransUNet_Teacher_HDR(dim=32, num_blocks=num_blocks, heads=(1, 2, 4, 8))
    assert _n_params(model) == expected, \
        f"num_blocks={num_blocks}: {_n_params(model)} params, expected {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Gradients
# ─────────────────────────────────────────────────────────────────────────────

def test_backward_reaches_patch_embed_the_latent_and_to_rgb(tiny_kwargs):
    """
    One backward from the output must leave finite, non-zero gradients at the
    three ends of the network: patch_embed (the first conv), the Restormer
    latent (deepest point, behind two PixelUnshuffle stages and a skip concat)
    and to_rgb (the final projection).  A detached skip or a mis-wired concat
    typically shows up as a *zero* gradient in the latent while the shallow
    layers keep training, which is invisible outside the loss curve.
    """
    model = _teacher(tiny_kwargs).train()
    out = model(packed_bayer(1, 16, 16, seed=69))
    out.pow(2).mean().backward()

    probes = {
        "patch_embed.weight":          model.patch_embed.weight,
        "latent.attn.qkv.weight":      model.latent[0].attn.qkv.weight,
        "latent.attn.temperature":     model.latent[0].attn.temperature,
        "latent.ffn.project_out":      model.latent[0].ffn.project_out.weight,
        "latent_fusion.weight":        model.latent_fusion.weight,
        "to_rgb.weight":               model.to_rgb.weight,
    }
    for name, param in probes.items():
        assert param.grad is not None, f"{name} received no gradient"
        assert_finite(param.grad, f"{name}.grad")
        assert float(param.grad.abs().sum()) > 0.0, f"{name} gradient is all zeros"


def test_no_parameter_is_dead_weight(tiny_kwargs):
    """
    Stronger version of the probe above: every parameter of the tiny teacher
    gets gradient signal.  An unreachable sub-module would be silently
    untrained and would still occupy bytes in every checkpoint.
    """
    model = _teacher(tiny_kwargs).train()
    model(packed_bayer(1, 16, 16, seed=71)).pow(2).mean().backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert dead == [], f"parameters with no gradient signal: {dead}"


def test_gradient_flows_back_to_the_input(tiny_kwargs):
    """
    The teacher must be differentiable w.r.t. its input, which is what makes it
    usable as a stage after add_photon_noise / the differentiable GBTF front end
    in an end-to-end pipeline.
    """
    model = _teacher(tiny_kwargs).train()
    x = packed_bayer(1, 16, 16, seed=73).requires_grad_(True)
    model(x).pow(2).mean().backward()
    assert x.grad is not None
    assert_shape(x.grad, x.shape, "input grad")
    assert_finite(x.grad, "input grad")
    assert float(x.grad.abs().sum()) > 0.0


def test_the_lower_clamp_blocks_gradient_for_floored_pixels(tiny_kwargs):
    """
    clamp(min=...) is not a pass-through: pixels pushed below _CLAMP_EPS get
    exactly zero gradient.  Driving the whole raw output to -5 must therefore
    leave to_rgb with a zero gradient — the documented cost of the floor, and
    the reason a badly initialised head can stall training instead of
    recovering.
    """
    model = _drive_to_rgb(_teacher(tiny_kwargs), -5.0).train()
    out = model(packed_bayer(1, 16, 16, seed=75))
    out.sum().backward()
    assert float(out.min()) == pytest.approx(_CLAMP_EPS, rel=1e-9)
    assert float(model.to_rgb.bias.grad.abs().sum()) == 0.0, \
        "gradient leaked through the lower clamp"


def test_eval_upper_clamp_also_blocks_gradient(tiny_kwargs):
    """
    Mirror image at the top end: in eval() a saturated pixel gets no gradient
    either.  Anything that computes a loss without switching back to train()
    would therefore see a zero gradient for every over-bright pixel — worth
    pinning so the mode switch in the training loop is understood as
    load-bearing, not cosmetic.
    """
    model = _drive_to_rgb(_teacher(tiny_kwargs), 5.0).eval()
    out = model(packed_bayer(1, 16, 16, seed=77))
    out.sum().backward()
    assert float(out.max()) == pytest.approx(1.0, rel=1e-6)
    assert float(model.to_rgb.bias.grad.abs().sum()) == 0.0, \
        "gradient leaked through the eval-mode upper clamp"
