"""
Exhaustive tests for DifferentiableGBTF_BGGR (DifferentiableGBTF_BGGR.py).

This module is the fixed, non-learned demosaicer used in two load-bearing
places:

  * train_A100_MoE_two_phase.py:535  ``y_rgb = gbtf(F.pixel_shuffle(y, 2))``
    turns the clean packed Bayer ground truth into the RGB target that every
    loss (L1 + LPIPS) is measured against.  A demosaicing error here is baked
    silently into the training signal.
  * test_dual_MoE_two_phase.py:441-442 demosaics both GT and the noisy input
    to produce every reported PSNR/SSIM number.

So the invariants worth pinning are: the [B,1,H,W] -> [B,3,H,W] contract, the
BGGR->RGB channel mapping, bit-exact pass-through of the *measured* samples,
exactness on inputs a demosaicer cannot get wrong (flat fields, achromatic
1-D ramps), locality of the filter, and end-to-end differentiability -- the
class name promises the last one and the forward pass is full of in-place
index assignments into ``Dst`` that could easily break autograd.

Bayer geometry used throughout (BGGR, anchored at pixel (0,0)):

    (0,0) B      (0,1) G1        -> RGB channel 2, 1
    (1,0) G2     (1,1) R         -> RGB channel 1, 0
"""

import math

import pytest
import torch
import torch.nn.functional as F

from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR
from helpers import (
    assert_finite,
    assert_in_range,
    assert_shape,
    constant_mosaic,
    mosaic_from_packed,
    packed_bayer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — conftest/helpers are shared)
# ─────────────────────────────────────────────────────────────────────────────

BUFFER_NAMES = ("HK", "VK", "box5x5", "Prb", "k_cross")


@pytest.fixture
def gbtf():
    """A fresh demosaicer in eval mode, the way both call sites use it."""
    m = DifferentiableGBTF_BGGR()
    m.eval()
    return m


def flat_colour_mosaic(H, W, b, g, r, batch=1, dtype=torch.float32):
    """
    Sensor mosaic of a *spatially flat but chromatic* scene: every B site holds
    `b`, every G site `g`, every R site `r`.  Works for odd H/W too (strided
    assignment just truncates), which `helpers.mosaic_from_packed` cannot do.
    """
    mo = torch.zeros((batch, 1, H, W), dtype=dtype)
    mo[:, 0, 0::2, 0::2] = b     # B
    mo[:, 0, 0::2, 1::2] = g     # G1
    mo[:, 0, 1::2, 0::2] = g     # G2
    mo[:, 0, 1::2, 1::2] = r     # R
    return mo


def mosaic_from_rgb(rgb):
    """
    Sample a full-colour [B,3,H,W] image at the BGGR sites -> [B,1,H,W] mosaic.
    The inverse problem the demosaicer is meant to solve, so `rgb` doubles as
    ground truth.
    """
    B, _, H, W = rgb.shape
    mo = torch.zeros((B, 1, H, W), dtype=rgb.dtype)
    mo[:, 0, 0::2, 0::2] = rgb[:, 2, 0::2, 0::2]   # B
    mo[:, 0, 0::2, 1::2] = rgb[:, 1, 0::2, 1::2]   # G1
    mo[:, 0, 1::2, 0::2] = rgb[:, 1, 1::2, 0::2]   # G2
    mo[:, 0, 1::2, 1::2] = rgb[:, 0, 1::2, 1::2]   # R
    return mo


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def grid(H, W):
    y = torch.arange(H, dtype=torch.float32).view(-1, 1)
    x = torch.arange(W, dtype=torch.float32).view(1, -1)
    return y, x


# ─────────────────────────────────────────────────────────────────────────────
# 1. Output contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("batch,h,w", [(1, 8, 8), (2, 4, 6), (3, 16, 12), (1, 3, 3)])
def test_output_shape_dtype_finite_and_in_unit_range(gbtf, batch, h, w):
    """
    The published contract: [B,1,H,W] mosaic -> [B,3,H,W] RGB, float32,
    all-finite, inside [0,1].  Every downstream metric and the LPIPS VGG loss
    assume exactly this, so it is checked across batch sizes and both square
    and non-square sensor sizes.  h/w here are PACKED dims; the mosaic is 2h x 2w.
    """
    mosaic = mosaic_from_packed(packed_bayer(batch, h, w, seed=h + w))
    assert_shape(mosaic, (batch, 1, 2 * h, 2 * w), "mosaic")

    out = gbtf(mosaic)

    assert_shape(out, (batch, 3, 2 * h, 2 * w), "gbtf output")
    assert out.dtype is torch.float32
    assert_finite(out, "gbtf output")
    assert_in_range(out, 0.0, 1.0, "gbtf output", atol=0.0)


def test_reflect_padding_imposes_a_minimum_spatial_size(gbtf):
    """
    Step 4 reflect-pads the colour-difference map by 4, and reflect padding
    requires pad < dim, so the smallest mosaic the module accepts is 5x5.
    That means a packed 2x2 patch (4x4 sensor) *crashes* rather than returning
    a degraded result — worth knowing before someone picks a tiny PATCH_SIZE.
    """
    with pytest.raises(RuntimeError, match="Padding size"):
        gbtf(torch.rand(1, 1, 4, 4))

    for size in (5, 6):
        out = gbtf(torch.rand(1, 1, size, size))
        assert_shape(out, (1, 3, size, size), f"{size}x{size} output")
        assert_finite(out, "small output")


def test_batch_elements_are_independent(gbtf):
    """
    Nothing in the forward pass may mix batch elements: the training loop
    demosaics whole batches while the eval loop runs batch_size=1, and the two
    must agree bit-for-bit or the reported PSNRs describe a different image
    than the one that was trained on.
    """
    mosaic = mosaic_from_packed(packed_bayer(3, 8, 8, seed=11))
    batched = gbtf(mosaic)
    for i in range(3):
        single = gbtf(mosaic[i:i + 1])
        assert torch.equal(batched[i:i + 1], single), f"batch element {i} differs"


def test_forward_is_deterministic_and_mode_invariant(gbtf):
    """
    The demosaicer is a fixed filter: repeated calls must be bit-identical and
    train()/eval() must make no difference (unlike the denoisers, which clamp
    to 1.0 only in eval).  Both call sites call .eval() and then rely on the
    result as ground truth, so any mode-dependence would be a silent trap.
    """
    mosaic = mosaic_from_packed(packed_bayer(1, 8, 8, seed=5))
    first = gbtf(mosaic)
    assert torch.equal(first, gbtf(mosaic))

    gbtf.train()
    train_out = gbtf(mosaic)
    gbtf.eval()
    assert torch.equal(train_out, gbtf(mosaic))


def test_non_contiguous_input_gives_the_same_result(gbtf):
    """
    A cropped view of a larger sensor image (what patch-based inference hands
    over) is non-contiguous.  Strided indexing plus F.pad must handle that
    identically to a contiguous copy, otherwise patch inference and full-frame
    inference silently disagree.
    """
    big = torch.rand(1, 1, 32, 32)
    view = big[:, :, 4:20, 6:22]
    assert not view.is_contiguous()
    assert torch.equal(gbtf(view), gbtf(view.contiguous()))


def test_buffers_are_not_mutated_by_forward(gbtf):
    """
    forward() must be side-effect free on the module state; the kernels are
    constants.  A stray in-place op on a buffer would corrupt every subsequent
    call in the training loop.
    """
    before = {k: v.clone() for k, v in gbtf.state_dict().items()}
    gbtf(mosaic_from_packed(packed_bayer(2, 8, 8, seed=2)))
    for k, v in gbtf.state_dict().items():
        assert torch.equal(before[k], v), f"buffer {k} was mutated by forward"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Native-site fidelity — the measured samples must survive untouched
# ─────────────────────────────────────────────────────────────────────────────

def test_native_green_sites_are_passed_through_bit_exactly(gbtf):
    """
    At a green photosite the green value was *measured*; interpolating it would
    throw away real data.  Step 4's ``torch.where(mask_missing_G, G_new, mosaic)``
    is supposed to keep the measurement, and this pins both green sub-lattices
    (G1 at even-row/odd-col, G2 at odd-row/even-col) to bit-exact equality.
    """
    mosaic = mosaic_from_packed(packed_bayer(2, 8, 8, seed=7))
    out = gbtf(mosaic)

    assert torch.equal(out[:, 1, 0::2, 1::2], mosaic[:, 0, 0::2, 1::2]), "G1 site altered"
    assert torch.equal(out[:, 1, 1::2, 0::2], mosaic[:, 0, 1::2, 0::2]), "G2 site altered"


def test_native_red_and_blue_sites_are_passed_through_bit_exactly(gbtf):
    """
    Same argument for the chroma lattices: blue is measured at (0,0) and must
    land untouched in RGB channel 2, red is measured at (1,1) and must land in
    RGB channel 0.  This is also the strongest available check that the later
    ``torch.where`` mask routing (steps 5 and 6) does not overwrite the
    measurements it is only supposed to fill in *around*.
    """
    mosaic = mosaic_from_packed(packed_bayer(2, 8, 8, seed=7))
    out = gbtf(mosaic)

    assert torch.equal(out[:, 2, 0::2, 0::2], mosaic[:, 0, 0::2, 0::2]), "native B altered"
    assert torch.equal(out[:, 0, 1::2, 1::2], mosaic[:, 0, 1::2, 1::2]), "native R altered"


def test_channel_mapping_is_rgb_not_bgr(gbtf):
    """
    The module claims to emit "true RGB" from a BGGR sensor, i.e. the *blue*
    photosite value must come out of channel 2 and the *red* one out of
    channel 0.  Distinct per-colour levels make an accidental identity/BGR
    mapping impossible to miss — and it would corrupt every LPIPS number,
    since VGG expects RGB.
    """
    b_lvl, g_lvl, r_lvl = 0.9, 0.5, 0.1
    out = gbtf(flat_colour_mosaic(16, 16, b_lvl, g_lvl, r_lvl))

    assert out[:, 0].mean() == pytest.approx(r_lvl, abs=1e-3), "channel 0 is not red"
    assert out[:, 1].mean() == pytest.approx(g_lvl, abs=1e-3), "channel 1 is not green"
    assert out[:, 2].mean() == pytest.approx(b_lvl, abs=1e-3), "channel 2 is not blue"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Flat fields — the cases a demosaicer must get exactly right
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 1.0])
def test_uniform_grey_mosaic_stays_exactly_uniform(gbtf, value):
    """
    A flat grey field has zero colour difference, so every interpolated value
    must reduce to the measured constant — including at the image border,
    where the reflect/zero padding could easily bleed something else in.
    Any deviation would show up as a coloured frame around every GT image.
    """
    mosaic = mosaic_from_packed(constant_mosaic(1, 16, 16, value))
    out = gbtf(mosaic)

    assert_shape(out, (1, 3, 32, 32), "output")
    assert float((out - value).abs().max()) == 0.0, \
        f"uniform {value} mosaic did not stay uniform (max dev "\
        f"{float((out - value).abs().max())})"


@pytest.mark.parametrize("H,W", [(16, 16), (6, 6), (5, 5), (12, 20)])
def test_flat_chromatic_field_reconstructs_every_channel(gbtf, H, W):
    """
    A spatially flat but *chromatic* field (B=0.2, G=0.5, R=0.8) is the
    textbook case: the true colour difference is constant, so GBTF should
    recover (R,G,B) = (0.8,0.5,0.2) at every pixel, borders included.  The
    residual 1.2e-4 is the Prb kernel's DC gain shortfall (see
    test_prb_kernel_has_unit_dc_gain), so the tolerance is set just above it.
    """
    out = gbtf(flat_colour_mosaic(H, W, 0.2, 0.5, 0.8))
    target = torch.tensor([0.8, 0.5, 0.2]).view(1, 3, 1, 1)

    dev = float((out - target).abs().max())
    assert dev < 2e-4, f"flat chromatic field off by {dev} at {H}x{W}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reconstruction quality on solvable scenes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("axis", ["x", "y"])
def test_achromatic_one_dimensional_scene_is_reconstructed_exactly(gbtf, axis):
    """
    When a scene is achromatic (R=G=B) and varies along a single axis, GBTF's
    directional weighting has a zero-gradient direction available and should
    reconstruct the image *exactly* — the strongest correctness statement one
    can make about this filter without reimplementing it.  It simultaneously
    proves the interpolated R/B channels stay achromatic (no false colour) and
    that the four direction weights are paired with the matching directional
    sums (a swapped pairing would smear the varying axis).
    """
    H = W = 32
    y, x = grid(H, W)
    coord = x if axis == "x" else y
    lum = (0.5 + 0.35 * torch.sin(2 * math.pi * coord / 9.0)).expand(H, W).clamp(0, 1)
    rgb = lum.view(1, 1, H, W).expand(1, 3, H, W).contiguous()

    out = gbtf(mosaic_from_rgb(rgb))

    assert float((out - rgb).abs().max()) < 1e-6, \
        f"achromatic 1-D ({axis}) scene not reproduced: max err "\
        f"{float((out - rgb).abs().max())}"
    spread = float((out.max(dim=1).values - out.min(dim=1).values).max())
    assert spread < 1e-6, f"false colour on achromatic input (spread {spread})"


def test_smooth_achromatic_two_dimensional_scene_is_near_exact(gbtf):
    """
    A band-limited achromatic scene varying in *both* axes is still an easy
    target: PSNR must stay very high and the residual chroma tiny.  This is
    the regression guard for the tentative-green step (step 4) — a mis-scaled
    AccumH/AccumV or a wrong 0.2 normalisation would drop this by tens of dB.
    """
    H = W = 32
    y, x = grid(H, W)
    lum = (0.5 + 0.2 * torch.sin(2 * math.pi * x / 29.0)
               + 0.2 * torch.cos(2 * math.pi * y / 23.0)).clamp(0, 1)
    rgb = lum.view(1, 1, H, W).expand(1, 3, H, W).contiguous()

    out = gbtf(mosaic_from_rgb(rgb))

    assert psnr(out, rgb) > 45.0, f"smooth achromatic PSNR only {psnr(out, rgb):.2f} dB"
    spread = float((out.max(dim=1).values - out.min(dim=1).values).max())
    assert spread < 0.05, f"excessive false colour (spread {spread})"


def test_smooth_chromatic_scene_reconstructs_well_in_the_interior(gbtf):
    """
    A smooth but strongly chromatic scene (each channel a different low
    frequency) breaks GBTF's constant-colour-difference assumption, so it is
    the honest lower bound on quality.  Pinning interior PSNR > 30 dB catches
    a broken chroma stage (steps 5/6) without being sensitive to the border
    handling, which is checked separately.
    """
    H = W = 32
    y, x = grid(H, W)
    rgb = torch.stack([
        (0.5 + 0.30 * torch.sin(2 * math.pi * x / 40.0)).expand(H, W),
        (0.5 + 0.30 * torch.sin(2 * math.pi * y / 48.0 + 1.0)).expand(H, W),
        (0.5 + 0.25 * torch.cos(2 * math.pi * (x + y) / 56.0)).expand(H, W),
    ], dim=0).unsqueeze(0).clamp(0, 1)

    out = gbtf(mosaic_from_rgb(rgb))
    interior = psnr(out[:, :, 4:-4, 4:-4], rgb[:, :, 4:-4, 4:-4])
    assert interior > 30.0, f"smooth chromatic interior PSNR only {interior:.2f} dB"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Locality — a demosaicer is a small local filter
# ─────────────────────────────────────────────────────────────────────────────
#
# Legitimate support of one output pixel: HCD spans +/-2, the +/-1 gradient
# difference and the 5x5 box sum push the direction weights out to +/-5, and
# step 4 sums HCD over +/-4 -> at most 8 mosaic pixels in each direction.

@pytest.mark.parametrize("axis", [2, 3])
def test_output_depends_only_on_a_local_neighbourhood(gbtf, axis):
    """
    Perturbing the last four rows/columns of a 32x32 mosaic must not change the
    first four: 28 pixels is far outside any GBTF filter support.  It does,
    because step 3 shifts V_sum/H_sum with ``torch.roll`` (circular) instead of
    an edge-clamped shift, so ``E_val[..., 0] == H_sum[..., -2]``.  Real
    consequences: the border of every demosaiced GT frame is interpolated with
    weights derived from the *opposite* border, and patch-wise inference
    (test_dual_MoE_two_phase.infer_patches) cannot agree with full-frame
    inference at tile seams.
    """
    H = W = 32
    torch.manual_seed(0)
    mosaic = torch.rand(1, 1, H, W) * 0.4 + 0.3
    base = gbtf(mosaic)

    far = mosaic.clone()
    if axis == 2:
        far[:, :, H - 4:, :] = torch.rand(1, 1, 4, W)
        delta = (gbtf(far) - base)[:, :, :4, :].abs().max()
    else:
        far[:, :, :, W - 4:] = torch.rand(1, 1, H, 4)
        delta = (gbtf(far) - base)[:, :, :, :4].abs().max()

    assert float(delta) < 1e-6, \
        f"far-edge content changed the opposite border by {float(delta)}"


def test_gradient_support_of_a_corner_pixel_is_local(gbtf):
    """
    The autograd view of the same defect, and the sharper statement: the
    gradient of the top-left output pixel w.r.t. the mosaic must vanish more
    than 8 pixels away.  It is instead non-zero on rows/columns 25..31 — the
    wrapped ``torch.roll`` weights literally give a corner pixel a dependency
    on the far corner of the sensor.
    """
    H = W = 32
    mosaic = (torch.rand(1, 1, H, W) * 0.4 + 0.3).requires_grad_(True)
    gbtf(mosaic)[0, 1, 0, 0].backward()
    g = mosaic.grad[0, 0].abs()

    far_cols = [c for c in range(12, W) if float(g[:, c].max()) > 0]
    far_rows = [r for r in range(12, H) if float(g[r, :].max()) > 0]
    assert not far_cols and not far_rows, \
        f"corner output depends on distant rows {far_rows} / cols {far_cols}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Differentiability — the whole point of the class name
# ─────────────────────────────────────────────────────────────────────────────

def test_backward_yields_a_finite_nonzero_gradient_everywhere(gbtf):
    """
    forward() writes into ``Dst`` with in-place indexed assignments
    (``Dst[:, 2, 0::2, 0::2] = ...``) and later reads ``Dst`` back inside
    ``torch.where`` before overwriting it again — exactly the pattern that
    raises "a variable needed for gradient computation has been modified by an
    inplace operation".  This test proves backward() runs and that *every*
    mosaic sample receives a finite, non-zero gradient, so the module can sit
    inside a differentiable pipeline.
    """
    mosaic = mosaic_from_packed(packed_bayer(1, 8, 8, seed=3)).clone().requires_grad_(True)
    out = gbtf(mosaic)
    assert out.requires_grad
    out.sum().backward()

    g = mosaic.grad
    assert g is not None, "no gradient reached the input"
    assert_shape(g, mosaic.shape, "input grad")
    assert_finite(g, "input grad")
    assert int((g == 0).sum()) == 0, \
        f"{int((g == 0).sum())} mosaic samples received a zero gradient"


def test_gradient_matches_the_numerical_jacobian(gbtf):
    """
    A real correctness check on the autograd path rather than a smoke test:
    torch.autograd.gradcheck compares the analytic Jacobian against central
    finite differences in float64.  Inputs are held in [0.4,0.6] so no
    ``clamp(0,1)`` saturates and the function is locally smooth.
    """
    m = DifferentiableGBTF_BGGR().double().eval()
    mosaic = (0.4 + 0.2 * torch.rand(1, 1, 6, 6, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(m, (mosaic,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_no_grad_context_produces_a_detached_output(gbtf):
    """
    Both call sites run the demosaicer under torch.no_grad()/inference; the
    result must not carry a graph, or the training loop would backprop into a
    frozen filter and waste memory on a 4k-pixel-wide graph.
    """
    mosaic = mosaic_from_packed(packed_bayer(1, 8, 8, seed=4)).requires_grad_(True)
    with torch.no_grad():
        out = gbtf(mosaic)
    assert not out.requires_grad
    assert out.grad_fn is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Buffers, parameters, dtype and device
# ─────────────────────────────────────────────────────────────────────────────

def test_module_has_zero_trainable_parameters(gbtf):
    """
    GBTF is a *fixed* filter.  Both call sites do
    ``for p in gbtf.parameters(): p.requires_grad_(False)`` which is a silent
    no-op precisely because there are no parameters; if a kernel ever became an
    nn.Parameter it would start receiving gradients and drift during training
    without anyone noticing.
    """
    assert list(gbtf.parameters()) == []
    assert list(gbtf.named_parameters()) == []
    assert sum(p.numel() for p in gbtf.parameters()) == 0


def test_kernels_are_registered_persistent_buffers(gbtf):
    """
    HK/VK/box5x5/Prb/k_cross must be *registered* buffers, not plain
    attributes, or ``.to(device)`` would leave them on the CPU and the whole
    forward pass would fail mid-training on the first GPU batch.  Registration
    also puts them in state_dict, which is what makes them travel with a
    checkpoint.
    """
    names = dict(gbtf.named_buffers())
    assert set(names) == set(BUFFER_NAMES)
    assert set(gbtf.state_dict()) == set(BUFFER_NAMES)

    expected_shapes = {
        "HK": (1, 1, 1, 5), "VK": (1, 1, 5, 1), "box5x5": (1, 1, 5, 5),
        "Prb": (1, 1, 7, 7), "k_cross": (1, 1, 3, 3),
    }
    for name, shape in expected_shapes.items():
        assert_shape(names[name], shape, name)
        assert names[name].dtype is torch.float32


def test_kernel_constants_are_the_gbtf_ones(gbtf):
    """
    Freeze the numeric kernels so an "optimisation" cannot quietly change the
    filter that generated every historical PSNR number.  The invariants that
    matter: HK/VK are the same Hamilton-Adams 1-D kernel (unit DC gain, so a
    flat field is preserved) transposed into the two axes, box5x5 is a plain
    unnormalised 5x5 sum, and k_cross is a 4-neighbour average with a ZERO
    centre — the zero centre is what lets step 6 read only valid neighbours
    while the green sites of Dst's chroma channels are still zero.
    """
    ha = torch.tensor([-0.25, 0.5, 0.5, 0.5, -0.25])
    assert torch.equal(gbtf.HK.view(-1), ha)
    assert torch.equal(gbtf.VK.view(-1), ha)
    assert float(gbtf.HK.sum()) == pytest.approx(1.0, abs=1e-7)

    assert torch.equal(gbtf.box5x5, torch.ones(1, 1, 5, 5))

    assert float(gbtf.k_cross[0, 0, 1, 1]) == 0.0, "k_cross centre must be zero"
    assert float(gbtf.k_cross.sum()) == pytest.approx(1.0, abs=1e-7)

    # Prb only touches sites of the same Bayer parity class as its centre,
    # which is why it can interpolate R at a B site (and vice versa).
    prb = gbtf.Prb[0, 0]
    for i in range(7):
        for j in range(7):
            if (i + j) % 2 == 1:
                assert float(prb[i, j]) == 0.0, f"Prb[{i},{j}] breaks parity"
    assert torch.equal(prb, prb.flip(-1))
    assert torch.equal(prb, prb.flip(-2))
    assert torch.equal(prb, prb.t())


def test_prb_kernel_has_unit_dc_gain(gbtf):
    """
    Every other Prb tap is an exact 32nd (0.3125 = 10/32) and the reference
    GBTF kernel is [[-1,10,10,-1]-style]/32, whose taps sum to exactly 1.  A
    unit DC gain is what makes ``RB = G - Prb*(G-C)`` reproduce a flat colour
    exactly.  The 4-decimal rounding of 1/32 leaves the gain at 0.9996, which
    is why test_flat_chromatic_field_reconstructs_every_channel needs a 2e-4
    tolerance instead of hitting the constant exactly.
    """
    assert float(gbtf.Prb.sum()) == pytest.approx(1.0, abs=1e-7)


def test_dtype_conversions_move_the_buffers(gbtf):
    """
    ``.double()`` / ``.float()`` must reach the buffers (they only do because
    they are registered).  Mixed precision is the practical concern: the
    training loop calls ``gbtf(F.pixel_shuffle(y.float(), 2))`` and would break
    if the kernels had been left in another dtype.
    """
    gbtf.double()
    assert all(b.dtype is torch.float64 for b in gbtf.buffers())
    gbtf.float()
    assert all(b.dtype is torch.float32 for b in gbtf.buffers())
    assert all(b.device.type == "cpu" for b in gbtf.to(torch.device("cpu")).buffers())


def test_input_dtype_must_match_the_buffer_dtype(gbtf):
    """
    Documented dtype behaviour: the module does NOT cast, so float32 kernels
    plus a float64 mosaic raise instead of silently up/down-casting.  Once the
    module is ``.double()``d, float64 flows end to end and float32 is then the
    input that raises.  Callers must therefore keep the explicit ``.float()``
    they already have.
    """
    with pytest.raises(RuntimeError, match="Double"):
        gbtf(torch.rand(1, 1, 8, 8, dtype=torch.float64))

    m64 = DifferentiableGBTF_BGGR().double().eval()
    out = m64(torch.rand(1, 1, 8, 8, dtype=torch.float64))
    assert out.dtype is torch.float64
    assert_finite(out, "float64 output")
    assert_in_range(out, 0.0, 1.0, "float64 output", atol=0.0)

    with pytest.raises(RuntimeError, match="Float"):
        m64(torch.rand(1, 1, 8, 8, dtype=torch.float32))


def test_state_dict_roundtrip_preserves_behaviour(gbtf):
    """
    Buffers are persistent, so a checkpoint that ever contains this submodule
    must reload into an identical filter.  Cheap guard against someone flipping
    them to persistent=False and getting a subtly different demosaicer after a
    resume.
    """
    fresh = DifferentiableGBTF_BGGR()
    missing, unexpected = fresh.load_state_dict(gbtf.state_dict(), strict=True)
    assert missing == [] and unexpected == []
    mosaic = mosaic_from_packed(packed_bayer(1, 8, 8, seed=9))
    assert torch.equal(fresh.eval()(mosaic), gbtf(mosaic))


@pytest.mark.gpu
def test_gpu_matches_cpu(gbtf, gpu_device):
    """
    ``.to(device)`` is how both call sites use the module; the GPU result must
    match the CPU one to float32 tolerance, and the index-arange tensors built
    inside forward() must land on the input's device (they use `mosaic.device`,
    so a regression to a hardcoded CPU arange would raise here).

    Device-agnostic on purpose: this is exactly the class of bug that shows
    up first on a new backend, so it has to run on Aurora's XPU too.
    """
    mosaic = mosaic_from_packed(packed_bayer(2, 8, 8, seed=13))
    cpu_out = gbtf(mosaic)

    gpu_m = gbtf.to(gpu_device)
    assert all(b.device.type == gpu_device.type for b in gpu_m.buffers())
    gpu_out = gpu_m(mosaic.to(gpu_device))
    assert gpu_out.device.type == gpu_device.type
    assert torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Documented behaviour on inputs outside the contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("H,W", [(9, 8), (8, 9), (9, 9), (7, 7)])
def test_odd_dimensions_run_and_stay_anchored_at_the_origin(gbtf, H, W):
    """
    A Bayer mosaic needs even dimensions, but the module performs no shape
    validation.  Documented real behaviour: odd H/W run fine and the pattern
    stays anchored at pixel (0,0), so native samples are still preserved and a
    flat chromatic field is still exact — the last row/column simply holds an
    incomplete 2x2 cell.  Nothing raises and nothing becomes NaN, but nothing
    warns either, so a caller passing an odd crop gets a plausible-looking
    result with a half-cell edge.
    """
    mosaic = torch.rand(1, 1, H, W)
    out = gbtf(mosaic)

    assert_shape(out, (1, 3, H, W), "odd-size output")
    assert_finite(out, "odd-size output")
    assert_in_range(out, 0.0, 1.0, "odd-size output", atol=0.0)
    assert torch.equal(out[:, 1, 0::2, 1::2], mosaic[:, 0, 0::2, 1::2])
    assert torch.equal(out[:, 2, 0::2, 0::2], mosaic[:, 0, 0::2, 0::2])

    flat = gbtf(flat_colour_mosaic(H, W, 0.2, 0.5, 0.8))
    target = torch.tensor([0.8, 0.5, 0.2]).view(1, 3, 1, 1)
    assert float((flat - target).abs().max()) < 2e-4


def test_out_of_range_input_is_clamped_only_where_it_is_interpolated(gbtf):
    """
    Documented limitation of the [0,1] output guarantee: every *interpolated*
    value goes through clamp(0,1), but the measured samples are copied
    straight through (step 0 and the ``mosaic`` branch of step 4's
    torch.where), so a mosaic outside [0,1] produces an output outside [0,1] at
    native sites only.  Not a defect in practice — train_A100 feeds
    dataset-normalised Bayer and test_dual clamps to [0,1] before calling —
    but it means the module must not be treated as a range sanitiser.
    """
    mosaic = mosaic_from_packed(packed_bayer(1, 8, 8, seed=5, low=1.5, high=2.0))
    out = gbtf(mosaic)

    # Native samples survive verbatim, above 1.0 included.
    assert torch.equal(out[:, 1, 0::2, 1::2], mosaic[:, 0, 0::2, 1::2])
    assert torch.equal(out[:, 2, 0::2, 0::2], mosaic[:, 0, 0::2, 0::2])
    assert torch.equal(out[:, 0, 1::2, 1::2], mosaic[:, 0, 1::2, 1::2])
    assert float(out.max()) > 1.0

    # Interpolated positions are clamped, so they saturate at exactly 1.0.
    assert float(out[:, 1, 0::2, 0::2].max()) == 1.0
    assert float(out[:, 0, 0::2, 0::2].max()) == 1.0
    assert float(out[:, 2, 1::2, 1::2].max()) == 1.0


@pytest.mark.parametrize("shape,exc", [
    ((1, 3, 8, 8), RuntimeError),   # multi-channel: conv2d rejects it
    ((1, 8, 8), ValueError),        # unbatched: the 4-way unpack fails
    ((1, 1, 1, 8, 8), ValueError),  # 5-D: same
])
def test_wrongly_shaped_input_raises(gbtf, shape, exc):
    """
    forward() validates nothing explicitly (``B, _, H, W = mosaic.shape``), so
    this records which malformed inputs still fail loudly rather than silently
    demosaicing channel 0 of something.  It matters because a caller who
    forgets the pixel_shuffle would hand over a packed [B,4,h,w] tensor — which
    lands in the RuntimeError case, not in a wrong-but-quiet result.
    """
    with pytest.raises(exc):
        gbtf(torch.rand(*shape))

    # The specific mistake worth calling out: forgetting F.pixel_shuffle.
    with pytest.raises(RuntimeError):
        gbtf(packed_bayer(1, 8, 8, seed=1))
