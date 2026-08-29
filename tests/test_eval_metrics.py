"""
tests/test_eval_metrics.py — metric and IO utilities of test_dual_MoE_two_phase.py
==================================================================================
Everything under test here is module-level in the benchmark script (the eval
loop itself sits behind an ``if __name__ == "__main__"`` guard), so it can be
imported and exercised without a checkpoint, a dataset or a GPU.

Coverage:
  * importing the benchmark module is side-effect free (no benchmark run, no
    directories created, nothing printed);
  * ``psnr``      — the mse==0 branch, an analytic value, per-image averaging,
                    ``data_range`` scaling;
  * ``ssim``      — self-similarity == 1, symmetry, monotonicity in noise level,
                    boundedness, the grouped-conv C=1/C=3 paths, the
                    degenerate constant-image case, and the normalisation of
                    the Gaussian window (captured off ``F.conv2d``);
  * ``hdr_tonemap`` — must be BIT-IDENTICAL to the training script's version,
                    otherwise test PSNR-µ is not comparable with the W&B
                    training curves (the script says so itself);
  * ``packed_bayer_to_mosaic`` — the exact BGGR placement and the
                    pixel_unshuffle round trip;
  * ``save_jpg``  — a real, readable, correctly-sized JPEG; clamping instead of
                    crashing; the quality kwarg actually reaching PIL;
  * ``Tee``       — mirrors to file *and* stdout, makes its parent directory,
                    is line-buffered, restores stdout on close, works as a
                    context manager.  Every Tee test restores ``sys.stdout``
                    through a fixture teardown so a failure here cannot poison
                    the rest of the session.
"""

import inspect
import math
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import test_dual_MoE_two_phase as ev
import train_A100_MoE_two_phase as tr
from helpers import assert_finite, assert_shape, packed_bayer


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — see the harness rules)
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_image(b=1, c=3, h=32, w=32):
    """Deterministic horizontal ramp — a signal with structure but no noise."""
    ramp = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w)
    return ramp.expand(b, c, h, w).contiguous()


def _decoded_uint8(path, h, w):
    """Reopen a saved JPEG and return its pixels as a [h, w, 3] uint8 tensor."""
    with Image.open(str(path)) as im:
        raw = bytearray(im.convert("RGB").tobytes())
    return torch.frombuffer(raw, dtype=torch.uint8).view(h, w, 3)


def _spy_on_conv2d(monkeypatch):
    """
    Records every weight tensor handed to F.conv2d and returns the list.

    `ssim` builds its Gaussian window internally and never returns it, so this
    is the only way to assert directly on the window instead of inferring its
    shape from the score.  monkeypatch undoes the patch at teardown.
    """
    real_conv2d = torch.nn.functional.conv2d
    seen = []

    def spy(input, weight, *args, **kwargs):
        seen.append(weight.detach().clone())
        return real_conv2d(input, weight, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "conv2d", spy)
    return seen


@pytest.fixture
def stdout_guard():
    """
    Hands the test the current sys.stdout and *always* puts it back.

    Tee.close() assigns sys.stdout = sys.__stdout__, which would otherwise
    detach pytest's capture for every test that runs after a failure here.
    """
    saved = sys.stdout
    try:
        yield saved
    finally:
        sys.stdout = saved


# ═════════════════════════════════════════════════════════════════════════════
# Import purity
# ═════════════════════════════════════════════════════════════════════════════

def test_import_does_not_expose_main_only_configuration():
    """
    The benchmark's configuration (CHECKPOINT, OUTPUT_DIR, DEVICE, ...) must
    stay inside the __main__ guard while the utilities stay importable.

    If any of it leaked to module level, importing this module for a unit test
    would pin a hardcoded checkpoint path / dataset dir and could allocate a
    CUDA device on a login node.
    """
    for util in ("Tee", "hdr_tonemap", "psnr", "ssim",
                 "packed_bayer_to_mosaic", "save_jpg",
                 "load_model_from_checkpoint", "infer_full", "infer_patches",
                 "estimate_flops"):
        assert hasattr(ev, util), f"{util} should be importable from the module"

    for guarded in ("CHECKPOINT", "OUTPUT_DIR", "DATASET_DIR", "DEVICE",
                    "MODEL_KWARGS", "INFERENCE", "METRIC_MU", "PATCH_SIZE",
                    "test_loader", "test_dataset", "model", "metrics"):
        assert not hasattr(ev, guarded), \
            f"{guarded} escaped the __main__ guard — importing the module now has side effects"


def test_import_creates_no_files_and_prints_nothing(tmp_path):
    """
    A fresh interpreter importing the benchmark must not run the benchmark:
    no output directory, no log.txt, no results.csv, no stdout chatter.

    Run in a scratch cwd so that any relative path the module might create
    (OUTPUT_DIR is a relative "test_results/..." string) shows up here.
    """
    repo_root = os.path.dirname(os.path.abspath(ev.__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "4"
    env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.run(
        [sys.executable, "-c",
         "import test_dual_MoE_two_phase as m; print('IMPORTED', m.__name__)"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"import failed:\n{proc.stderr}"
    assert proc.stdout.strip() == "IMPORTED test_dual_MoE_two_phase", \
        f"import printed unexpected output:\n{proc.stdout}"
    assert list(tmp_path.iterdir()) == [], \
        f"import created files: {[p.name for p in tmp_path.iterdir()]}"


# ═════════════════════════════════════════════════════════════════════════════
# psnr
# ═════════════════════════════════════════════════════════════════════════════

def test_psnr_identical_images_return_100():
    """
    mse == 0 must take the sentinel branch and yield exactly 100.0.

    Without it the score is 10*log10(1/0) = +inf, which poisons the running
    mean of the whole benchmark.
    """
    x = torch.rand(2, 3, 16, 16)
    out = ev.psnr(x, x)
    assert out == pytest.approx(100.0, abs=0.0), out
    assert math.isfinite(out)


def test_psnr_analytic_value_for_known_mse():
    """A constant 0.1 error over the whole image is exactly 20 dB at data_range=1."""
    gt = torch.zeros(1, 3, 16, 16)
    pred = torch.full((1, 3, 16, 16), 0.1)
    expected = 10.0 * math.log10(1.0 / 0.01)          # 20 dB
    assert ev.psnr(pred, gt) == pytest.approx(expected, abs=1e-4)


def test_psnr_averages_per_image_not_pooled_mse():
    """
    PSNR is the mean of the per-image scores, NOT 10log10(1/pooled_mse).

    The two differ (16.99 vs 16.02 dB here); reporting the pooled version would
    silently understate every multi-image average in the results CSV.
    """
    gt = torch.zeros(2, 3, 8, 8)
    pred = gt.clone()
    pred[0] += 0.1                                     # mse 0.01 -> 20 dB
    pred[1] += 0.2                                     # mse 0.04 -> ~13.98 dB
    per_image_mean = 0.5 * (10 * math.log10(1 / 0.01) + 10 * math.log10(1 / 0.04))
    pooled = 10 * math.log10(1 / 0.025)
    got = ev.psnr(pred, gt)
    assert got == pytest.approx(per_image_mean, abs=1e-3)
    assert abs(got - pooled) > 0.5, "psnr looks like it pooled the batch MSE"


def test_psnr_zero_branch_is_per_image():
    """
    The mse==0 sentinel is applied per image, so a batch of [identical, noisy]
    averages 100 dB with the noisy image's real score.
    """
    gt = torch.zeros(2, 3, 8, 8)
    pred = gt.clone()
    pred[1] += 0.1
    expected = 0.5 * (100.0 + 10 * math.log10(1 / 0.01))
    assert ev.psnr(pred, gt) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("data_range", [1.0, 2.0, 255.0])
def test_psnr_respects_data_range(data_range):
    """
    data_range enters as 10log10(range^2/mse), i.e. +20log10(range) dB over the
    unit-range score.  The benchmark feeds [0,1] tensors, so a wrong default
    would shift every reported number by a constant.
    """
    gt = torch.zeros(1, 1, 12, 12)
    pred = torch.full((1, 1, 12, 12), 0.1)
    base = 10 * math.log10(1 / 0.01)
    expected = base + 20 * math.log10(data_range)
    assert ev.psnr(pred, gt, data_range=data_range) == pytest.approx(expected, abs=1e-3)


def test_psnr_returns_python_float_and_does_not_need_grad():
    """
    psnr must return a plain float (it is appended to lists and formatted), and
    must not propagate autograd state — it is called inside the eval loop on
    tensors that may still carry graph history.
    """
    gt = torch.rand(1, 3, 8, 8)
    pred = (gt + 0.05).requires_grad_(True)
    out = ev.psnr(pred, gt)
    assert isinstance(out, float)
    assert math.isfinite(out)


# ═════════════════════════════════════════════════════════════════════════════
# ssim
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["random", "constant", "gradient", "zeros"])
def test_ssim_self_similarity_is_one(name):
    """
    SSIM(x, x) == 1 for every image type, including the degenerate constant and
    all-zero cases where both local variances are 0 (num and den must cancel
    rather than divide 0/0).
    """
    images = {
        "random": torch.rand(1, 3, 32, 32),
        "constant": torch.full((1, 3, 32, 32), 0.37),
        "gradient": _gradient_image(),
        "zeros": torch.zeros(1, 3, 32, 32),
    }
    x = images[name]
    got = ev.ssim(x, x)
    assert math.isfinite(got)
    assert got == pytest.approx(1.0, abs=1e-5), f"SSIM({name},{name}) = {got}"


def test_ssim_is_symmetric():
    """
    SSIM(a,b) == SSIM(b,a).  Asymmetry would mean the pred/gt argument order
    changes the reported score, making runs incomparable.
    """
    a = torch.rand(1, 3, 32, 32)
    b = (a + 0.05 * torch.randn_like(a)).clamp(0, 1)
    assert ev.ssim(a, b) == pytest.approx(ev.ssim(b, a), abs=1e-6)


def test_ssim_decreases_monotonically_with_noise():
    """
    Three increasing noise levels must give strictly decreasing SSIM, and all
    must sit below the perfect score.  This is the property the benchmark
    actually relies on when it compares denoisers.
    """
    clean = _gradient_image(h=48, w=48) * 0.8 + 0.1
    scores = []
    for sigma in (0.01, 0.05, 0.2):
        gen = torch.Generator().manual_seed(7)
        noisy = (clean + sigma * torch.randn(clean.shape, generator=gen)).clamp(0, 1)
        scores.append(ev.ssim(noisy, clean))

    assert all(math.isfinite(s) for s in scores), scores
    assert scores[0] < 1.0
    assert scores[0] > scores[1] > scores[2], f"not monotone in noise: {scores}"


@pytest.mark.parametrize("pair", ["identical", "noisy", "anticorrelated",
                                  "zero_vs_one", "impulse"])
def test_ssim_is_bounded_in_minus_one_one(pair):
    """
    SSIM is mathematically bounded by [-1, 1]; a value outside it means the
    variance terms went numerically wrong (negative variance from the
    conv-of-squares trick, or a mis-normalised window).
    """
    a = torch.rand(1, 3, 32, 32)
    pairs = {
        "identical": (a, a.clone()),
        "noisy": (a, (a + 0.3 * torch.randn_like(a)).clamp(0, 1)),
        "anticorrelated": (a, 1.0 - a),
        "zero_vs_one": (torch.zeros(1, 1, 24, 24), torch.ones(1, 1, 24, 24)),
        "impulse": (torch.eye(32).view(1, 1, 32, 32),
                    1.0 - torch.eye(32).view(1, 1, 32, 32)),
    }
    x, y = pairs[pair]
    got = ev.ssim(x, y)
    assert math.isfinite(got), f"SSIM({pair}) is not finite: {got}"
    assert -1.0 - 1e-6 <= got <= 1.0 + 1e-6, f"SSIM({pair}) out of range: {got}"


def test_ssim_constant_pair_does_not_divide_by_zero():
    """
    Two different constant images have zero variance everywhere, so the
    structure term is C2/C2; only the stabiliser C1 keeps the luminance term
    alive.  The result must be finite and in range (it is < 1 because the two
    constants differ).
    """
    a = torch.full((1, 1, 24, 24), 0.3)
    b = torch.full((1, 1, 24, 24), 0.7)
    got = ev.ssim(a, b)
    assert math.isfinite(got), got
    assert -1.0 - 1e-6 <= got < 1.0, got


@pytest.mark.parametrize("channels", [1, 3])
def test_ssim_handles_channel_counts(channels):
    """
    The grouped-conv path uses kernel.expand(C, 1, k, k) with groups=C, so C
    must be read from the input rather than hardcoded to 3.  C=1 is used for
    single-plane comparisons, C=3 for the RGB metrics.
    """
    x = torch.rand(2, channels, 32, 32)
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    assert ev.ssim(x, x) == pytest.approx(1.0, abs=1e-5)
    got = ev.ssim(x, y)
    assert math.isfinite(got) and 0.0 < got < 1.0, got


def test_ssim_treats_channels_independently():
    """
    With groups=C every channel is filtered by its own copy of the window, so
    replicating a 1-channel pair into 3 channels must not change the score.
    If the grouped conv were wired wrong (channels summed together), the 3-ch
    score would drift away from the 1-ch score.
    """
    x1 = torch.rand(1, 1, 32, 32)
    y1 = (x1 + 0.08 * torch.randn_like(x1)).clamp(0, 1)
    x3 = x1.expand(1, 3, 32, 32).contiguous()
    y3 = y1.expand(1, 3, 32, 32).contiguous()
    assert ev.ssim(x3, y3) == pytest.approx(ev.ssim(x1, y1), abs=1e-6)


def test_ssim_averages_over_the_batch():
    """
    The score is a mean over batch and channels, so a batch of [perfect, noisy]
    must land midway between the two individual scores.
    """
    x = torch.rand(1, 3, 32, 32)
    y = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    both_x = torch.cat([x, x], dim=0)
    x_then_y = torch.cat([x, y], dim=0)
    single = ev.ssim(y, x)
    expected = 0.5 * (ev.ssim(x, x) + single)
    assert ev.ssim(x_then_y, both_x) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("window_size", [7, 11])
def test_ssim_gaussian_window_is_normalised(monkeypatch, window_size):
    """
    The window handed to every conv must be a normalised 2-D Gaussian
    (sigma=1.5, sums to 1) shaped [C, 1, k, k] for the grouped conv.

    A window that does not sum to 1 turns the "local mean" convolutions into a
    scaled mean, which biases mu, the variances and hence every SSIM number.
    Captured straight off F.conv2d because ssim never returns the kernel.
    """
    seen = _spy_on_conv2d(monkeypatch)
    channels = 3
    x = torch.rand(1, channels, 32, 32)
    ev.ssim(x, x, window_size=window_size)

    # mu1, mu2, s1, s2, s12
    assert len(seen) == 5, f"expected 5 convolutions, saw {len(seen)}"
    for k in seen:
        assert_shape(k, (channels, 1, window_size, window_size), "ssim window")

    kernel = seen[0][0, 0]
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6), \
        f"window sums to {float(kernel.sum())}, not 1"
    assert bool((kernel >= 0).all()), "Gaussian window must be non-negative"

    # every group shares the same window (the .expand)
    for c in range(channels):
        assert torch.equal(seen[0][c, 0], kernel)

    # separable Gaussian with sigma = 1.5, peaked at the centre
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    expected = g[:, None] * g[None, :]
    assert torch.allclose(kernel, expected, atol=1e-6)
    centre = window_size // 2
    assert kernel.argmax().item() == centre * window_size + centre


# ═════════════════════════════════════════════════════════════════════════════
# hdr_tonemap — cross-file invariant
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mu", [1.0, 100, 5000, 20000])
def test_hdr_tonemap_identical_to_training_script(mu):
    """
    The benchmark's own comment says a different µ-law makes test PSNR-µ
    incomparable to the W&B training curves.  The two implementations must
    therefore agree bit for bit, not merely approximately.
    """
    x = torch.tensor([[[[0.0, 1e-6, 1e-3, 0.25, 0.5, 0.75, 1.0, 4.0]]]])
    a = ev.hdr_tonemap(x, mu)
    b = tr.hdr_tonemap(x, mu)
    assert torch.equal(a, b), f"mu={mu}: eval {a} != train {b}"


def test_hdr_tonemap_default_mu_matches_training_default():
    """
    Both scripts must default to mu=5000; a silent divergence in the default is
    exactly the mismatch the comment warns about, since the benchmark calls
    hdr_tonemap(x, METRIC_MU) but training may rely on the default.
    """
    d_eval = inspect.signature(ev.hdr_tonemap).parameters["mu"].default
    d_train = inspect.signature(tr.hdr_tonemap).parameters["mu"].default
    assert d_eval == d_train == 5000, (d_eval, d_train)

    x = torch.rand(1, 3, 8, 8)
    assert torch.equal(ev.hdr_tonemap(x), tr.hdr_tonemap(x))
    assert torch.equal(ev.hdr_tonemap(x), ev.hdr_tonemap(x, 5000))


def test_hdr_tonemap_is_a_monotone_map_of_unit_range():
    """
    µ-law must be strictly increasing and map [0,1] onto [0,1] (0->0, 1->1);
    the visualisation path clamps to [0,1] afterwards and would crush the
    highlights if the normalisation were wrong.
    """
    x = torch.linspace(0.0, 1.0, 257).view(1, 1, 1, -1)
    y = ev.hdr_tonemap(x)
    assert_finite(y, "tonemapped")
    assert float(y[..., 0]) == pytest.approx(0.0, abs=0.0)
    assert float(y[..., -1]) == pytest.approx(1.0, abs=1e-6)
    diffs = y[..., 1:] - y[..., :-1]
    assert bool((diffs > 0).all()), "tonemap is not strictly increasing"
    # It is a compressive (concave) map: it lifts midtones.
    assert float(y[..., 128]) > float(x[..., 128])


# ═════════════════════════════════════════════════════════════════════════════
# packed_bayer_to_mosaic
# ═════════════════════════════════════════════════════════════════════════════

def test_packed_bayer_to_mosaic_shape():
    """[B,4,h,w] -> [B,1,2h,2w]; non-square and batched inputs must not transpose."""
    packed = packed_bayer(batch=2, h=6, w=10, seed=3)
    mosaic = ev.packed_bayer_to_mosaic(packed)
    assert_shape(mosaic, (2, 1, 12, 20), "mosaic")


def test_packed_bayer_to_mosaic_bggr_placement():
    """
    The intra-cell layout must be exactly ch0=B->(0,0), ch1=G1->(0,1),
    ch2=G2->(1,0), ch3=R->(1,1).  A swapped G2/R (the classic BGGR/RGGB
    mix-up) would silently feed the GBTF demosaicer red as green, wrecking
    every GT/noisy reference image in the benchmark.
    """
    B, G1, G2, R = 0.1, 0.2, 0.3, 0.4
    packed = torch.zeros(1, 4, 3, 5)
    for c, v in enumerate((B, G1, G2, R)):
        packed[0, c] = v
    mosaic = ev.packed_bayer_to_mosaic(packed)[0, 0]

    assert mosaic.shape == (6, 10)
    assert torch.allclose(mosaic[0::2, 0::2], torch.full((3, 5), B))
    assert torch.allclose(mosaic[0::2, 1::2], torch.full((3, 5), G1))
    assert torch.allclose(mosaic[1::2, 0::2], torch.full((3, 5), G2))
    assert torch.allclose(mosaic[1::2, 1::2], torch.full((3, 5), R))


def test_packed_bayer_to_mosaic_positionwise_against_hand_built_tensor():
    """
    Position-by-position check with all-distinct values, so no accidental
    symmetry can hide a wrong (row, col) offset for any channel.
    """
    h, w = 3, 4
    packed = torch.arange(4 * h * w, dtype=torch.float32).view(1, 4, h, w)
    mosaic = ev.packed_bayer_to_mosaic(packed)
    for c in range(4):
        dy, dx = c // 2, c % 2
        for y in range(h):
            for x in range(w):
                assert float(mosaic[0, 0, 2 * y + dy, 2 * x + dx]) == \
                    float(packed[0, c, y, x]), (c, y, x)


def test_packed_bayer_to_mosaic_roundtrips_with_pixel_unshuffle():
    """
    pixel_unshuffle(mosaic, 2) must return the original packed tensor exactly —
    the dataset packs with unshuffle and the benchmark unpacks with shuffle, so
    they have to be strict inverses or the two domains drift apart.
    """
    packed = packed_bayer(batch=2, h=8, w=8, seed=11)
    mosaic = ev.packed_bayer_to_mosaic(packed)
    back = F.pixel_unshuffle(mosaic, 2)
    assert torch.equal(back, packed)


def test_packed_bayer_to_mosaic_preserves_values_and_dtype():
    """No rescaling, clamping or dtype change — it is a pure re-layout."""
    packed = packed_bayer(batch=1, h=8, w=8, seed=5) * 3.0 - 1.0
    mosaic = ev.packed_bayer_to_mosaic(packed)
    assert mosaic.dtype == packed.dtype
    assert float(mosaic.min()) == pytest.approx(float(packed.min()))
    assert float(mosaic.max()) == pytest.approx(float(packed.max()))
    assert float(mosaic.sum()) == pytest.approx(float(packed.sum()), rel=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# save_jpg
# ═════════════════════════════════════════════════════════════════════════════

def test_save_jpg_writes_readable_rgb_jpeg_of_right_size(tmp_path):
    """
    A [3,H,W] float tensor must land on disk as a real JPEG of size (W, H) in
    RGB.  Odd dimensions are included because the saved comparison strip is
    3 panels + 2px separators wide, i.e. never a round number.
    """
    t = torch.rand(3, 17, 23)
    path = tmp_path / "vis.jpg"
    ev.save_jpg(t, str(path))

    assert path.is_file() and path.stat().st_size > 0
    with Image.open(str(path)) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"
        assert im.size == (23, 17)      # PIL reports (width, height)


def test_save_jpg_clamps_out_of_range_instead_of_crashing(tmp_path):
    """
    The eval loop passes tonemapped tensors that can sit slightly outside
    [0,1]; save_jpg must clamp (not raise, not wrap around).  Negative regions
    must come back black and >1 regions white.
    """
    t = torch.empty(3, 8, 8)
    t[:, :, :4] = -3.0
    t[:, :, 4:] = 4.0
    path = tmp_path / "clamped.jpg"
    ev.save_jpg(t, str(path), quality=100)

    arr = _decoded_uint8(path, 8, 8)          # [H, W, 3]
    assert int(arr[:, :4, :].max()) <= 4, "negative input should clamp to black"
    assert int(arr[:, 4:, :].min()) >= 251, "input > 1 should clamp to white"


def test_save_jpg_does_not_mutate_the_input_tensor(tmp_path):
    """
    clamp() must produce a copy: the same tensor is saved and then reused for
    metrics in the eval loop, so an in-place clamp would silently alter the
    numbers reported for that image.
    """
    t = torch.randn(3, 8, 8) * 2.0
    original = t.clone()
    ev.save_jpg(t, str(tmp_path / "x.jpg"))
    assert torch.equal(t, original)


def test_save_jpg_roundtrips_a_smooth_image(tmp_path):
    """
    At quality=100 a smooth image must survive the JPEG trip to within a few
    levels — this pins that the tensor is scaled to 0..255 (and not, say,
    written as raw floats or transposed).
    """
    t = _gradient_image(b=1, c=3, h=32, w=32)[0]
    path = tmp_path / "smooth.jpg"
    ev.save_jpg(t, str(path), quality=100)

    got = _decoded_uint8(path, 32, 32).permute(2, 0, 1).float() / 255.0
    assert float((got - t).abs().mean()) < 0.02


def test_save_jpg_quality_kwarg_reaches_pil(tmp_path):
    """
    save_jpg exists only because save_image cannot pass `quality`; a low
    quality must therefore produce a visibly smaller file than a high one.
    """
    t = torch.rand(3, 64, 64)
    low = tmp_path / "low.jpg"
    high = tmp_path / "high.jpg"
    ev.save_jpg(t, str(low), quality=10)
    ev.save_jpg(t, str(high), quality=95)
    assert low.stat().st_size < high.stat().st_size


def test_save_jpg_supports_single_channel(tmp_path):
    """
    A [1,H,W] tensor is a legal float image; it must be written as an 8-bit
    grayscale JPEG rather than raising on the channel count.
    """
    path = tmp_path / "gray.jpg"
    ev.save_jpg(torch.rand(1, 12, 10), str(path))
    with Image.open(str(path)) as im:
        assert im.format == "JPEG"
        assert im.mode == "L"
        assert im.size == (10, 12)


# ═════════════════════════════════════════════════════════════════════════════
# Tee
# ═════════════════════════════════════════════════════════════════════════════

def test_tee_writes_to_both_log_file_and_terminal(tmp_path, capfd, stdout_guard):
    """
    Tee must duplicate output: the log file gets everything AND the real
    terminal still sees it.  Tee targets sys.__stdout__ (fd 1), so capfd is the
    right capture level to observe the terminal half.
    """
    path = tmp_path / "log.txt"
    tee = ev.Tee(str(path))
    try:
        sys.stdout = tee
        print("hello benchmark")
        print("second line")
    finally:
        sys.stdout = stdout_guard
        tee.close()

    on_disk = path.read_text()
    assert "hello benchmark" in on_disk
    assert "second line" in on_disk

    captured = capfd.readouterr().out
    assert "hello benchmark" in captured, "Tee did not mirror to the terminal"
    assert "second line" in captured


def test_tee_creates_missing_parent_directories(tmp_path, stdout_guard):
    """
    OUTPUT_DIR is derived from the checkpoint name and may not exist yet, so
    Tee has to makedirs its parent chain (recursively) instead of raising
    FileNotFoundError before the benchmark ever starts.
    """
    path = tmp_path / "deep" / "nested" / "run" / "log.txt"
    assert not path.parent.exists()
    tee = ev.Tee(str(path))
    try:
        tee.write("created\n")
    finally:
        tee.close()
        sys.stdout = stdout_guard
    assert path.parent.is_dir()
    assert "created" in path.read_text()


def test_tee_accepts_a_bare_filename(tmp_path, monkeypatch, stdout_guard):
    """
    os.path.dirname("log.txt") is "" — the `or "."` fallback must keep makedirs
    from blowing up when the log path has no directory component.
    """
    monkeypatch.chdir(tmp_path)
    tee = ev.Tee("log.txt")
    try:
        tee.write("bare\n")
    finally:
        tee.close()
        sys.stdout = stdout_guard
    assert (tmp_path / "log.txt").read_text() == "bare\n"


def test_tee_is_line_buffered_so_content_hits_disk_before_close(tmp_path, stdout_guard):
    """
    The benchmark runs for hours and the log is tailed while it runs, so a
    completed line must be readable from another handle before close().
    """
    path = tmp_path / "log.txt"
    tee = ev.Tee(str(path))
    try:
        tee.write("streamed line\n")
        assert "streamed line" in path.read_text(), \
            "log file is not line-buffered — nothing on disk until close()"
        tee.write("no newline yet")
        tee.flush()
        assert "no newline yet" in path.read_text(), "flush() did not reach the file"
    finally:
        tee.close()
        sys.stdout = stdout_guard


def test_tee_close_restores_stdout_and_closes_the_file(tmp_path, stdout_guard):
    """
    close() must put sys.stdout back (to the real sys.__stdout__ it saved) and
    close the log handle, otherwise the benchmark leaves the interpreter with a
    dangling Tee and an unflushed file.
    """
    path = tmp_path / "log.txt"
    tee = ev.Tee(str(path))
    sys.stdout = tee
    try:
        print("before close")
        tee.close()
        assert sys.stdout is not tee
        assert sys.stdout is sys.__stdout__
        assert tee._log.closed
    finally:
        sys.stdout = stdout_guard
    assert "before close" in path.read_text()


def test_tee_works_as_a_context_manager(tmp_path, capfd, stdout_guard):
    """
    The docstring advertises `with Tee(path)`: __enter__ returns the Tee and
    __exit__ must close it (restoring stdout) even though the body replaced
    sys.stdout itself.
    """
    path = tmp_path / "ctx.txt"
    try:
        with ev.Tee(str(path)) as tee:
            sys.stdout = tee
            print("inside context")
            assert not tee._log.closed
        assert tee._log.closed
        assert sys.stdout is sys.__stdout__
    finally:
        sys.stdout = stdout_guard

    assert "inside context" in path.read_text()
    assert "inside context" in capfd.readouterr().out


def test_tee_overwrites_an_existing_log(tmp_path, stdout_guard):
    """
    The log is opened in "w" mode: a re-run into the same OUTPUT_DIR must start
    from an empty log rather than appending to the previous run's output.
    """
    path = tmp_path / "log.txt"
    path.write_text("stale content from a previous run\n")
    tee = ev.Tee(str(path))
    try:
        tee.write("fresh\n")
    finally:
        tee.close()
        sys.stdout = stdout_guard
    text = path.read_text()
    assert "stale content" not in text
    assert text == "fresh\n"


def test_tee_flush_does_not_raise_and_forwards_to_both(tmp_path, capfd, stdout_guard):
    """
    torch/DataLoader code paths call sys.stdout.flush(); Tee must forward it to
    the terminal and the log instead of raising AttributeError.
    """
    path = tmp_path / "log.txt"
    tee = ev.Tee(str(path))
    try:
        sys.stdout = tee
        sys.stdout.write("flushed\n")
        sys.stdout.flush()
    finally:
        sys.stdout = stdout_guard
        tee.close()
    assert "flushed" in path.read_text()
    assert "flushed" in capfd.readouterr().out
