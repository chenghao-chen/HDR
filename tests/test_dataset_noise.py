"""
tests/test_dataset_noise.py — exhaustive tests for HDR_Mobile_dataset.add_photon_noise
=====================================================================================

`add_photon_noise` is the only place where the training/eval pairs are
synthesised, so every property here is a property of the data the network
actually sees:

  * the (noisy, gt) pair must be shape-, dtype- and range-compatible with the
    models (`[0, 1]`, finite, same shape as the stored packed-Bayer tensor);
  * `noisy` must live on the n-bit lattice (it is what a real ADC would emit)
    while `gt` must stay continuous (it is the regression target);
  * the noise magnitude must obey  var[DN] = shot_gain * signal[DN] + read_var
    — the module docstring calls out that an earlier version capped this at
    ~14 DN² (~100x too weak), so this is a regression guard for the headline
    fix of the module;
  * an explicit `torch.Generator` must make the call bit-exact, because
    `MobileHDRDataset` relies on that for the reproducible test-split
    benchmark;
  * the norm_min/norm_max override must win over the tensor's own min/max, so
    that a dark crop of a bright frame keeps its absolute brightness.

Everything is CPU-only and small; the whole file runs in a few seconds.
"""

import inspect
import math

import pytest
import torch

from HDR_Mobile_dataset import add_photon_noise

from helpers import assert_finite, assert_in_range, assert_shape, packed_bayer_3d


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — conftest/helpers are shared)
# ─────────────────────────────────────────────────────────────────────────────

def pix_max_for(nbits):
    """Full-scale digital number for an nbits ADC — mirrors the source."""
    return float(2 ** nbits - 1)


def gen(seed):
    """A fresh, independently seeded CPU generator."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def dn(t, nbits):
    """Convert a [0,1] output tensor back to digital numbers."""
    return t * pix_max_for(nbits)


def draw_alphas(n, seed=7, nbits=10):
    """
    Recover the hidden `alpha` draws exactly.

    With a 1-pixel image of value 1.0 and the range pinned to [0, 1], the
    normalised signal is exactly pix_max, so
        gt = (pix_max * alpha).clamp(0, pix_max) / pix_max == alpha.
    That makes `gt` a lossless probe of the alpha sampler.
    """
    g = gen(seed)
    ones = torch.ones(1, 1)
    out = []
    for _ in range(n):
        _, gt = add_photon_noise(ones, nbits=nbits, random_alpha=True,
                                 norm_min=0.0, norm_max=1.0, generator=g)
        out.append(gt.item())
    return torch.tensor(out)


def shipped_defaults():
    """
    The default arguments `add_photon_noise` actually ships with.

    Read from the signature rather than hardcoded, so the noise-level tests
    below assert something about the *code* and not about a copy of the
    docstring's numbers.
    """
    d = {k: v.default for k, v in inspect.signature(add_photon_noise).parameters.items()}
    return d


def measured_variance_dn(level, nbits=10, shot_gain=14.0, read_var=147.5,
                         seed=0, size=384):
    """
    Empirical per-pixel variance (in DN²) of the noise at a constant signal.

    `read_noise_range` is collapsed to a single value so the read term is
    deterministic and a single large frame is a valid variance sample.
    Returns (measured_var, expected_var, clipped_fraction).
    """
    pm = pix_max_for(nbits)
    img = torch.full((size, size), float(level))
    noisy, gt = add_photon_noise(
        img, nbits=nbits, random_alpha=False, do_expand=False,
        shot_gain=shot_gain, read_noise_range=(read_var, read_var),
        norm_min=0.0, norm_max=1.0, generator=gen(seed))
    vals = dn(noisy, nbits)
    signal = level * pm
    expected = shot_gain * signal + read_var
    clipped = ((vals <= 0.0) | (vals >= pm)).float().mean().item()
    return float(vals.var()), expected, clipped


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signature / contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [
    (4, 8, 8),        # unbatched packed Bayer — what the Dataset stores
    (4, 16, 24),      # non-square
    (2, 4, 8, 8),     # batched
    (8, 8),           # bare 2D (single Bayer plane)
    (1,),             # single pixel
])
def test_returns_pair_with_input_shape(shape):
    """
    Both returned tensors must have exactly the input shape.

    MobileHDRDataset hands the result straight to the collate functions, which
    index `[4, H, W]`; any silent reshape/broadcast here would corrupt the
    batch layout rather than raise.
    """
    x = torch.rand(*shape)
    noisy, gt = add_photon_noise(x, nbits=10, generator=gen(0))
    assert_shape(noisy, shape, "noisy")
    assert_shape(gt, shape, "gt")


@pytest.mark.parametrize("nbits", [10, 12])
@pytest.mark.parametrize("random_alpha", [False, True])
@pytest.mark.parametrize("do_expand", [False, True])
def test_outputs_finite_and_in_unit_range(nbits, random_alpha, do_expand):
    """
    Every code-path combination must yield finite outputs inside [0, 1].

    The models and the tone-mapper assume normalised inputs; an out-of-range
    or non-finite sample poisons a whole batch's loss with no error message.
    """
    x = packed_bayer_3d(h=16, w=16, seed=3)
    noisy, gt = add_photon_noise(x, nbits=nbits, random_alpha=random_alpha,
                                 do_expand=do_expand, generator=gen(11))
    assert_finite(noisy, "noisy")
    assert_finite(gt, "gt")
    assert_in_range(noisy, 0.0, 1.0, "noisy", atol=0.0)
    assert_in_range(gt, 0.0, 1.0, "gt", atol=0.0)


def test_unit_range_holds_when_signal_overflows_the_override_range():
    """
    Values far outside [norm_min, norm_max] must clamp, not escape [0, 1].

    A crop can legitimately contain the full frame's maximum while the cached
    range came from a different (e.g. re-scaled) source; the clamp is the only
    thing keeping the target inside the model's output range.
    """
    x = torch.rand(4, 8, 8) * 10.0 - 5.0          # spans roughly [-5, 5]
    noisy, gt = add_photon_noise(x, nbits=10, random_alpha=False,
                                 norm_min=0.0, norm_max=0.1, generator=gen(1))
    assert_in_range(noisy, 0.0, 1.0, "noisy", atol=0.0)
    assert_in_range(gt, 0.0, 1.0, "gt", atol=0.0)
    # Both extremes must actually be exercised by this input.
    assert float(gt.max()) == pytest.approx(1.0), "bright pixels must saturate"
    assert float(gt.min()) == 0.0, "sub-norm_min pixels must clamp to black"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    """
    Output dtype follows the input dtype.

    The Gaussian draw is created with `dtype=clean.dtype`; a silent promotion
    to float64 would double dataloader memory and break pinned-memory
    transfers downstream.
    """
    x = torch.rand(4, 8, 8, dtype=dtype)
    noisy, gt = add_photon_noise(x, nbits=10, generator=gen(0))
    assert noisy.dtype is dtype
    assert gt.dtype is dtype


def test_input_tensor_is_not_mutated():
    """
    The caller's tensor must be left untouched.

    MobileHDRDataset loads with `mmap=True`, so an in-place write here would
    either raise or corrupt the on-disk cache for every later epoch.
    """
    x = packed_bayer_3d(h=8, w=8, seed=5)
    before = x.clone()
    add_photon_noise(x, nbits=10, do_expand=True, generator=gen(29))
    assert torch.equal(x, before), "add_photon_noise mutated its input"


def test_accepts_readonly_mmap_tensor(tmp_path):
    """
    A tensor loaded exactly the way the Dataset loads it must work.

    `torch.load(..., mmap=True)` returns a read-only-backed tensor; this pins
    down that no operation in the noise path needs a writable input storage.
    """
    path = tmp_path / "frame.pt"
    torch.save(packed_bayer_3d(h=16, w=16, seed=2), str(path))
    t = torch.load(str(path), weights_only=True, mmap=True)
    noisy, gt = add_photon_noise(t, nbits=10, generator=gen(0))
    assert_shape(noisy, (4, 16, 16), "noisy")
    assert_finite(noisy, "noisy")
    assert_finite(gt, "gt")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quantisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nbits", [10, 12])
def test_noisy_lies_exactly_on_the_nbit_lattice(nbits):
    """
    `noisy` is round()ed in DN, so noisy * pix_max must be an integer.

    This is what makes the synthetic input look like real ADC output; losing
    it (e.g. by dividing before rounding) would hand the network a
    quantisation-free input it will never see at deployment.
    """
    pm = pix_max_for(nbits)
    x = packed_bayer_3d(h=24, w=24, seed=nbits)
    noisy, _ = add_photon_noise(x, nbits=nbits, random_alpha=False,
                                generator=gen(4))
    vals = dn(noisy, nbits)
    residual = (vals - vals.round()).abs().max()
    assert float(residual) < 1e-3, \
        f"noisy is off the 1/{pm:.0f} lattice by up to {float(residual)} DN"
    # And the lattice really has pix_max+1 levels, not more.
    assert float(vals.max()) <= pm
    assert float(vals.min()) >= 0.0


@pytest.mark.parametrize("nbits", [10, 12])
def test_noisy_quantisation_step_is_one_over_pix_max(nbits):
    """
    The gap between neighbouring distinct `noisy` values is a multiple of
    1/pix_max, and the smallest observed gap is exactly 1/pix_max.

    Pins the *scale* of the lattice, not just its integrality: a wrong
    pix_max (e.g. 2**nbits instead of 2**nbits - 1) still yields integers in
    its own units but shifts every value.
    """
    pm = pix_max_for(nbits)
    x = torch.rand(64, 64)
    noisy, _ = add_photon_noise(x, nbits=nbits, random_alpha=False,
                                generator=gen(9))
    uniq = torch.unique(noisy)
    assert uniq.numel() > 50, "need many distinct levels for this to be meaningful"
    steps = (uniq[1:] - uniq[:-1]) * pm
    assert float((steps - steps.round()).abs().max()) < 1e-3
    assert float(steps.min()) == pytest.approx(1.0, abs=1e-3), \
        f"smallest level gap is {float(steps.min())}/pix_max, expected 1"


def test_gt_is_not_quantised():
    """
    `gt` skips the round(), so it must NOT sit on the 1/pix_max lattice.

    The target has to keep sub-LSB detail — otherwise the best achievable
    PSNR is capped by the quantiser and the reported numbers are meaningless.
    """
    pm = pix_max_for(10)
    # A ramp whose spacing is deliberately not a multiple of 1 DN.
    x = torch.linspace(0.031, 0.907, 97).view(1, 97, 1).expand(4, 97, 64).contiguous()
    _, gt = add_photon_noise(x, nbits=10, random_alpha=False,
                             norm_min=0.0, norm_max=1.0, generator=gen(0))
    vals = gt * pm
    off_lattice = (vals - vals.round()).abs() > 1e-3
    assert off_lattice.float().mean() > 0.9, \
        "gt appears quantised — most values land on the DN lattice"


def test_gt_is_a_smooth_monotone_signal():
    """
    `gt` preserves the monotone structure of the clean input; `noisy` does not.

    A smooth target is the whole point of the (noisy, gt) split: if the noise
    leaked into `gt` the loss would be asking the network to predict noise.
    """
    ramp = torch.linspace(0.0, 1.0, 97).view(97, 1).expand(97, 32).contiguous()
    noisy, gt = add_photon_noise(ramp, nbits=10, random_alpha=False,
                                 norm_min=0.0, norm_max=1.0, generator=gen(2))
    col_gt = gt[:, 0]
    assert bool((col_gt[1:] - col_gt[:-1] >= -1e-7).all()), "gt is not monotone"
    # The noisy column is dominated by noise, so it is far from monotone.
    noisy_col = noisy[:, 0]
    violations = (noisy_col[1:] - noisy_col[:-1] < 0).float().mean()
    assert float(violations) > 0.1, "noisy looks noise-free — no noise was added?"


def test_gt_differs_from_noisy():
    """
    The two outputs must be genuinely different tensors.

    A copy-paste slip returning `clean` twice would silently make the task
    trivial and every training run would look wonderful.
    """
    x = packed_bayer_3d(h=16, w=16, seed=6)
    noisy, gt = add_photon_noise(x, nbits=10, random_alpha=False,
                                 generator=gen(3))
    assert not torch.equal(noisy, gt)
    # The difference is real noise, not a rounding artefact (>> 1 LSB).
    diff_dn = (dn(noisy, 10) - dn(gt, 10)).abs()
    assert float(diff_dn.mean()) > 5.0, \
        f"mean |noisy-gt| is only {float(diff_dn.mean()):.2f} DN"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Determinism
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("random_alpha,do_expand", [
    (False, False), (True, False), (True, True),
])
def test_same_seed_generator_is_bit_exact(random_alpha, do_expand):
    """
    Two identically seeded generators must produce identical (noisy, gt).

    MobileHDRDataset's test split seeds a generator with
    `test_noise_seed + idx` precisely so the benchmark is byte-identical
    across runs; without this, PSNR comparisons between checkpoints are noise.
    """
    x = packed_bayer_3d(h=16, w=16, seed=8)
    kw = dict(nbits=10, random_alpha=random_alpha, do_expand=do_expand)
    n1, g1 = add_photon_noise(x, generator=gen(2025), **kw)
    n2, g2 = add_photon_noise(x, generator=gen(2025), **kw)
    assert torch.equal(n1, n2), "same seed produced different noise"
    assert torch.equal(g1, g2), "same seed produced different gt"


def test_different_seeds_give_different_noise():
    """
    Distinct seeds must decorrelate both the noise and the alpha draw.

    The test split derives its seed from the sample index; if the seed were
    ignored every test image would get the same noise field.
    """
    x = packed_bayer_3d(h=16, w=16, seed=8)
    n1, g1 = add_photon_noise(x, nbits=10, random_alpha=True, generator=gen(2025))
    n2, g2 = add_photon_noise(x, nbits=10, random_alpha=True, generator=gen(2026))
    assert not torch.equal(n1, n2)
    assert not torch.equal(g1, g2), "alpha did not change with the seed"


def test_explicit_generator_isolates_from_global_rng():
    """
    With an explicit generator the result must not depend on the global seed,
    and the call must not advance the global RNG.

    Both halves matter: the first keeps the test-split benchmark stable no
    matter what the training loop seeded, the second keeps the *augmentation*
    stream (which uses the global RNG) reproducible around it.
    """
    x = packed_bayer_3d(h=8, w=8, seed=1)

    torch.manual_seed(1)
    before_a = torch.rand(3)
    torch.manual_seed(1)
    n_a, gt_a = add_photon_noise(x, nbits=10, random_alpha=True, generator=gen(77))
    after_a = torch.rand(3)

    torch.manual_seed(999)
    n_b, gt_b = add_photon_noise(x, nbits=10, random_alpha=True, generator=gen(77))

    assert torch.equal(n_a, n_b), "explicit generator is contaminated by the global seed"
    assert torch.equal(gt_a, gt_b)
    assert torch.equal(before_a, after_a), "the call consumed global RNG draws"


def test_generator_none_uses_global_rng():
    """
    Without a generator the function must follow the global torch seed.

    The train split passes `generator=None` and relies on `seed_worker` for
    per-worker variety; if the noise ignored the global RNG every worker
    would emit the same noise field forever.
    """
    x = packed_bayer_3d(h=8, w=8, seed=1)
    torch.manual_seed(4242)
    n1, _ = add_photon_noise(x, nbits=10, random_alpha=True)
    torch.manual_seed(4242)
    n2, _ = add_photon_noise(x, nbits=10, random_alpha=True)
    n3, _ = add_photon_noise(x, nbits=10, random_alpha=True)   # RNG advanced
    assert torch.equal(n1, n2), "global-seeded call is not reproducible"
    assert not torch.equal(n1, n3), "consecutive calls reused the same noise"


# ─────────────────────────────────────────────────────────────────────────────
# 4. alpha (the low-light scaler)
# ─────────────────────────────────────────────────────────────────────────────

def test_random_alpha_false_means_alpha_exactly_one():
    """
    `random_alpha=False` must leave the normalised signal untouched.

    This is the path the test split always takes, so the eval target must be
    the *full-brightness* frame; a stray scaling here would silently change
    every reported PSNR.
    """
    pm = pix_max_for(10)
    x = packed_bayer_3d(h=16, w=16, seed=4)
    _, gt = add_photon_noise(x, nbits=10, random_alpha=False,
                             norm_min=0.0, norm_max=1.0, generator=gen(0))
    expected = ((x - 0.0) / 1.0 * pm).clamp(0.0, pm) / pm
    assert torch.allclose(gt, expected, atol=1e-7), \
        "gt is not the unscaled normalised signal"
    assert float(gt.max()) == pytest.approx(float(x.max()), abs=1e-6)


def test_random_alpha_is_triangular_and_biased_dark():
    """
    alpha = |U1 - U2| is triangular with mode 0 and mean 1/3.

    The module's whole low-light story rests on this bias: if alpha were
    uniform (mean 0.5) the dataset would contain far fewer of the severely
    under-exposed frames the denoiser is supposed to specialise in.
    """
    a = draw_alphas(4000, seed=7)
    assert float(a.mean()) < 0.5, "alpha is not biased toward darkness"
    assert float(a.mean()) == pytest.approx(1.0 / 3.0, abs=0.03), \
        f"mean alpha {float(a.mean()):.4f} is not the triangular 1/3"
    # Triangular density 2(1-x): the median is 1 - 1/sqrt(2) ~ 0.293.
    assert float(a.median()) == pytest.approx(1.0 - 1.0 / math.sqrt(2.0), abs=0.03)
    # Mode at 0 -> the lowest decile must be much more populated than the top.
    assert float((a < 0.1).float().mean()) > float((a > 0.9).float().mean())


def test_random_alpha_respects_its_floor_and_ceiling():
    """
    alpha is clamped to >= 0.01 and can never exceed 1.0.

    The floor is what stops a training pair from being pure black (an
    all-zero target makes the SNR map and the tone-mapped loss degenerate);
    the ceiling follows from |U1-U2| <= 1 and keeps gt inside [0, 1].
    """
    a = draw_alphas(4000, seed=7)
    assert float(a.min()) >= 0.01 - 1e-6, f"alpha fell to {float(a.min())}"
    assert float(a.max()) <= 1.0 + 1e-6
    # The clamp must actually be exercised: P(|U1-U2| < 0.01) ~ 2%.
    hits = (a <= 0.01 + 1e-6).sum().item()
    assert hits > 0, "the 0.01 floor never fired in 4000 draws"


def test_alpha_scales_the_clean_signal_linearly():
    """
    gt must be the *alpha-scaled* clean signal, i.e. a pure rescale of the
    unscaled gt — alpha may not distort the image's relative structure.

    If alpha were applied after normalisation-per-crop, or applied per pixel,
    the target would no longer be a plain exposure change of the input.
    """
    x = packed_bayer_3d(h=16, w=16, seed=12)
    _, gt_full = add_photon_noise(x, nbits=10, random_alpha=False,
                                  norm_min=0.0, norm_max=1.0, generator=gen(50))
    _, gt_dark = add_photon_noise(x, nbits=10, random_alpha=True,
                                  norm_min=0.0, norm_max=1.0, generator=gen(50))
    ratio = gt_dark[gt_full > 0.05] / gt_full[gt_full > 0.05]
    assert ratio.numel() > 100
    assert float(ratio.std()) < 1e-4, \
        f"alpha is not a single global scale (ratio std {float(ratio.std()):.2e})"
    assert 0.01 - 1e-6 <= float(ratio.mean()) <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 5. Noise magnitude — var[DN] = shot_gain * signal[DN] + read_var
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nbits,level", [
    (10, 0.20), (10, 0.40), (10, 0.60),
    (12, 0.30), (12, 0.50),
])
def test_measured_variance_matches_the_poisson_gaussian_model(nbits, level):
    """
    The empirical per-pixel variance in DN must equal
    shot_gain*signal + read_var.

    This is the module's headline claim (its docstring notes the previous
    version was ~100x too weak because it multiplied the *normalised* signal
    by shot_gain). Levels are chosen so clipping at 0/pix_max is negligible,
    which makes the sample variance an unbiased estimator.
    """
    meas, exp, clipped = measured_variance_dn(level, nbits=nbits, seed=nbits * 7)
    assert clipped < 1e-4, f"{clipped:.4f} of pixels clipped — estimator invalid"
    assert meas == pytest.approx(exp, rel=0.04), \
        f"nbits={nbits} level={level}: measured {meas:.1f} DN^2 vs expected {exp:.1f}"


def test_variance_is_linear_in_the_signal():
    """
    The slope of var(signal) must be shot_gain and the intercept read_var.

    Fitting both from two well-separated signal levels catches the class of
    bug the docstring warns about: a normalised-signal shot term collapses
    the slope to ~shot_gain/pix_max, i.e. essentially zero.
    """
    lo_level, hi_level = 0.20, 0.60
    pm = pix_max_for(10)
    lo, _, _ = measured_variance_dn(lo_level, seed=101)
    hi, _, _ = measured_variance_dn(hi_level, seed=202)
    slope = (hi - lo) / ((hi_level - lo_level) * pm)
    intercept = lo - slope * lo_level * pm
    assert slope == pytest.approx(14.0, rel=0.08), \
        f"fitted shot_gain {slope:.2f}, expected 14"
    assert intercept == pytest.approx(147.5, abs=120.0), \
        f"fitted read_var {intercept:.1f}, expected ~147.5"


def test_shot_term_scales_with_shot_gain():
    """
    Doubling `shot_gain` doubles the signal-dependent part of the variance,
    and shot_gain=0 leaves exactly the read-noise floor.

    Pins the parameter down as a real knob — the two-phase trainer may want to
    sweep it, and a hardcoded gain would make that a silent no-op.
    """
    level, read_var = 0.40, 150.0
    v0, _, _ = measured_variance_dn(level, shot_gain=0.0, read_var=read_var, seed=1)
    v1, _, _ = measured_variance_dn(level, shot_gain=14.0, read_var=read_var, seed=1)
    v2, _, _ = measured_variance_dn(level, shot_gain=28.0, read_var=read_var, seed=1)
    assert v0 == pytest.approx(read_var, rel=0.05), \
        f"shot_gain=0 gave var {v0:.1f}, expected the read floor {read_var}"
    assert (v2 - v0) / (v1 - v0) == pytest.approx(2.0, rel=0.05)


def test_noise_is_not_the_old_100x_too_weak_version():
    """
    Regression guard for the documented fix: sigma in the highlights must be
    ~120 DN at 10 bit, not ~4 DN.

    The buggy predecessor normalised to [0,1] before multiplying by
    shot_gain, capping var at ~14 DN^2 (sigma ~ 3.7 DN). Any reappearance of
    that shows up here as a two-orders-of-magnitude variance collapse.
    """
    # Signal 0.85*pix_max would clip, so measure at 0.6 and extrapolate the
    # model instead of the sample: check the fitted highlight sigma.
    d = shipped_defaults()
    shot_gain = float(d["shot_gain"])
    lo, hi = (float(v) for v in d["read_noise_range"])
    meas, exp, clipped = measured_variance_dn(
        0.60, shot_gain=shot_gain, read_var=0.5 * (lo + hi), seed=31)
    assert clipped < 1e-4
    # The old model capped var at shot_gain DN^2 (sigma ~ 3.7 DN); the fixed
    # one is ~100x that. Anything under 100x the cap is the bug coming back.
    assert meas > 100.0 * shot_gain, \
        f"variance {meas:.1f} DN^2 is in the range of the old capped model"
    # sigma in a full-scale highlight, computed from the SHIPPED defaults —
    # the docstring promises ~120 DN at 10 bit (12% of full scale).
    pm = pix_max_for(10)
    sigma_full = math.sqrt(shot_gain * pm + 0.5 * (lo + hi))
    assert sigma_full == pytest.approx(120.0, abs=3.0), \
        f"defaults give a highlight sigma of {sigma_full:.1f} DN, not ~120"
    assert sigma_full / pm > 0.10, "highlight noise is under 10% of full scale"


def test_shipped_defaults_are_the_high_noise_recipe():
    """
    A call that passes NO noise parameters must already produce the high-noise
    recipe: var in DN = 14 * signal + U(135, 160).

    Every other variance test pins `shot_gain`/`read_noise_range` explicitly,
    so this is the only guard against the defaults themselves regressing to
    the weak values the module docstring says were wrong. Both callers in
    MobileHDRDataset inherit those defaults.
    """
    d = shipped_defaults()
    assert d["nbits"] == 10, "the default bit depth changed"
    assert d["random_alpha"] is True and d["do_expand"] is False
    assert float(d["shot_gain"]) == pytest.approx(14.0)
    lo, hi = (float(v) for v in d["read_noise_range"])
    assert (lo, hi) == pytest.approx((135.0, 160.0))
    assert d["norm_min"] is None and d["norm_max"] is None
    assert d["generator"] is None

    # Now measure with defaults only (random_alpha off so the level is known).
    pm = pix_max_for(10)
    level = 0.40
    img = torch.full((384, 384), level)
    noisy, _ = add_photon_noise(img, random_alpha=False,
                               norm_min=0.0, norm_max=1.0, generator=gen(12))
    vals = dn(noisy, 10)
    assert float(((vals <= 0.0) | (vals >= pm)).float().mean()) < 1e-4
    meas = float(vals.var())
    signal = level * pm
    # read_var is a single unknown draw from [lo, hi]; allow 4% sampling slack.
    assert 0.96 * (14.0 * signal + lo) < meas < 1.04 * (14.0 * signal + hi), \
        f"default-parameter variance {meas:.1f} DN^2 outside " \
        f"[{14.0 * signal + lo:.1f}, {14.0 * signal + hi:.1f}]"


def test_read_var_is_drawn_uniformly_from_read_noise_range():
    """
    Each call draws read_var ~ U(read_noise_range) — one draw per call, shared
    by every pixel.

    With shot_gain=0 the per-call variance *is* read_var, so the spread of
    per-call variances directly exposes the sampler. A collapsed sampler
    (always the low end, or always the mean) would understate the black-level
    noise the denoiser must handle.
    """
    g = gen(17)
    img = torch.full((96, 96), 0.4)
    variances = []
    for _ in range(150):
        noisy, _ = add_photon_noise(img, nbits=12, random_alpha=False,
                                    shot_gain=0.0, read_noise_range=(135.0, 160.0),
                                    norm_min=0.0, norm_max=1.0, generator=g)
        variances.append(float(dn(noisy, 12).var()))
    v = torch.tensor(variances)
    # Per-call sampling error at 96*96 pixels is ~0.65% -> allow 4 DN^2 slack.
    assert float(v.min()) > 135.0 - 6.0, f"read_var dipped to {float(v.min()):.1f}"
    assert float(v.max()) < 160.0 + 6.0, f"read_var rose to {float(v.max()):.1f}"
    assert float(v.mean()) == pytest.approx(147.5, abs=4.0)
    # The full range must be used, not just its centre.
    assert float(v.min()) < 140.0 and float(v.max()) > 155.0, \
        "read_noise_range is not being spanned"


def test_noisy_is_an_unbiased_estimate_of_gt():
    """
    Averaging many noise draws must converge to `gt`.

    Zero-mean noise is what makes `gt` the correct regression target; a DC
    offset between the pair would teach the network a brightness shift.
    """
    g = gen(5)
    img = torch.full((64, 64), 0.40)
    acc = torch.zeros(64, 64)
    K = 150
    gt = None
    for _ in range(K):
        noisy, gt = add_photon_noise(img, nbits=10, random_alpha=False,
                                     norm_min=0.0, norm_max=1.0, generator=g)
        acc += noisy
    mean_dn = dn(acc / K, 10)
    gt_dn = dn(gt, 10)
    # sigma ~ 85 DN; the mean over K draws AND all 64*64 pixels has a standard
    # error of 85/sqrt(150*4096) ~ 0.11 DN, so a 1 DN budget is ~9 sigma.
    bias = float(mean_dn.mean() - gt_dn.mean())
    assert abs(bias) < 1.0, f"noise has a DC bias of {bias:.3f} DN relative to gt"
    # Per pixel the residual must still be pure sampling noise: E|.| for
    # 150 averaged draws is sigma/sqrt(K)*sqrt(2/pi) ~ 5.5 DN.
    assert float((mean_dn - gt_dn).abs().mean()) < 8.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Degenerate normalisation range
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,desc", [
    ({}, "constant image, auto min/max"),
    ({"norm_min": 0.5, "norm_max": 0.5}, "explicit equal bounds"),
    ({"norm_min": 0.0, "norm_max": 5e-7}, "range below the 1e-6 guard"),
    ({"norm_min": 1.0, "norm_max": 0.0}, "inverted bounds"),
])
def test_degenerate_range_does_not_divide_by_zero(kwargs, desc):
    """
    rng <= 1e-6 must take the `image * 0` path: finite outputs, gt all black.

    A flat frame is a real occurrence (a fully clipped or fully black capture)
    and dividing by rng would put NaN into the loss for the whole batch.
    """
    img = torch.full((4, 8, 8), 0.5)
    noisy, gt = add_photon_noise(img, nbits=10, random_alpha=False,
                                 generator=gen(1), **kwargs)
    assert_finite(noisy, f"noisy [{desc}]")
    assert_finite(gt, f"gt [{desc}]")
    assert_in_range(noisy, 0.0, 1.0, "noisy", atol=0.0)
    assert bool((gt == 0.0).all()), f"gt is not zeroed for {desc}"


def test_degenerate_range_still_carries_read_noise():
    """
    Even with the signal zeroed, `noisy` must show the read-noise floor
    (rectified by the clamp at 0), not be identically zero.

    It documents what a flat frame turns into: an all-black target plus pure
    read noise — a valid, if useless, pair, and definitely not a NaN.
    """
    read_var = 147.5
    img = torch.zeros(4, 64, 64)
    noisy, gt = add_photon_noise(img, nbits=10, random_alpha=False,
                                 read_noise_range=(read_var, read_var),
                                 generator=gen(1))
    vals = dn(noisy, 10)
    assert float(vals.max()) > 0.0, "no read noise at all"
    # A zero-mean Gaussian rectified at 0 has mean sigma/sqrt(2*pi) and
    # sd sigma*sqrt(1/2 - 1/(2*pi)); sigma = sqrt(147.5) ~ 12.1 DN.
    sigma = math.sqrt(read_var)
    assert float(vals.mean()) == pytest.approx(sigma / math.sqrt(2 * math.pi),
                                               rel=0.08)
    assert float(vals.std()) == pytest.approx(
        sigma * math.sqrt(0.5 - 1.0 / (2 * math.pi)), rel=0.08)
    # Exactly half the draws are clipped to black.
    assert float((vals == 0.0).float().mean()) == pytest.approx(0.5, abs=0.03)
    assert float(gt.max()) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. norm_min / norm_max override
# ─────────────────────────────────────────────────────────────────────────────

def test_override_keeps_a_dark_crop_dark():
    """
    Passing the FULL frame's range must stop a dark crop from being
    auto-stretched to full scale.

    This is the documented reason the Dataset caches per-file min/max: with
    per-crop normalisation every dark crop would arrive at the network as a
    bright one and the low-light branch would never see real darkness.
    """
    full = torch.rand(4, 32, 32)
    full[0, 0, 0] = 0.0
    full[0, 0, 1] = 1.0                      # full-frame range is [0, 1]
    crop = (full[:, 8:12, 8:12] * 0.05).clone()   # a genuinely dark crop

    _, gt_full_range = add_photon_noise(crop, nbits=10, random_alpha=False,
                                        norm_min=0.0, norm_max=1.0,
                                        generator=gen(3))
    _, gt_self_range = add_photon_noise(crop, nbits=10, random_alpha=False,
                                        generator=gen(3))
    assert float(gt_full_range.max()) < 0.06, \
        f"dark crop was stretched to {float(gt_full_range.max()):.3f}"
    assert float(gt_self_range.max()) == pytest.approx(1.0, abs=1e-6), \
        "self-normalisation should stretch the crop to full scale"
    assert float(gt_full_range.max()) == pytest.approx(float(crop.max()), abs=1e-3)


def test_override_preserves_relative_brightness_between_crops():
    """
    Two crops of the same frame, normalised by the shared frame range, keep
    their brightness ordering and ratio.

    Absolute-brightness consistency across crops is what lets a single model
    learn a signal-dependent noise prior; per-crop normalisation destroys it.
    """
    # Three bands so the frame range really is [0.0, 0.8]; the crops below are
    # taken from the two non-black bands.
    frame = torch.zeros(4, 24, 16)
    frame[:, 8:16, :] = 0.10
    frame[:, 16:24, :] = 0.80
    rmin, rmax = float(frame.min()), float(frame.max())
    assert rmin == 0.0

    dark = frame[:, 8:16, :].clone()
    bright = frame[:, 16:24, :].clone()
    _, gt_dark = add_photon_noise(dark, nbits=10, random_alpha=False,
                                  norm_min=rmin, norm_max=rmax, generator=gen(1))
    _, gt_bright = add_photon_noise(bright, nbits=10, random_alpha=False,
                                    norm_min=rmin, norm_max=rmax, generator=gen(1))
    assert float(gt_dark.mean()) < float(gt_bright.mean())
    # (0.10-0)/0.80 = 0.125 and (0.80-0)/0.80 = 1.0
    assert float(gt_dark.mean()) == pytest.approx(0.125, abs=2e-3)
    assert float(gt_bright.mean()) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("which", ["min_only", "max_only"])
def test_partial_override_falls_back_to_the_tensor(which):
    """
    Supplying only one bound must use the tensor's own value for the other.

    The `is None` checks make this legal; a truthiness test would break
    `norm_min=0.0`, the single most likely value a caller passes.
    """
    x = torch.rand(4, 8, 8) * 0.5 + 0.25            # values in [0.25, 0.75]
    pm = pix_max_for(10)
    if which == "min_only":
        _, gt = add_photon_noise(x, nbits=10, random_alpha=False,
                                 norm_min=0.0, generator=gen(0))
        expected = (x / float(x.max()) * pm).clamp(0, pm) / pm
    else:
        _, gt = add_photon_noise(x, nbits=10, random_alpha=False,
                                 norm_max=1.0, generator=gen(0))
        lo = float(x.min())
        expected = ((x - lo) / (1.0 - lo) * pm).clamp(0, pm) / pm
    assert torch.allclose(gt, expected, atol=1e-5), \
        f"{which} override did not fall back to the tensor's own bound"


def test_norm_min_zero_is_honoured_not_treated_as_missing():
    """
    `norm_min=0.0` must be respected even though it is falsy.

    MobileHDRDataset passes the cached per-file min, which is legitimately
    0.0 for most frames; a `if norm_min:` style check would silently switch
    those frames back to per-crop normalisation.
    """
    x = torch.rand(4, 8, 8) * 0.4 + 0.6            # min well above 0
    pm = pix_max_for(10)
    _, gt = add_photon_noise(x, nbits=10, random_alpha=False,
                             norm_min=0.0, norm_max=1.0, generator=gen(0))
    expected = (x * pm).clamp(0, pm) / pm
    assert torch.allclose(gt, expected, atol=1e-6)
    assert float(gt.min()) > 0.5, \
        "norm_min=0.0 was ignored — the crop got stretched from its own min"


# ─────────────────────────────────────────────────────────────────────────────
# 8. do_expand (highlight expansion)
# ─────────────────────────────────────────────────────────────────────────────

# Seeds chosen so the first generator draw decides the `< 0.3` branch
# (random_alpha=False, so no draws are consumed before it).
EXPAND_FIRES_SEED = 90     # first draw 0.182 -> fires; offset ~ 662 DN
EXPAND_SKIPS_SEED = 0      # first draw 0.496 -> skipped


def test_do_expand_adds_a_pedestal_and_saturates_highlights():
    """
    When the <0.3 branch fires, a constant offset lifts the whole frame and
    the brightest pixels clip at pix_max.

    That clipping is the point of the augmentation: without it the dataset
    contains no saturated highlights and the network never learns to handle
    them.
    """
    pm = pix_max_for(10)
    # Normalised signal spans [0, 0.5]*pix_max, so nothing clips on its own.
    img = torch.linspace(0.0, 0.5, 64).view(1, 64, 1).expand(4, 64, 64).contiguous()

    _, gt_plain = add_photon_noise(img, nbits=10, random_alpha=False,
                                   do_expand=False, norm_min=0.0, norm_max=1.0,
                                   generator=gen(EXPAND_FIRES_SEED))
    noisy, gt_exp = add_photon_noise(img, nbits=10, random_alpha=False,
                                     do_expand=True, norm_min=0.0, norm_max=1.0,
                                     generator=gen(EXPAND_FIRES_SEED))

    assert float(gt_plain.max()) == pytest.approx(0.5, abs=1e-3), \
        "baseline should not saturate"
    # A pedestal: the darkest pixel is lifted off black.
    offset_dn = float(gt_exp.min()) * pm
    assert offset_dn > 32.0 - 1e-3, f"no pedestal added (offset {offset_dn:.1f} DN)"
    # Highlights really saturate, but not the whole frame.
    sat = (gt_exp >= 1.0).float().mean()
    assert 0.0 < float(sat) < 1.0, \
        f"expected partial saturation, got fraction {float(sat):.3f}"
    assert float((noisy >= 1.0).float().mean()) > 0.0, \
        "noisy shows no saturated pixels"
    # The pedestal is a pure shift where nothing clipped.
    unclipped = gt_exp < 1.0
    shift = (gt_exp[unclipped] - gt_plain[unclipped]) * pm
    assert float(shift.std()) < 1e-2, "the offset is not spatially constant"


def test_do_expand_is_a_no_op_when_the_branch_does_not_fire():
    """
    With a draw >= 0.3 the clean signal must be byte-identical to do_expand=False.

    70% of samples take this path; a stray offset leaking in would brighten
    the entire dataset.
    """
    img = torch.linspace(0.0, 0.5, 64).view(1, 64, 1).expand(4, 64, 64).contiguous()
    _, gt_skip = add_photon_noise(img, nbits=10, random_alpha=False, do_expand=True,
                                  norm_min=0.0, norm_max=1.0,
                                  generator=gen(EXPAND_SKIPS_SEED))
    _, gt_ref = add_photon_noise(img, nbits=10, random_alpha=False, do_expand=False,
                                 norm_min=0.0, norm_max=1.0,
                                 generator=gen(EXPAND_SKIPS_SEED))
    assert torch.equal(gt_skip, gt_ref)
    assert float(gt_skip.min()) == 0.0, "a pedestal was added on the skip path"


def test_do_expand_offset_stays_in_its_documented_decade():
    """
    offset = 2**(U*6 + 5) must land in [32, 2048] DN, and the branch must fire
    on roughly 30% of calls.

    At 12 bit the pedestal is always recoverable as gt.min()*pix_max (2048 <
    4095), which makes this a direct read-out of the sampler.
    """
    pm = pix_max_for(12)
    img = torch.zeros(4, 8, 8)
    img[0, 0, 0] = 1.0            # keep the range non-degenerate, min stays 0
    g = gen(4321)
    offsets = []
    N = 400
    for _ in range(N):
        _, gt = add_photon_noise(img, nbits=12, random_alpha=False, do_expand=True,
                                 norm_min=0.0, norm_max=1.0, generator=g)
        o = float(gt.min()) * pm
        if o > 1e-3:
            offsets.append(o)
    fired = len(offsets) / N
    assert 0.22 < fired < 0.38, f"branch fired on {fired:.3f} of calls, expected ~0.30"
    o = torch.tensor(offsets)
    assert float(o.min()) >= 32.0 - 1.0, f"offset {float(o.min()):.1f} below 2**5"
    assert float(o.max()) <= 2048.0 + 1.0, f"offset {float(o.max()):.1f} above 2**11"
    # log2(offset) is uniform on [5, 11] -> mean ~ 8.
    assert float(torch.log2(o).mean()) == pytest.approx(8.0, abs=0.4)


@pytest.mark.xfail(reason="BUG: do_expand offset is drawn on an absolute 2**[5,11] DN "
                          "scale that ignores pix_max, so at nbits=10 (offset up to "
                          "2048 > pix_max=1023) ~5% of samples saturate the ENTIRE "
                          "frame to a constant 1.0 gt",
                   strict=False)
def test_do_expand_never_destroys_the_whole_frame():
    """
    Highlight expansion must clip *highlights*, leaving some unsaturated
    signal — a training pair whose gt is uniformly 1.0 everywhere carries no
    information at all and teaches the network to output white.

    The offset is sampled from 2**[5, 11] = [32, 2048] DN independently of
    nbits. At the default nbits=10 (pix_max=1023) the top ~17% of that decade
    exceeds full scale, so ~0.3 * 0.167 = 5% of do_expand samples come back as
    a flat white gt regardless of image content.
    """
    img = torch.linspace(0.0, 1.0, 32).view(1, 32, 1).expand(4, 32, 32).contiguous()
    g = gen(99)
    dead = []
    N = 300
    for i in range(N):
        _, gt = add_photon_noise(img, nbits=10, random_alpha=False, do_expand=True,
                                 norm_min=0.0, norm_max=1.0, generator=g)
        if float(gt.min()) >= 1.0:      # every pixel saturated
            dead.append(i)
    assert not dead, (
        f"{len(dead)}/{N} do_expand samples ({100.0 * len(dead) / N:.1f}%) came back "
        f"as a completely saturated, information-free gt (first at draw {dead[0] if dead else None})"
    )
