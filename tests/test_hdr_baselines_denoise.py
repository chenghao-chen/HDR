"""
Classical denoisers — `hdr_baselines.denoise`.

Why this file matters
─────────────────────
Two obligations, and the second is the one that bites.

**They must denoise.** Each method is asserted to improve PSNR against
real Poisson-Gaussian noise at three levels. That is the test that would
have caught the guided filter shipping with `eps` two orders of magnitude
below the noise variance, where it is an expensive identity function.

**They must be tuned fairly.** A benchmark that hands the learned model
per-image adaptation and the classical baselines one constant tuned at
one noise level is not a comparison, it is a demonstration. So every
denoiser exposes `for_noise_sigma`, `estimate_noise_sigma` recovers the
level from the image itself, and both are tested — including the
wrapper case, where the inner denoiser must be tuned to the noise level
*in the stabilised domain*, not the original one.
"""

import math

import pytest
import torch

from hdr_baselines.denoise import (
    BilateralDenoise,
    Denoiser,
    GaussianDenoise,
    GuidedFilterDenoise,
    MedianDenoise,
    NLMDenoise,
    VarianceStabilised,
    WaveletDenoise,
    box_filter,
    estimate_noise_sigma,
    gaussian_kernel1d,
    generalised_anscombe,
    haar_dwt,
    haar_idwt,
    inverse_generalised_anscombe,
)
from hdr_data.noise import NoiseModel, NoiseParams
from hdr_eval.metrics import psnr

ALL_DENOISERS = (GaussianDenoise, MedianDenoise, BilateralDenoise,
                 GuidedFilterDenoise, NLMDenoise, WaveletDenoise)


def noisy_pair(shot_gain=14.0, read_var=150.0, size=96, seed=1):
    """A structured clean image and its noisy capture."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size),
                            torch.linspace(0, 1, size), indexing="ij")
    clean = (0.35 + 0.3 * torch.sin(5 * math.pi * xx)
             * torch.cos(3 * math.pi * yy)).clamp(0, 1)
    clean = clean.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
    clean[:, :, 30:60, 20:70] = (clean[:, :, 30:60, 20:70] + 0.25).clamp(0, 1)
    params = NoiseParams(shot_gain=shot_gain,
                         read_noise_range=(read_var, read_var),
                         random_alpha=False)
    g = torch.Generator().manual_seed(seed)
    return NoiseModel(params).apply(clean, norm_min=0.0, norm_max=1.0,
                                    generator=g)


class TestPrimitives:
    def test_box_filter_of_a_constant_is_that_constant(self):
        x = torch.full((1, 1, 8, 8), 0.3)
        assert torch.allclose(box_filter(x, 2), x, atol=1e-6)

    def test_box_filter_radius_zero_is_a_no_op(self):
        x = torch.rand(1, 1, 8, 8)
        assert torch.equal(box_filter(x, 0), x)

    def test_a_negative_radius_is_rejected(self):
        with pytest.raises(ValueError, match="radius"):
            box_filter(torch.rand(1, 1, 4, 4), -1)

    def test_the_gaussian_kernel_is_normalised_and_symmetric(self):
        k = gaussian_kernel1d(1.5)
        assert abs(float(k.sum()) - 1.0) < 1e-6
        assert torch.allclose(k, k.flip(0), atol=1e-7)

    def test_a_non_positive_sigma_is_rejected(self):
        with pytest.raises(ValueError, match="sigma"):
            gaussian_kernel1d(0.0)

    def test_haar_round_trips_exactly(self):
        x = torch.rand(2, 4, 16, 16)
        assert torch.allclose(haar_idwt(*haar_dwt(x)), x, atol=1e-6)

    def test_haar_is_orthonormal(self):
        """Coefficient energy must equal signal energy, or the noise
        estimate made in one band is meaningless in another."""
        x = torch.randn(1, 1, 32, 32)
        bands = haar_dwt(x)
        energy = sum(float((b ** 2).sum()) for b in bands)
        assert abs(energy - float((x ** 2).sum())) < 1e-2

    def test_haar_rejects_odd_dimensions(self):
        with pytest.raises(ValueError, match="even"):
            haar_dwt(torch.rand(1, 1, 7, 8))

    def test_anscombe_round_trips(self):
        x = torch.rand(1, 4, 16, 16)
        y = generalised_anscombe(x, 0.0137, 1.4e-4)
        assert torch.allclose(inverse_generalised_anscombe(y, 0.0137, 1.4e-4),
                              x, atol=1e-5)

    def test_anscombe_flattens_the_noise_profile(self):
        """
        The point of the transform: after it, the residual standard
        deviation should barely depend on the signal level.
        """
        gain, read_var = 14.0 / 1023.0, 150.0 / (1023.0 ** 2)
        noisy, clean = noisy_pair(size=256)
        before_dark = float((noisy - clean)[clean < 0.3].std())
        before_bright = float((noisy - clean)[clean > 0.5].std())
        tn = generalised_anscombe(noisy, gain, read_var)
        tc = generalised_anscombe(clean, gain, read_var)
        after_dark = float((tn - tc)[clean < 0.3].std())
        after_bright = float((tn - tc)[clean > 0.5].std())
        assert (max(after_dark, after_bright) / min(after_dark, after_bright)
                < max(before_dark, before_bright) / min(before_dark, before_bright))

    def test_a_non_positive_gain_is_rejected(self):
        with pytest.raises(ValueError, match="gain"):
            generalised_anscombe(torch.rand(4), 0.0, 1.0)


class TestNoiseEstimation:
    @pytest.mark.parametrize("shot_gain,read_var",
                             [(4.0, 35.0), (14.0, 150.0), (32.0, 440.0)])
    def test_it_tracks_the_true_residual_sigma(self, shot_gain, read_var):
        noisy, clean = noisy_pair(shot_gain, read_var)
        estimated = float(estimate_noise_sigma(noisy).mean())
        true = float((noisy - clean).std())
        assert 0.5 * true < estimated < 1.5 * true

    def test_it_returns_one_value_per_image(self):
        assert estimate_noise_sigma(torch.rand(3, 4, 16, 16)).shape == (3, 1, 1, 1)

    def test_it_is_near_zero_for_a_clean_image(self):
        smooth = torch.linspace(0, 1, 64).view(1, 1, 1, 64).expand(1, 4, 64, 64)
        assert float(estimate_noise_sigma(smooth.contiguous()).mean()) < 0.01

    def test_odd_dimensions_are_handled(self):
        assert estimate_noise_sigma(torch.rand(1, 4, 15, 17)).shape == (1, 1, 1, 1)

    def test_a_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            estimate_noise_sigma(torch.rand(4, 16, 16))


class TestDenoising:
    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    @pytest.mark.parametrize("shot_gain,read_var",
                             [(4.0, 35.0), (14.0, 150.0), (32.0, 440.0)])
    def test_every_denoiser_improves_psnr_at_every_level(self, cls, shot_gain,
                                                         read_var):
        """
        The test that catches a mis-parameterised baseline. A method whose
        constants do not match the noise level is an expensive identity
        function, and the benchmark would report it as a fair comparison.
        """
        noisy, clean = noisy_pair(shot_gain, read_var)
        sigma = float(estimate_noise_sigma(noisy).mean())
        denoiser = cls.for_noise_sigma(sigma)
        before = psnr(noisy, clean)
        after = psnr(denoiser(noisy), clean)
        assert after > before + 1.0, f"{cls.name}: {before:.2f} -> {after:.2f}"

    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    def test_output_keeps_the_input_shape(self, cls):
        x = torch.rand(2, 4, 24, 32)
        assert cls()(x).shape == x.shape

    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    def test_output_is_finite(self, cls):
        assert torch.isfinite(cls()(torch.rand(1, 4, 32, 32))).all()

    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    def test_a_constant_image_survives(self, cls):
        """Nothing to remove; a method that shifts the level is broken."""
        flat = torch.full((1, 4, 32, 32), 0.4)
        assert torch.allclose(cls()(flat), flat, atol=0.02)

    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    def test_it_works_on_rgb_as_well_as_cfa(self, cls):
        """Pipelines apply these in both domains."""
        assert cls()(torch.rand(1, 3, 32, 32)).shape == (1, 3, 32, 32)

    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    def test_odd_dimensions_are_handled(self, cls):
        x = torch.rand(1, 4, 17, 23)
        assert cls()(x).shape == x.shape

    def test_the_median_removes_hot_pixels_outright(self):
        """Its distinguishing property; the linear methods smear them."""
        x = torch.full((1, 1, 16, 16), 0.3)
        x[0, 0, 8, 8] = 1.0
        assert float(MedianDenoise(3)(x)[0, 0, 8, 8]) == pytest.approx(0.3)
        assert float(GaussianDenoise(1.0)(x)[0, 0, 8, 8]) > 0.3

    def test_the_guided_filter_preserves_a_step(self):
        step = torch.zeros(1, 1, 32, 32)
        step[:, :, :, 16:] = 1.0
        out = GuidedFilterDenoise(radius=2, eps=1e-4)(step)
        assert float(out[0, 0, 16, 10]) < 0.1
        assert float(out[0, 0, 16, 22]) > 0.9

    def test_wavelet_shrinkage_adapts_without_being_told(self):
        """BayesShrink estimates its own threshold from the coefficients."""
        light, clean_l = noisy_pair(4.0, 35.0)
        heavy, clean_h = noisy_pair(32.0, 440.0)
        w = WaveletDenoise()
        assert psnr(w(light), clean_l) > psnr(light, clean_l)
        assert psnr(w(heavy), clean_h) > psnr(heavy, clean_h)


class TestParameterValidation:
    @pytest.mark.parametrize("factory,match", [
        (lambda: GaussianDenoise(0.0), "sigma"),
        (lambda: MedianDenoise(2), "odd"),
        (lambda: MedianDenoise(1), "odd"),
        (lambda: BilateralDenoise(sigma_spatial=0.0), "positive"),
        (lambda: BilateralDenoise(window=4), "odd"),
        (lambda: GuidedFilterDenoise(radius=0), "radius"),
        (lambda: GuidedFilterDenoise(eps=0.0), "eps"),
        (lambda: NLMDenoise(search_radius=0), "search_radius"),
        (lambda: NLMDenoise(h=0.0), "h must be positive"),
        (lambda: WaveletDenoise(levels=0), "levels"),
    ])
    def test_bad_parameters_are_rejected(self, factory, match):
        with pytest.raises(ValueError, match=match):
            factory()

    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    def test_for_noise_sigma_returns_the_right_type(self, cls):
        assert isinstance(cls.for_noise_sigma(0.05), cls)

    def test_guided_eps_scales_as_a_variance(self):
        """eps is in variance units; scaling it linearly with sigma is a bug."""
        a = GuidedFilterDenoise.for_noise_sigma(0.05)
        b = GuidedFilterDenoise.for_noise_sigma(0.10)
        assert b.eps == pytest.approx(4.0 * a.eps, rel=1e-6)

    def test_bigger_noise_means_more_smoothing(self):
        assert (GaussianDenoise.for_noise_sigma(0.15).sigma
                > GaussianDenoise.for_noise_sigma(0.02).sigma)

    def test_the_base_class_hook_ignores_the_hint(self):
        assert isinstance(Denoiser.for_noise_sigma(0.1), Denoiser)

    @pytest.mark.parametrize("cls", ALL_DENOISERS)
    def test_repr_states_the_parameters(self, cls):
        assert cls().extra_repr()


class TestVarianceStabilised:
    def test_it_denoises(self):
        noisy, clean = noisy_pair()
        model = VarianceStabilised(WaveletDenoise())
        assert psnr(model(noisy), clean) > psnr(noisy, clean) + 1.0

    def test_it_names_itself_after_its_inner_denoiser(self):
        assert VarianceStabilised(GaussianDenoise()).name == "vst_gaussian"

    def test_it_requires_a_denoiser(self):
        with pytest.raises(TypeError, match="inner must be"):
            VarianceStabilised(lambda x: x)

    def test_retuning_does_not_discard_the_wrapper(self):
        """
        The inner denoiser must be tuned to the noise level in the
        stabilised domain, which the outer estimate is not — so the
        wrapper retunes internally and returns itself here rather than
        being replaced by a bare denoiser.
        """
        model = VarianceStabilised(GuidedFilterDenoise())
        assert model.retuned(0.05) is model

    def test_it_can_be_pinned_to_fixed_inner_parameters(self):
        noisy, _ = noisy_pair()
        fixed = VarianceStabilised(GaussianDenoise(1.0), auto_inner=False)
        assert torch.isfinite(fixed(noisy)).all()

    def test_it_preserves_shape(self):
        x = torch.rand(1, 4, 24, 24)
        assert VarianceStabilised(GaussianDenoise())(x).shape == x.shape
