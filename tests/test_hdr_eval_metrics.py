"""
Image quality metrics — `hdr_eval.metrics`.

Why this file matters
─────────────────────
Two obligations:

* **Continuity with the existing numbers.** `psnr` and `ssim` here must
  agree with `test_dual_MoE_two_phase`'s implementations to floating-point
  precision, or every historical result becomes incomparable with every
  new one. Those equalities are asserted directly against the script.

* **The metrics must measure what they claim.** A metric that returns a
  plausible number for the wrong reason is worse than none: it makes a
  benchmark confidently wrong. So each is pinned by a property that only
  a correct implementation has — PSNR against its closed form, SSIM
  against a self-comparison, MS-SSIM against single-scale SSIM on a
  low-frequency signal, CIEDE2000 against the case it exists for (a hue
  error that PSNR cannot see).
"""

import math

import pytest
import torch

import test_dual_MoE_two_phase as script
from hdr_eval import metrics as M
from hdr_eval.tonemap import mu_law


@pytest.fixture
def pair():
    g = torch.Generator().manual_seed(4)
    return (torch.rand((2, 3, 64, 64), generator=g),
            torch.rand((2, 3, 64, 64), generator=g))


class TestAgreementWithTheExistingScript:
    def test_psnr_matches(self, pair):
        pred, gt = pair
        assert abs(M.psnr(pred, gt) - script.psnr(pred, gt)) < 1e-5

    def test_ssim_matches(self, pair):
        pred, gt = pair
        assert abs(M.ssim(pred, gt) - script.ssim(pred, gt)) < 1e-6

    def test_ssim_matches_on_a_realistic_pair(self, pair):
        """Random noise is an easy case; a near-match is the real one."""
        pred, gt = pair
        near = gt * 0.98 + 0.01
        assert abs(M.ssim(near, gt) - script.ssim(near, gt)) < 1e-6

    def test_psnr_matches_through_the_tone_curve(self, pair):
        pred, gt = pair
        a = M.psnr(mu_law(pred), mu_law(gt))
        b = script.psnr(script.hdr_tonemap(pred), script.hdr_tonemap(gt))
        assert abs(a - b) < 1e-5


class TestPixelMetrics:
    def test_psnr_matches_its_closed_form(self):
        pred = torch.full((1, 3, 8, 8), 0.5)
        gt = torch.full((1, 3, 8, 8), 0.6)
        expected = 10 * math.log10(1.0 / 0.01)
        assert abs(M.psnr(pred, gt) - expected) < 1e-4

    def test_identical_images_hit_the_ceiling_not_infinity(self, pair):
        """An infinite entry would poison every average it lands in."""
        pred, _ = pair
        assert M.psnr(pred, pred) == M.PSNR_CEILING_DB
        assert torch.isfinite(M.psnr_per_image(pred, pred)).all()

    def test_per_image_values_are_independent(self):
        pred = torch.stack([torch.full((3, 8, 8), 0.5),
                            torch.full((3, 8, 8), 0.9)])
        gt = torch.full((2, 3, 8, 8), 0.5)
        values = M.psnr_per_image(pred, gt)
        assert values[0] > values[1]

    def test_mse_and_mae_are_what_they_say(self):
        pred = torch.full((1, 3, 4, 4), 0.7)
        gt = torch.full((1, 3, 4, 4), 0.2)
        assert abs(float(M.mse_per_image(pred, gt)) - 0.25) < 1e-6
        assert abs(float(M.mae_per_image(pred, gt)) - 0.5) < 1e-6

    def test_data_range_scales_the_result(self, pair):
        pred, gt = pair
        assert M.psnr(pred, gt, data_range=2.0) > M.psnr(pred, gt)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            M.psnr(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 9))

    def test_unbatched_input_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            M.psnr(torch.rand(3, 8, 8), torch.rand(3, 8, 8))


class TestSSIM:
    def test_a_perfect_match_scores_one(self, pair):
        pred, _ = pair
        assert abs(M.ssim(pred, pred) - 1.0) < 1e-6

    def test_it_falls_as_the_images_diverge(self, pair):
        pred, gt = pair
        close = gt + 0.01 * torch.randn_like(gt)
        far = gt + 0.2 * torch.randn_like(gt)
        assert M.ssim(close, gt) > M.ssim(far, gt)

    def test_it_is_symmetric(self, pair):
        pred, gt = pair
        assert abs(M.ssim(pred, gt) - M.ssim(gt, pred)) < 1e-6

    def test_the_map_has_the_input_shape(self, pair):
        pred, gt = pair
        assert M.ssim_map(pred, gt).shape == pred.shape

    def test_valid_mode_drops_the_padded_border(self, pair):
        pred, gt = pair
        assert M.ssim_map(pred, gt, padding=False).shape[-1] == pred.shape[-1] - 10

    def test_padding_choice_changes_the_score(self, pair):
        """Documented difference; the default reproduces the old numbers."""
        pred, gt = pair
        padded = float(M.ssim_map(pred, gt).mean())
        valid = float(M.ssim_map(pred, gt, padding=False).mean())
        assert padded != valid


class TestMSSSIM:
    def test_a_perfect_match_scores_one(self):
        x = torch.rand(1, 3, 192, 192)
        assert abs(M.ms_ssim(x, x) - 1.0) < 1e-4

    def test_it_falls_as_the_images_diverge(self):
        gt = torch.rand(1, 3, 192, 192)
        close = gt * 0.99 + 0.005
        far = gt * 0.5 + 0.25
        assert M.ms_ssim(close, gt) > M.ms_ssim(far, gt)

    def test_it_is_more_forgiving_than_single_scale_on_fine_noise(self):
        """
        Multi-scale exists because fine texture differences dominate
        single-scale SSIM; on a smooth image with a small high-frequency
        perturbation MS-SSIM should score higher.
        """
        base = torch.linspace(0, 1, 192).view(1, 1, 1, 192).expand(1, 3, 192, 192)
        gt = base.contiguous()
        pred = gt + 0.02 * torch.randn_like(gt)
        assert M.ms_ssim(pred, gt) > M.ssim(pred, gt)

    def test_too_small_an_input_explains_the_requirement(self):
        small = torch.rand(1, 3, 64, 64)
        with pytest.raises(ValueError, match="at least 192x192"):
            M.ms_ssim(small, small)

    def test_fewer_scales_allow_smaller_inputs(self):
        small = torch.rand(1, 3, 64, 64)
        value = M.ms_ssim(small, small, weights=(0.5, 0.5))
        assert abs(value - 1.0) < 1e-3

    def test_empty_weights_are_rejected(self):
        x = torch.rand(1, 3, 192, 192)
        with pytest.raises(ValueError, match="weights"):
            M.ms_ssim(x, x, weights=())


class TestColour:
    def test_lab_of_white_and_black(self):
        white = M.rgb_to_lab(torch.ones(1, 3, 2, 2))[0, :, 0, 0]
        black = M.rgb_to_lab(torch.zeros(1, 3, 2, 2))[0, :, 0, 0]
        assert abs(float(white[0]) - 100.0) < 1e-3
        assert abs(float(white[1])) < 1e-3 and abs(float(white[2])) < 1e-3
        assert abs(float(black[0])) < 1e-3

    def test_lab_lightness_is_monotonic_in_luminance(self):
        greys = torch.tensor([0.1, 0.4, 0.9]).view(1, 3, 1, 1).expand(1, 3, 1, 3)
        grey_image = torch.stack([greys[0, i] for i in range(3)]).unsqueeze(0)
        lab = M.rgb_to_lab(grey_image)
        assert torch.all(lab[0, 0, 0, 1:] >= lab[0, 0, 0, :-1])

    def test_identical_images_have_zero_colour_difference(self, pair):
        pred, _ = pair
        assert float(M.delta_e_2000(pred, pred).max()) < 1e-4
        assert float(M.delta_e_76(pred, pred).max()) < 1e-4

    def test_it_catches_a_hue_error_psnr_cannot_see(self):
        """
        The reason this metric is here. A red-to-green swap and a small
        luminance change can score the same PSNR; only a colour metric
        separates them.
        """
        red = torch.zeros(1, 3, 8, 8); red[:, 0] = 0.5
        green = torch.zeros(1, 3, 8, 8); green[:, 1] = 0.5
        dimmer_red = torch.zeros(1, 3, 8, 8); dimmer_red[:, 0] = 0.42

        hue_error = float(M.delta_e_2000(red, green))
        luma_error = float(M.delta_e_2000(red, dimmer_red))
        assert hue_error > 10 * luma_error

    def test_it_is_finite_on_neutral_colours(self):
        """Hue is undefined for greys; the formula must not divide by zero."""
        grey = torch.full((1, 3, 8, 8), 0.5)
        other = torch.full((1, 3, 8, 8), 0.6)
        assert torch.isfinite(M.delta_e_2000(grey, other)).all()
        assert torch.isfinite(M.delta_e_2000(grey, grey)).all()

    def test_it_is_finite_at_the_extremes(self):
        black = torch.zeros(1, 3, 4, 4)
        white = torch.ones(1, 3, 4, 4)
        assert torch.isfinite(M.delta_e_2000(black, white)).all()

    def test_it_grows_with_the_error(self):
        gt = torch.rand(1, 3, 16, 16)
        near = (gt + 0.02).clamp(0, 1)
        far = (gt + 0.3).clamp(0, 1)
        assert float(M.delta_e_2000(far, gt)) > float(M.delta_e_2000(near, gt))

    def test_wrong_channel_count_is_rejected(self):
        with pytest.raises(ValueError, match="3 channels"):
            M.rgb_to_lab(torch.rand(1, 4, 8, 8))


class TestLPIPS:
    def test_availability_is_reported_not_assumed(self):
        metric = M.LPIPSMetric()
        assert isinstance(metric.available, bool)

    @pytest.mark.slow
    def test_a_perfect_match_scores_near_zero(self):
        metric = M.LPIPSMetric(tile=None)
        if not metric.available:
            pytest.skip("lpips is not installed")
        x = torch.rand(1, 3, 64, 64)
        assert float(metric(x, x).abs().max()) < 1e-4

    @pytest.mark.slow
    def test_it_grows_with_perceptual_distance(self):
        metric = M.LPIPSMetric(tile=None)
        if not metric.available:
            pytest.skip("lpips is not installed")
        gt = torch.rand(1, 3, 64, 64)
        near = (gt + 0.02).clamp(0, 1)
        far = torch.rand(1, 3, 64, 64)
        assert float(metric(near, gt)) < float(metric(far, gt))


class TestMetricSuite:
    def test_the_default_columns(self):
        suite = M.MetricSuite()
        assert suite.names == ["psnr_linear", "mae_linear", "ssim_linear",
                               "psnr_mu", "mae_mu", "ssim_mu", "delta_e2000"]

    def test_disabling_a_domain_drops_its_columns(self):
        suite = M.MetricSuite(M.MetricConfig(linear=False))
        assert not any(n.endswith("_linear") for n in suite.names)

    def test_it_returns_plain_floats_for_a_csv_row(self, pair):
        pred, gt = pair
        row = M.MetricSuite()(pred, gt)
        assert set(row) == set(M.MetricSuite().names)
        assert all(isinstance(v, float) for v in row.values())

    def test_the_columns_match_what_was_advertised(self, pair):
        pred, gt = pair
        suite = M.MetricSuite(M.MetricConfig(mae=False, delta_e=False))
        assert set(suite(pred, gt)) == set(suite.names)

    def test_tone_mapped_psnr_differs_from_linear(self, pair):
        pred, gt = pair
        row = M.MetricSuite()(pred, gt)
        assert row["psnr_linear"] != row["psnr_mu"]

    def test_inputs_are_clamped_not_rejected(self):
        """Model output can stray slightly outside [0, 1]; that is not fatal."""
        pred = torch.full((1, 3, 8, 8), 1.5)
        gt = torch.full((1, 3, 8, 8), 1.0)
        row = M.MetricSuite()(pred, gt)
        assert row["psnr_linear"] == M.PSNR_CEILING_DB

    def test_the_mu_constant_is_configurable_and_matters(self, pair):
        pred, gt = pair
        a = M.MetricSuite(M.MetricConfig(mu=5000.0))(pred, gt)["psnr_mu"]
        b = M.MetricSuite(M.MetricConfig(mu=100.0))(pred, gt)["psnr_mu"]
        assert a != b
