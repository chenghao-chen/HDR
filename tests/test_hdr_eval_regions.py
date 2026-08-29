"""
Per-region stratification — `hdr_eval.regions`.

Why this file matters
─────────────────────
A whole-frame PSNR on an HDR image is a highlight metric wearing a
disguise: squared error is dominated by the brightest pixels, so a model
can gain 2 dB overall while getting worse everywhere a low-light denoiser
is supposed to work. These masks are what make the failure visible, so
their correctness is the difference between a benchmark that reports the
problem and one that hides it.

The masks must (a) partition the image, (b) be derived from the ground
truth so two models are compared on identical pixels, and (c) report
honestly when a band is empty — a metric over zero pixels is undefined,
and silently returning 0 dB would read as catastrophic failure while
returning 100 dB would read as success.
"""

import pytest
import torch

from hdr_eval import regions as R


@pytest.fixture
def gt():
    g = torch.Generator().manual_seed(2)
    return torch.rand((2, 3, 32, 32), generator=g)


class TestLuminance:
    def test_it_uses_rec709_weights(self):
        rgb = torch.zeros(1, 3, 1, 1)
        rgb[:, 1] = 1.0
        assert abs(float(R.luminance(rgb)) - 0.7152) < 1e-6

    def test_green_dominates_red_dominates_blue(self):
        out = []
        for c in range(3):
            rgb = torch.zeros(1, 3, 1, 1)
            rgb[:, c] = 1.0
            out.append(float(R.luminance(rgb)))
        assert out[1] > out[0] > out[2]

    def test_it_returns_one_channel(self, gt):
        assert R.luminance(gt).shape == (2, 1, 32, 32)

    def test_wrong_channel_count_is_rejected(self):
        with pytest.raises(ValueError, match="3 colour channels"):
            R.luminance(torch.rand(1, 4, 8, 8))


class TestBands:
    def test_luminance_bands_partition_the_image(self, gt):
        masks = R.luminance_bands(gt)
        total = sum(m.int() for m in masks.values())
        assert torch.equal(total, torch.ones_like(total))

    def test_the_default_edges_isolate_the_deep_shadows(self):
        """Below 2% of full scale gets its own band — where low light lives."""
        assert R.DEFAULT_LUMINANCE_EDGES[1] == 0.02

    def test_band_names_carry_their_bounds(self, gt):
        names = list(R.luminance_bands(gt))
        assert names[0].startswith("lum[0,")

    def test_snr_bands_partition_the_map(self):
        snr = torch.rand(1, 1, 16, 16)
        masks = R.snr_bands(snr)
        total = sum(m.int() for m in masks.values())
        assert torch.equal(total, torch.ones_like(total))

    def test_the_snr_edges_include_the_routers_own_threshold(self):
        """The test script reports "% pixels with SNR < 0.5"; align with it."""
        assert 0.5 in R.DEFAULT_SNR_EDGES

    def test_a_multichannel_snr_map_is_rejected(self):
        with pytest.raises(ValueError, match="single-channel"):
            R.snr_bands(torch.rand(1, 3, 8, 8))

    def test_non_increasing_edges_are_rejected(self, gt):
        with pytest.raises(ValueError, match="strictly increasing"):
            R.luminance_bands(gt, edges=(0.0, 0.5, 0.2))

    def test_a_single_edge_is_rejected(self, gt):
        with pytest.raises(ValueError, match="at least two edges"):
            R.luminance_bands(gt, edges=(0.5,))

    def test_quantile_bands_are_equally_populated(self, gt):
        masks = R.quantile_bands(gt, num_bands=4)
        counts = [float(m.float().mean()) for m in masks.values()]
        assert all(abs(c - 0.25) < 0.02 for c in counts)

    def test_quantile_bands_partition_the_image(self, gt):
        masks = R.quantile_bands(gt, num_bands=4)
        total = sum(m.int() for m in masks.values())
        assert torch.equal(total, torch.ones_like(total))

    def test_too_few_quantile_bands_is_rejected(self, gt):
        with pytest.raises(ValueError, match="num_bands"):
            R.quantile_bands(gt, num_bands=1)


class TestSpecialMasks:
    def test_saturation_finds_clipped_pixels(self):
        gt = torch.zeros(1, 3, 4, 4)
        gt[:, 0, 0, 0] = 1.0
        mask = R.saturation_mask(gt)
        assert bool(mask[0, 0, 0, 0]) and float(mask.float().sum()) == 1.0

    def test_saturation_triggers_on_any_channel(self):
        gt = torch.zeros(1, 3, 2, 2)
        gt[:, 2] = 1.0
        assert float(R.saturation_mask(gt).float().mean()) == 1.0

    def test_a_bad_saturation_threshold_is_rejected(self, gt):
        with pytest.raises(ValueError, match="threshold"):
            R.saturation_mask(gt, threshold=0.0)

    def test_edges_land_on_the_edge(self):
        """A single step: the mask must sit on it, not in the flat regions."""
        gt = torch.zeros(1, 3, 32, 32)
        gt[:, :, :, 16:] = 1.0
        mask = R.edge_mask(gt, quantile=0.95)[0, 0]
        assert bool(mask[:, 14:18].any())
        assert not bool(mask[:, :10].any())

    def test_edge_coverage_matches_the_quantile(self, gt):
        mask = R.edge_mask(gt, quantile=0.9)
        assert 0.05 < float(mask.float().mean()) < 0.15

    def test_a_bad_edge_quantile_is_rejected(self, gt):
        with pytest.raises(ValueError, match="quantile"):
            R.edge_mask(gt, quantile=1.0)


class TestMaskedMetrics:
    def test_a_full_mask_matches_the_unmasked_metric(self, gt):
        pred = gt * 0.9
        full = torch.ones(2, 1, 32, 32, dtype=torch.bool)
        assert torch.allclose(R.masked_psnr(pred, gt, full),
                              R.masked_psnr(pred, gt, None), atol=1e-5)

    def test_masking_isolates_the_selected_pixels(self):
        gt = torch.full((1, 3, 8, 8), 0.5)
        pred = gt.clone()
        pred[:, :, :4] = 0.9                      # error only in the top half
        top = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        top[:, :, :4] = True
        assert float(R.masked_psnr(pred, gt, top)) < float(
            R.masked_psnr(pred, gt, ~top))

    def test_an_empty_mask_reports_nan_not_a_number(self):
        """
        The honest answer. Zero would read as total failure and the PSNR
        ceiling would read as perfection; both would be fabricated.
        """
        gt = torch.rand(2, 3, 8, 8)
        empty = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
        assert torch.isnan(R.masked_mse(gt, gt, empty)).all()
        assert torch.isnan(R.masked_psnr(gt, gt, empty)).all()

    def test_masked_mae_matches_a_hand_computation(self):
        gt = torch.zeros(1, 3, 4, 4)
        pred = torch.full((1, 3, 4, 4), 0.4)
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[:, :, 0, 0] = True
        assert abs(float(R.masked_mae(pred, gt, mask)) - 0.4) < 1e-6

    def test_a_rank_mismatch_is_rejected(self, gt):
        with pytest.raises(ValueError, match="same rank"):
            R.masked_mse(gt, gt, torch.ones(32, 32, dtype=torch.bool))


class TestStratify:
    def test_it_returns_one_value_and_one_coverage_per_band(self, gt):
        pred = gt * 0.9
        out = R.stratify(pred, gt, R.luminance_bands(gt))
        bands = list(R.luminance_bands(gt))
        for band in bands:
            assert band in out and f"{band}.coverage" in out

    def test_coverage_sums_to_one_over_a_partition(self, gt):
        pred = gt * 0.9
        out = R.stratify(pred, gt, R.luminance_bands(gt))
        total = sum(v for k, v in out.items() if k.endswith(".coverage"))
        assert abs(total - 1.0) < 1e-5

    def test_coverage_can_be_suppressed(self, gt):
        out = R.stratify(gt, gt, R.luminance_bands(gt), include_coverage=False)
        assert not any(k.endswith(".coverage") for k in out)

    def test_it_accepts_any_masked_metric(self, gt):
        out = R.stratify(gt * 0.9, gt, R.luminance_bands(gt), fn=R.masked_mae)
        assert all(isinstance(v, float) for v in out.values())

    def test_values_are_plain_floats_for_a_csv_row(self, gt):
        out = R.stratify(gt * 0.9, gt, R.luminance_bands(gt))
        assert all(isinstance(v, float) for v in out.values())
