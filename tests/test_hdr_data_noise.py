"""
Sensor noise synthesis and calibration — `hdr_data.noise`.

Why this file matters
─────────────────────
The noise model *is* the training distribution. Two things must hold and
neither is visible from a shape check:

1. **Bit-exactness with the original.** Every released checkpoint was
   trained on `HDR_Mobile_dataset.add_photon_noise`. If the new model
   diverges from it — a reordered random draw is enough — then the
   benchmark evaluates those checkpoints on data they were never trained
   for, and every comparison against them is wrong by an unknown amount.
   So the legacy preset is asserted equal to the original tensor for
   tensor, under a shared generator seed, across the argument
   combinations the dataset actually uses.

2. **Determinism.** The test split seeds a per-index generator so two
   benchmark runs see identical inputs. If that breaks, model A and model
   B are scored on different noise and the comparison is noise, too.

The rest covers the additions (per-channel gain, PRNU, banding, hot
pixels) — each asserted to do the specific thing it claims and to be OFF
by default — and the calibration path, which is checked by recovering
parameters the sampler was given.
"""

import math

import pytest
import torch

from HDR_Mobile_dataset import add_photon_noise
from hdr_data.noise import (
    NOISE_PRESETS,
    NoiseModel,
    NoiseParams,
    calibrate_from_pairs,
    get_preset,
    list_presets,
    poisson_gaussian,
    sigma_from_params,
    snr_db,
)


@pytest.fixture
def hdr_image():
    """An unnormalised 4-channel HDR tensor with a real dynamic range."""
    g = torch.Generator().manual_seed(11)
    ramp = torch.linspace(0.0, 3.0, 64).view(1, 64, 1).expand(4, 64, 64)
    return (ramp + torch.rand((4, 64, 64), generator=g) * 0.5).contiguous()


def gen(seed=7):
    return torch.Generator().manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# The equivalence that protects existing checkpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestLegacyEquivalence:
    @pytest.mark.parametrize("random_alpha", [True, False])
    @pytest.mark.parametrize("do_expand", [True, False])
    def test_bit_exact_against_add_photon_noise(self, hdr_image,
                                                random_alpha, do_expand):
        old_n, old_c = add_photon_noise(
            hdr_image, random_alpha=random_alpha, do_expand=do_expand,
            generator=gen())
        model = NoiseModel.legacy(random_alpha=random_alpha,
                                  do_expand=do_expand)
        new_n, new_c = model.apply(hdr_image, generator=gen())
        assert torch.equal(old_n, new_n)
        assert torch.equal(old_c, new_c)

    def test_bit_exact_with_an_explicit_normalisation_range(self, hdr_image):
        """The crop path passes the full frame's range; that must match too."""
        old_n, old_c = add_photon_noise(hdr_image, norm_min=0.0, norm_max=4.0,
                                        generator=gen())
        new_n, new_c = NoiseModel.legacy().apply(
            hdr_image, norm_min=0.0, norm_max=4.0, generator=gen())
        assert torch.equal(old_n, new_n) and torch.equal(old_c, new_c)

    @pytest.mark.parametrize("shot_gain,read_range",
                             [(1.0, (4.0, 6.0)), (32.0, (400.0, 480.0))])
    def test_bit_exact_at_other_noise_levels(self, hdr_image, shot_gain,
                                             read_range):
        old_n, old_c = add_photon_noise(
            hdr_image, shot_gain=shot_gain, read_noise_range=read_range,
            generator=gen())
        new_n, new_c = NoiseModel.legacy(
            shot_gain=shot_gain, read_noise_range=read_range
        ).apply(hdr_image, generator=gen())
        assert torch.equal(old_n, new_n) and torch.equal(old_c, new_c)

    def test_the_default_preset_is_the_legacy_one(self):
        assert NoiseParams().is_legacy
        assert get_preset("mobile_hdr").is_legacy
        assert get_preset("legacy") == get_preset("mobile_hdr")


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_the_same_seed_gives_the_same_tensors(self, hdr_image):
        model = NoiseModel.from_preset("realistic")
        a = model.apply(hdr_image, generator=gen(3))
        b = model.apply(hdr_image, generator=gen(3))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_different_seeds_give_different_tensors(self, hdr_image):
        model = NoiseModel.from_preset("realistic")
        a = model.apply(hdr_image, generator=gen(3))
        b = model.apply(hdr_image, generator=gen(4))
        assert not torch.equal(a[0], b[0])

    def test_no_generator_still_produces_valid_output(self, hdr_image):
        noisy, clean = NoiseModel().apply(hdr_image)
        assert torch.isfinite(noisy).all() and torch.isfinite(clean).all()


# ─────────────────────────────────────────────────────────────────────────────
# Parameter validation
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterValidation:
    @pytest.mark.parametrize("kwargs,match", [
        ({"nbits": 0}, "nbits"),
        ({"shot_gain": -1.0}, "shot_gain"),
        ({"read_noise_range": (10.0, 1.0)}, "read_noise_range"),
        ({"alpha_mode": "gaussian"}, "alpha_mode"),
        ({"alpha_range": (0.5, 0.2)}, "alpha_range"),
        ({"alpha_mode": "log_uniform", "alpha_range": (0.0, 1.0)}, "log_uniform"),
        ({"expand_prob": 1.5}, "expand_prob"),
        ({"hot_pixel_rate": 2.0}, "hot_pixel_rate"),
        ({"prnu_std": -0.1}, "prnu_std"),
    ])
    def test_bad_parameters_are_rejected_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            NoiseParams(**kwargs)

    def test_pix_max_follows_the_bit_depth(self):
        assert NoiseParams(nbits=10).pix_max == 1023.0
        assert NoiseParams(nbits=12).pix_max == 4095.0

    def test_replace_returns_a_modified_copy(self):
        p = NoiseParams()
        q = p.replace(shot_gain=1.0)
        assert q.shot_gain == 1.0 and p.shot_gain == 14.0

    def test_to_dict_is_json_friendly(self):
        import json
        json.dumps(NoiseParams().to_dict())

    def test_model_rejects_a_non_params_argument(self):
        with pytest.raises(TypeError):
            NoiseModel({"shot_gain": 14})


# ─────────────────────────────────────────────────────────────────────────────
# Output contract
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputContract:
    @pytest.mark.parametrize("name", sorted(NOISE_PRESETS))
    def test_every_preset_produces_normalised_finite_output(self, hdr_image, name):
        noisy, clean = NoiseModel.from_preset(name).apply(
            hdr_image, generator=gen())
        for t, label in ((noisy, "noisy"), (clean, "clean")):
            assert torch.isfinite(t).all(), f"{name}: {label} not finite"
            assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0

    def test_shape_is_preserved(self, hdr_image):
        noisy, _ = NoiseModel().apply(hdr_image, generator=gen())
        assert noisy.shape == hdr_image.shape

    def test_quantisation_puts_values_on_the_dn_grid(self, hdr_image):
        noisy, _ = NoiseModel(NoiseParams(nbits=8)).apply(
            hdr_image, generator=gen())
        on_grid = (noisy * 255.0).round() - (noisy * 255.0)
        assert float(on_grid.abs().max()) < 1e-4

    def test_disabling_quantisation_leaves_values_off_the_grid(self, hdr_image):
        p = NoiseParams(nbits=8, quantize=False)
        noisy, _ = NoiseModel(p).apply(hdr_image, generator=gen())
        on_grid = (noisy * 255.0).round() - (noisy * 255.0)
        assert float(on_grid.abs().max()) > 1e-3

    def test_a_constant_image_degenerates_to_zero(self):
        """The zero-range guard: no range means no signal to scale."""
        flat = torch.full((4, 8, 8), 2.5)
        noisy, clean = NoiseModel(
            NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                        random_alpha=False)).apply(flat, generator=gen())
        assert float(clean.abs().max()) == 0.0

    def test_meta_reports_the_realised_draws(self, hdr_image):
        noisy, clean, meta = NoiseModel().apply(
            hdr_image, generator=gen(), return_meta=True)
        assert set(meta) == {"alpha", "read_var", "pix_max", "norm_min", "norm_max"}
        assert 0.0 < meta["alpha"] <= 1.0
        assert 135.0 <= meta["read_var"] <= 160.0

    def test_call_is_an_alias_for_apply(self, hdr_image):
        model = NoiseModel()
        a = model(hdr_image, generator=gen())
        b = model.apply(hdr_image, generator=gen())
        assert torch.equal(a[0], b[0])


# ─────────────────────────────────────────────────────────────────────────────
# The additions, each isolated
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseComponents:
    def test_noise_grows_with_the_signal(self, hdr_image):
        """The defining property of shot noise; a constant-sigma model fails this."""
        params = NoiseParams(shot_gain=14.0, read_noise_range=(150.0, 150.0),
                             random_alpha=False)
        ramp = torch.linspace(0, 1, 256).view(1, 256, 1).expand(4, 256, 256)
        noisy, clean = NoiseModel(params).apply(
            ramp.contiguous(), norm_min=0.0, norm_max=1.0, generator=gen())
        residual = noisy - clean
        dark = float(residual[:, :64].std())
        bright = float(residual[:, 128:192].std())
        assert bright > 1.5 * dark

    def test_channel_gain_scales_each_channel(self):
        flat = torch.linspace(0, 1, 32).view(1, 32, 1).expand(4, 32, 32).contiguous()
        base = NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                           random_alpha=False, quantize=False)
        _, plain = NoiseModel(base).apply(flat, norm_min=0., norm_max=1.,
                                          generator=gen())
        gains = (0.5, 1.0, 1.0, 0.25)
        _, gained = NoiseModel(base.replace(channel_gain=gains)).apply(
            flat, norm_min=0., norm_max=1., generator=gen())
        for c, g in enumerate(gains):
            assert torch.allclose(gained[c], plain[c] * g, atol=1e-5)

    def test_channel_gain_length_must_match(self, hdr_image):
        params = NoiseParams(channel_gain=(1.0, 1.0))
        with pytest.raises(ValueError, match="channel_gain"):
            NoiseModel(params).apply(hdr_image, generator=gen())

    def test_row_banding_is_constant_along_a_row(self):
        """Banding must be structured, or it is just more white noise."""
        flat = torch.full((4, 32, 32), 0.5)
        params = NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                             random_alpha=False, row_fpn_std=20.0,
                             quantize=False)
        noisy, clean = NoiseModel(params).apply(
            flat, norm_min=0.0, norm_max=1.0, generator=gen())
        # Every pixel in a row shares one offset, so the row-wise variance
        # is zero while the column-wise variance is not.
        per_row = noisy[0].std(dim=1)
        assert float(per_row.max()) < 1e-6
        assert float(noisy[0].mean(dim=1).std()) > 1e-3

    def test_column_banding_is_constant_along_a_column(self):
        flat = torch.full((4, 32, 32), 0.5)
        params = NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                             random_alpha=False, col_fpn_std=20.0,
                             quantize=False)
        noisy, _ = NoiseModel(params).apply(flat, norm_min=0.0, norm_max=1.0,
                                            generator=gen())
        assert float(noisy[0].std(dim=0).max()) < 1e-6

    def test_hot_pixels_are_sparse_and_saturated(self):
        flat = torch.full((4, 128, 128), 0.3)
        params = NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                             random_alpha=False, hot_pixel_rate=0.01)
        noisy, _ = NoiseModel(params).apply(flat, norm_min=0.0, norm_max=1.0,
                                            generator=gen())
        hot = (noisy >= 0.999).float().mean()
        assert 0.002 < float(hot) < 0.03

    def test_prnu_is_multiplicative_and_zero_mean(self):
        flat = torch.full((4, 64, 64), 0.6)
        params = NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                             random_alpha=False, prnu_std=0.05, quantize=False)
        noisy, clean = NoiseModel(params).apply(
            flat, norm_min=0.0, norm_max=1.0, generator=gen())
        ratio = clean / 0.6
        assert abs(float(ratio.mean()) - 1.0) < 0.01
        assert 0.03 < float(ratio.std()) < 0.07

    def test_every_addition_is_off_by_default(self):
        p = NoiseParams()
        assert p.channel_gain is None
        assert p.prnu_std == p.row_fpn_std == p.col_fpn_std == 0.0
        assert p.hot_pixel_rate == 0.0

    def test_alpha_modes_respect_their_range(self):
        p = NoiseParams(alpha_mode="uniform", alpha_range=(0.2, 0.4))
        model = NoiseModel(p)
        draws = [model.sample_alpha(gen(i)) for i in range(50)]
        assert all(0.2 <= a <= 0.4 for a in draws)

    def test_log_uniform_alpha_favours_the_dark_end(self):
        p = NoiseParams(alpha_mode="log_uniform", alpha_range=(0.01, 1.0))
        model = NoiseModel(p)
        draws = [model.sample_alpha(gen(i)) for i in range(200)]
        assert sum(a < 0.1 for a in draws) > len(draws) * 0.4

    def test_fixed_alpha_mode_is_constant(self):
        p = NoiseParams(alpha_mode="fixed", alpha_range=(0.0, 0.7))
        model = NoiseModel(p)
        assert {round(model.sample_alpha(gen(i)), 6) for i in range(10)} == {0.7}

    def test_clip_high_can_be_disabled(self):
        big = torch.linspace(0, 1, 32).view(1, 32, 1).expand(4, 32, 32).contiguous()
        p = NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                        random_alpha=False, clip_high=False, do_expand=False)
        noisy, _ = NoiseModel(p).apply(big, norm_min=0.0, norm_max=0.5,
                                       generator=gen())
        assert float(noisy.max()) > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Analytic helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticHelpers:
    def test_sigma_follows_the_model(self):
        p = NoiseParams(shot_gain=14.0, read_noise_range=(150.0, 150.0))
        signal = torch.tensor([0.0, 100.0, 1023.0])
        sigma = sigma_from_params(signal, p)
        expected = torch.sqrt(14.0 * signal + 150.0)
        assert torch.allclose(sigma, expected)

    def test_sigma_uses_the_midpoint_read_variance_by_default(self):
        p = NoiseParams(shot_gain=0.0, read_noise_range=(100.0, 200.0))
        assert abs(float(sigma_from_params(torch.zeros(1), p))
                   - math.sqrt(150.0)) < 1e-5

    def test_snr_db_rises_with_signal(self):
        p = NoiseParams(shot_gain=14.0, read_noise_range=(150.0, 150.0))
        values = snr_db(torch.tensor([10.0, 100.0, 1000.0]), p)
        assert values[0] < values[1] < values[2]

    def test_poisson_gaussian_matches_its_own_sigma(self):
        clean = torch.full((512, 512), 400.0)
        out = poisson_gaussian(clean, 14.0, 150.0, gen())
        expected = math.sqrt(14.0 * 400.0 + 150.0)
        assert abs(float((out - clean).std()) - expected) / expected < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibration:
    @pytest.mark.parametrize("shot_gain,read_var",
                             [(4.0, 35.0), (14.0, 150.0), (32.0, 440.0)])
    def test_recovers_the_parameters_it_was_given(self, shot_gain, read_var):
        params = NoiseParams(shot_gain=shot_gain,
                             read_noise_range=(read_var, read_var),
                             random_alpha=False)
        ramp = torch.linspace(0, 1, 512).view(1, 512, 1).expand(4, 512, 512)
        noisy, clean = NoiseModel(params).apply(
            ramp.contiguous(), norm_min=0.0, norm_max=1.0, generator=gen())
        fit = calibrate_from_pairs(clean, noisy, nbits=10, num_bins=48)
        assert abs(fit["shot_gain"] - shot_gain) / shot_gain < 0.05
        assert abs(fit["read_var"] - read_var) / read_var < 0.20
        assert fit["r2"] > 0.95

    def test_clipped_bins_are_excluded(self):
        """Without the exclusion the fit is dragged badly off; with it, not."""
        params = NoiseParams(shot_gain=14.0, read_noise_range=(150.0, 150.0),
                             random_alpha=False)
        ramp = torch.linspace(0, 1, 512).view(1, 512, 1).expand(4, 512, 512)
        noisy, clean = NoiseModel(params).apply(
            ramp.contiguous(), norm_min=0.0, norm_max=1.0, generator=gen())
        good = calibrate_from_pairs(clean, noisy, num_bins=48)
        naive = calibrate_from_pairs(clean, noisy, num_bins=48, max_clip_frac=1.0)
        assert good["num_bins_clipped"] > 0
        assert abs(good["shot_gain"] - 14.0) < abs(naive["shot_gain"] - 14.0)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            calibrate_from_pairs(torch.rand(4, 8, 8), torch.rand(4, 8, 9))

    def test_a_constant_signal_cannot_be_calibrated(self):
        flat = torch.full((4, 64, 64), 0.5)
        with pytest.raises(ValueError, match="constant"):
            calibrate_from_pairs(flat, flat + 0.01)

    def test_too_few_usable_bins_explains_itself(self):
        ramp = torch.linspace(0, 1, 8).view(1, 8, 1).expand(1, 8, 8).contiguous()
        with pytest.raises(ValueError, match="usable bins"):
            calibrate_from_pairs(ramp, ramp + 0.001, num_bins=32,
                                 min_count=1000)


class TestPresets:
    def test_list_presets_is_sorted_and_complete(self):
        assert list_presets() == tuple(sorted(NOISE_PRESETS))

    def test_unknown_preset_lists_the_alternatives(self):
        with pytest.raises(KeyError, match="mobile_hdr"):
            get_preset("ultra")

    def test_graded_presets_are_monotonically_noisier(self, hdr_image):
        residuals = []
        for name in ("low", "medium", "high", "extreme"):
            noisy, clean = NoiseModel.from_preset(name).apply(
                hdr_image, norm_min=0.0, norm_max=4.0, generator=gen())
            residuals.append(float((noisy - clean).std()))
        assert residuals == sorted(residuals)

    def test_clean_preset_is_essentially_noise_free(self, hdr_image):
        noisy, clean = NoiseModel.from_preset("clean").apply(
            hdr_image, norm_min=0.0, norm_max=4.0, generator=gen())
        assert float((noisy - clean).abs().max()) <= 1.0 / 1023.0 + 1e-6

    def test_realistic_presets_enable_the_structured_terms(self):
        for name in ("realistic", "realistic_lowlight"):
            p = get_preset(name)
            assert not p.is_legacy
            assert p.channel_gain is not None and p.row_fpn_std > 0
