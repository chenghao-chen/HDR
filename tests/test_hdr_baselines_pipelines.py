"""
Classical pipelines — `hdr_baselines.pipelines`.

Why this file matters
─────────────────────
"Denoise then demosaic, or demosaic then denoise?" is a real question with
a real answer, and this class is what lets the benchmark answer it with
numbers instead of assertion. Both orders are therefore constructible and
both are tested — including the property that makes the answer come out:
demosaicing correlates the noise across pixels and channels, so a
white-noise denoiser applied afterwards has less to work with.

`auto_sigma` is the other load-bearing piece. Without it, one set of
constants has to serve every exposure in a test set, and the classical
baselines look worse than they are — which would flatter the learned model
for an entirely bogus reason.
"""

import math

import pytest
import torch

from hdr_baselines.denoise import (
    GaussianDenoise,
    GuidedFilterDenoise,
    WaveletDenoise,
)
from hdr_baselines.pipelines import DEMOSAIC_FUNCTIONS, ClassicalPipeline
from hdr_data.bayer import rgb_to_packed
from hdr_data.noise import NoiseModel, NoiseParams
from hdr_eval.metrics import psnr
from hdr_eval.tonemap import mu_law


@pytest.fixture
def scene():
    """A clean RGB scene, its CFA capture, and a noisy version of that."""
    h = w = 96
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w),
                            indexing="ij")
    rgb = torch.stack([
        0.4 + 0.3 * torch.sin(4 * math.pi * xx),
        0.45 + 0.3 * torch.cos(3 * math.pi * yy),
        0.4 + 0.25 * torch.sin(3 * math.pi * (xx + yy)),
    ]).unsqueeze(0).clamp(0, 1)
    rgb[:, :, 30:60, 20:70] = torch.tensor([0.8, 0.3, 0.2]).view(1, 3, 1, 1)

    packed = rgb_to_packed(rgb, "BGGR")
    params = NoiseParams(shot_gain=14.0, read_noise_range=(150.0, 150.0),
                         random_alpha=False)
    g = torch.Generator().manual_seed(3)
    noisy, _ = NoiseModel(params).apply(packed, norm_min=0.0, norm_max=1.0,
                                        generator=g)
    return rgb, noisy


class TestConstruction:
    def test_a_demosaic_only_pipeline_names_itself(self):
        assert ClassicalPipeline(None, "malvar").name == "malvar"

    def test_a_denoise_first_pipeline_names_itself(self):
        model = ClassicalPipeline(WaveletDenoise(), "gbtf")
        assert model.name == "wavelet+gbtf"

    def test_a_demosaic_first_pipeline_is_marked_post(self):
        model = ClassicalPipeline(WaveletDenoise(), "gbtf",
                                  order="demosaic_first")
        assert model.name == "wavelet+gbtf_post"

    def test_an_explicit_name_wins(self):
        assert ClassicalPipeline(None, "malvar", name="mine").name == "mine"

    def test_an_unknown_demosaicer_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown demosaic"):
            ClassicalPipeline(None, "magic")

    def test_an_unknown_order_is_rejected(self):
        with pytest.raises(ValueError, match="order must be"):
            ClassicalPipeline(None, "malvar", order="sideways")

    def test_a_non_denoiser_is_rejected(self):
        with pytest.raises(TypeError, match="denoiser must be"):
            ClassicalPipeline(lambda x: x, "malvar")

    def test_gbtf_with_a_foreign_pattern_is_rejected(self):
        with pytest.raises(ValueError, match="BGGR-only"):
            ClassicalPipeline(None, "gbtf", pattern="RGGB")

    def test_repr_states_the_composition(self):
        text = repr(ClassicalPipeline(WaveletDenoise(), "gbtf"))
        assert "wavelet" in text and "gbtf" in text


class TestBehaviour:
    def test_the_shapes_match_the_project_contract(self, scene):
        _, noisy = scene
        blended, experts, gates = ClassicalPipeline(GaussianDenoise(),
                                                    "malvar")(noisy, None)
        assert blended.shape == (1, 3, 96, 96)   # sensor res: 2x packed 48
        assert experts.shape[1] == 1 and gates.shape[1] == 1

    def test_denoising_helps(self, scene):
        rgb, noisy = scene
        plain = ClassicalPipeline(None, "gbtf")(noisy, None)[0]
        denoised = ClassicalPipeline(GaussianDenoise(), "gbtf")(noisy, None)[0]
        assert psnr(mu_law(denoised), mu_law(rgb)) > \
               psnr(mu_law(plain), mu_law(rgb)) + 1.0

    def test_denoising_first_beats_denoising_after(self, scene):
        """
        Demosaicing correlates the noise across pixels and channels, so a
        white-noise denoiser applied afterwards has less to work with.
        This is the comparison the class exists to make.
        """
        rgb, noisy = scene
        first = ClassicalPipeline(GaussianDenoise(), "malvar",
                                  order="denoise_first")(noisy, None)[0]
        after = ClassicalPipeline(GaussianDenoise(), "malvar",
                                  order="demosaic_first")(noisy, None)[0]
        assert psnr(mu_law(first), mu_law(rgb)) > psnr(mu_law(after), mu_law(rgb))

    @pytest.mark.parametrize("demosaic", sorted(DEMOSAIC_FUNCTIONS))
    def test_every_demosaicer_composes(self, scene, demosaic):
        _, noisy = scene
        out = ClassicalPipeline(WaveletDenoise(), demosaic)(noisy, None)[0]
        assert out.shape == (1, 3, 96, 96) and torch.isfinite(out).all()

    def test_output_stays_in_range(self, scene):
        _, noisy = scene
        out = ClassicalPipeline(GuidedFilterDenoise(), "gbtf")(noisy, None)[0]
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


class TestAutoSigma:
    def test_it_rescues_a_mis_tuned_denoiser(self, scene):
        """
        The guided filter with a default eps far below the noise variance
        is an identity function. Auto-tuning is what makes it a real
        baseline rather than a straw man.
        """
        rgb, noisy = scene
        fixed = ClassicalPipeline(GuidedFilterDenoise(radius=2, eps=1e-6),
                                  "gbtf", auto_sigma=False)(noisy, None)[0]
        auto = ClassicalPipeline(GuidedFilterDenoise(radius=2, eps=1e-6),
                                 "gbtf", auto_sigma=True)(noisy, None)[0]
        assert psnr(mu_law(auto), mu_law(rgb)) > \
               psnr(mu_law(fixed), mu_law(rgb)) + 1.0

    def test_it_adapts_across_noise_levels(self):
        """One constant cannot serve every exposure in a test set."""
        h = w = 64
        clean = torch.full((1, 4, h, w), 0.5)
        clean[:, :, :, 32:] = 0.8
        results = []
        for shot_gain, read_var in ((2.0, 20.0), (32.0, 440.0)):
            params = NoiseParams(shot_gain=shot_gain,
                                 read_noise_range=(read_var, read_var),
                                 random_alpha=False)
            g = torch.Generator().manual_seed(0)
            noisy, gt = NoiseModel(params).apply(clean, norm_min=0.0,
                                                 norm_max=1.0, generator=g)
            model = ClassicalPipeline(GaussianDenoise(), "bilinear")
            denoised = model._denoise(noisy)
            results.append(psnr(denoised, gt) - psnr(noisy, gt))
        assert all(gain > 0 for gain in results)

    def test_it_leaves_a_clean_image_alone(self):
        """A degenerate sigma estimate must not produce a degenerate filter."""
        flat = torch.full((1, 4, 32, 32), 0.4)
        model = ClassicalPipeline(GaussianDenoise(), "bilinear",
                                  auto_sigma=True)
        assert torch.allclose(model._denoise(flat), flat, atol=1e-3)

    def test_a_pipeline_without_a_denoiser_ignores_it(self, scene):
        _, noisy = scene
        model = ClassicalPipeline(None, "malvar", auto_sigma=True)
        assert torch.equal(model._denoise(noisy), noisy)
