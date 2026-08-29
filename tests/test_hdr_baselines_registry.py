"""
Baselines by name — `hdr_baselines.registry`.

Why this file matters
─────────────────────
`--baselines wavelet+gbtf_post,vst_guided+malvar,unet` has to become three
constructed models, and the combinations are generated rather than
enumerated — so the parser is the thing that decides whether a typo
becomes a clear error or a silently different model. Every accepted form
is exercised here, and every rejected one is checked to name the
alternatives rather than raise a bare KeyError.
"""

import pytest
import torch

from hdr_baselines.denoise import Denoiser, VarianceStabilised
from hdr_baselines.interface import BaselineModel
from hdr_baselines.pipelines import ClassicalPipeline
from hdr_baselines.registry import (
    DEFAULT_LINEUP,
    build_baseline,
    build_denoiser,
    describe_baselines,
    list_baselines,
    list_demosaicers,
    list_denoisers,
)


class TestListings:
    def test_the_listings_are_sorted(self):
        assert list_denoisers() == sorted(list_denoisers())
        assert list_demosaicers() == sorted(list_demosaicers())
        assert list_baselines() == sorted(list_baselines())

    def test_the_description_covers_every_accepted_form(self):
        text = describe_baselines()
        for fragment in ("Demosaic only", "_post", "vst_", "Learned"):
            assert fragment in text

    def test_the_default_lineup_is_all_buildable(self):
        for name in DEFAULT_LINEUP:
            assert isinstance(build_baseline(name), BaselineModel)


class TestDenoiserNames:
    @pytest.mark.parametrize("name", ["gaussian", "median", "bilateral",
                                      "guided", "nlm", "wavelet"])
    def test_every_registered_denoiser_builds(self, name):
        assert isinstance(build_denoiser(name), Denoiser)

    @pytest.mark.parametrize("name", ["vst_gaussian", "vst_wavelet",
                                      "vst_guided"])
    def test_the_vst_prefix_wraps(self, name):
        model = build_denoiser(name)
        assert isinstance(model, VarianceStabilised)
        assert model.name == name

    def test_an_unknown_denoiser_names_the_alternatives(self):
        with pytest.raises(KeyError, match="wavelet"):
            build_denoiser("magic")

    def test_an_unknown_vst_inner_is_reported(self):
        with pytest.raises(KeyError, match="Unknown denoiser"):
            build_denoiser("vst_magic")


class TestBaselineNames:
    @pytest.mark.parametrize("name", ["nearest", "bilinear", "malvar", "gbtf"])
    def test_a_bare_demosaicer_builds(self, name):
        model = build_baseline(name)
        assert model.name == name and not model.trainable

    @pytest.mark.parametrize("name", ["gaussian+malvar", "wavelet+gbtf",
                                      "nlm+bilinear", "vst_wavelet+gbtf"])
    def test_a_combination_builds_in_denoise_first_order(self, name):
        model = build_baseline(name)
        assert isinstance(model, ClassicalPipeline)
        assert model.order == "denoise_first" and model.name == name

    @pytest.mark.parametrize("name", ["gaussian+malvar_post",
                                      "wavelet+gbtf_post"])
    def test_the_post_suffix_selects_the_other_order(self, name):
        model = build_baseline(name)
        assert model.order == "demosaic_first" and model.name == name

    @pytest.mark.parametrize("name", ["dncnn", "unet", "demosaicnet",
                                      "restormer_lite"])
    def test_a_learned_architecture_builds(self, name):
        model = build_baseline(name)
        assert model.trainable and model.num_parameters > 0

    def test_learned_kwargs_are_forwarded(self):
        small = build_baseline("dncnn", width=8, depth=3)
        large = build_baseline("dncnn", width=16, depth=3)
        assert large.num_parameters > small.num_parameters

    def test_the_cfa_pattern_is_forwarded(self):
        assert build_baseline("malvar", pattern="RGGB").pattern == "RGGB"

    def test_auto_sigma_is_forwarded(self):
        assert build_baseline("wavelet+malvar", auto_sigma=False).auto_sigma is False


class TestErrors:
    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_baseline("")

    def test_an_unknown_name_prints_the_catalogue(self):
        with pytest.raises(KeyError, match="Demosaic only"):
            build_baseline("magic")

    def test_an_unknown_demosaicer_in_a_combination_is_reported(self):
        with pytest.raises(KeyError, match="Unknown demosaicer"):
            build_baseline("gaussian+magic")

    def test_an_unknown_denoiser_in_a_combination_is_reported(self):
        with pytest.raises(KeyError, match="Unknown denoiser"):
            build_baseline("magic+malvar")

    def test_gbtf_with_a_foreign_pattern_is_rejected(self):
        with pytest.raises(ValueError, match="BGGR"):
            build_baseline("gbtf", pattern="GRBG")


class TestEveryBuiltModelRuns:
    @pytest.mark.parametrize("name", [
        "nearest", "bilinear", "malvar", "gbtf",
        "gaussian+gbtf", "median+malvar", "bilateral+bilinear",
        "guided+malvar", "wavelet+gbtf", "nlm+malvar",
        "wavelet+gbtf_post", "vst_wavelet+gbtf", "vst_guided+malvar",
    ])
    def test_it_produces_the_contracted_output(self, name):
        model = build_baseline(name)
        model.eval()
        x = torch.rand(1, 4, 16, 16)
        snr = torch.rand(1, 1, 16, 16)
        with torch.no_grad():
            blended, experts, gates = model(x, snr)
        assert blended.shape == (1, 3, 32, 32)
        assert experts.shape == (1, 1, 3, 32, 32)
        assert gates.shape == (1, 1, 32, 32)
        assert torch.isfinite(blended).all()
        assert float(blended.min()) >= 0.0 and float(blended.max()) <= 1.0

    @pytest.mark.parametrize("name", ["dncnn", "unet", "demosaicnet",
                                      "restormer_lite"])
    def test_learned_models_produce_the_contracted_output(self, name):
        model = build_baseline(name)
        model.eval()
        with torch.no_grad():
            out = model(torch.rand(1, 4, 16, 16), torch.rand(1, 1, 16, 16))[0]
        assert out.shape == (1, 3, 32, 32) and torch.isfinite(out).all()
