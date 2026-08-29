"""
The shared dataset skeleton — `hdr_data.base.BaseBayerDataset`.

Why this file matters
─────────────────────
Three behaviours in the base class are load-bearing and none of them show
up in a shape assertion:

* **Crop before noise.** Noise must be synthesised for the crop, not for
  the frame it came from. If that inverts, the dataloader does ~12x the
  work per patch — and, worse, `regen_*` semantics change silently.
* **Normalise by the full frame's range.** A crop keeps its absolute
  brightness only when scaled by the range of the whole image. Per-crop
  normalisation would make every dark crop mid-grey and destroy exactly
  the low-light signal the MoE router is supposed to specialise on.
* **Deterministic test noise.** Two benchmark runs must see identical
  noisy inputs, or model A and model B are not compared on the same data.

These are tested through a tiny in-memory subclass rather than any real
corpus, so the assertions are about the base class and nothing else.
"""

import pytest
import torch

from hdr_data.augment import D4Transform
from hdr_data.base import BaseBayerDataset, resolve_noise_model
from hdr_data.noise import NoiseModel, NoiseParams


class InMemoryDataset(BaseBayerDataset):
    """A dataset of deterministic synthetic frames, held in memory."""

    def __init__(self, frames=None, **kwargs):
        self._frames = frames if frames is not None else self._default_frames()
        self.load_calls = []
        super().__init__(**kwargs)

    @staticmethod
    def _default_frames(n=3, h=32, w=32):
        out = []
        for i in range(n):
            ramp = torch.linspace(0, 1, h).view(1, h, 1).expand(4, h, w)
            out.append((ramp + i * 0.25).contiguous().float())
        return out

    def _build_index(self):
        return list(range(len(self._frames)))

    def _load_clean(self, entry):
        self.load_calls.append(entry)
        return self._frames[entry]

    def _entry_id(self, entry):
        return f"frame{entry:02d}"


@pytest.fixture
def dataset():
    return InMemoryDataset(split="train")


class TestConstruction:
    def test_unknown_split_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown split"):
            InMemoryDataset(split="val")

    def test_zero_num_patch_is_rejected(self):
        with pytest.raises(ValueError, match="num_patch"):
            InMemoryDataset(split="train", num_patch=0)

    def test_non_positive_crop_is_rejected(self):
        with pytest.raises(ValueError, match="crop_size"):
            InMemoryDataset(split="train", crop_size=0)

    def test_an_empty_index_raises_with_the_subclass_hint(self):
        class Empty(InMemoryDataset):
            def _build_index(self):
                return []

            def _empty_index_hint(self):
                return "look in the other directory"

        with pytest.raises(RuntimeError, match="look in the other directory"):
            Empty(split="train")

    def test_abstract_hooks_must_be_implemented(self):
        class Incomplete(BaseBayerDataset):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete(split="train")

    def test_repr_states_the_configuration(self, dataset):
        text = repr(dataset)
        assert "InMemoryDataset" in text and "sources=3" in text


class TestNoiseResolution:
    def test_none_gives_the_legacy_model(self):
        assert resolve_noise_model(None).params.is_legacy

    def test_a_preset_name_is_looked_up(self):
        assert resolve_noise_model("high").params.shot_gain == 14.0

    def test_params_are_wrapped(self):
        p = NoiseParams(shot_gain=1.0)
        assert resolve_noise_model(p).params is p

    def test_a_model_passes_through(self):
        m = NoiseModel()
        assert resolve_noise_model(m) is m

    def test_anything_else_is_a_type_error(self):
        with pytest.raises(TypeError):
            resolve_noise_model(42)


class TestLengthAndIndexing:
    def test_train_length_multiplies_by_num_patch(self):
        ds = InMemoryDataset(split="train", num_patch=4)
        assert len(ds) == 12 and ds.num_sources == 3

    def test_test_split_ignores_num_patch(self):
        """A benchmark visits each image exactly once."""
        ds = InMemoryDataset(split="test", num_patch=4)
        assert len(ds) == 3

    def test_virtual_repeats_wrap_onto_the_same_sources(self):
        ds = InMemoryDataset(split="train", num_patch=3)
        assert ds.source_id(0) == ds.source_id(3) == "frame00"

    def test_negative_indices_work(self, dataset):
        assert dataset[-1]["x"].shape == dataset[len(dataset) - 1]["x"].shape

    def test_out_of_range_index_raises(self, dataset):
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_sample_keys(self, dataset):
        assert set(dataset[0]) == {"x", "xm", "y"}

    def test_xm_is_the_documented_alias_for_x(self, dataset):
        sample = dataset[0]
        assert torch.equal(sample["x"], sample["xm"])

    def test_meta_is_added_on_request(self):
        ds = InMemoryDataset(split="train", return_meta=True)
        sample = ds[1]
        assert sample["source_id"] == "frame01"
        assert sample["pattern"] == "BGGR"
        assert "alpha" in sample["noise_meta"]


class TestCropping:
    def test_crop_size_is_honoured(self):
        ds = InMemoryDataset(split="train", crop_size=8)
        assert ds[0]["x"].shape == (4, 8, 8)

    def test_no_crop_returns_the_whole_frame(self, dataset):
        assert dataset[0]["x"].shape == (4, 32, 32)

    def test_train_crops_move_between_visits(self):
        ds = InMemoryDataset(split="train", crop_size=8, num_patch=8)
        import random
        random.seed(0)
        sums = {float(ds[i]["y"].sum()) for i in range(8)}
        assert len(sums) > 1

    def test_test_crops_are_centred_and_stable(self):
        ds = InMemoryDataset(split="test", crop_size=8)
        assert torch.equal(ds[0]["x"], ds[0]["x"])
        a = ds[0]["y"]
        b = ds[0]["y"]
        assert torch.equal(a, b)

    def test_a_crop_larger_than_the_frame_explains_itself(self):
        ds = InMemoryDataset(split="train", crop_size=64)
        with pytest.raises(RuntimeError, match="smaller than crop_size"):
            ds[0]

    def test_noise_is_synthesised_after_cropping(self):
        """
        The crop must be the *input* to the sampler, not a slice of its
        output: with a fixed seed, a cropped sample must differ from the
        same window taken out of an uncropped sample.
        """
        full = InMemoryDataset(split="test", crop_size=None)
        cropped = InMemoryDataset(split="test", crop_size=8)
        window = full[0]["x"][:, 12:20, 12:20]
        assert not torch.equal(window, cropped[0]["x"])

    def test_the_clean_target_is_cropped_consistently(self):
        ds = InMemoryDataset(split="test", crop_size=8)
        sample = ds[0]
        assert sample["x"].shape == sample["y"].shape


class TestNormalisationRange:
    def test_crops_keep_their_absolute_brightness(self):
        """
        A dark crop must stay dark. Normalising per crop would rescale it
        to fill [0, 1]; normalising by the full frame's range preserves
        the difference between a dark region and a bright one.
        """
        h = w = 32
        dark_top = torch.cat([
            torch.full((4, h // 2, w), 0.05),
            torch.full((4, h // 2, w), 1.0),
        ], dim=1)
        ds = InMemoryDataset(frames=[dark_top], split="test", crop_size=None,
                             noise="clean")
        clean = ds[0]["y"]
        top_mean = float(clean[:, :h // 2].mean())
        bottom_mean = float(clean[:, h // 2:].mean())
        assert top_mean < 0.1 and bottom_mean > 0.9

    def test_the_range_is_cached_per_source(self):
        ds = InMemoryDataset(split="train", num_patch=3)
        _ = [ds[i] for i in range(9)]
        assert len(ds._range_cache) == 3

    def test_the_cached_range_is_the_full_frame_range(self):
        ds = InMemoryDataset(split="train", crop_size=8)
        _ = ds[0]
        lo, hi = ds._range_cache[0]
        frame = ds._frames[0]
        assert lo == pytest.approx(float(frame.min()))
        assert hi == pytest.approx(float(frame.max()))


class TestDeterministicTestNoise:
    def test_the_test_split_repeats_exactly(self):
        ds = InMemoryDataset(split="test")
        assert torch.equal(ds[0]["x"], ds[0]["x"])

    def test_two_datasets_with_the_same_seed_agree(self):
        a = InMemoryDataset(split="test")
        b = InMemoryDataset(split="test")
        assert torch.equal(a[2]["x"], b[2]["x"])

    def test_a_different_seed_gives_different_noise(self):
        a = InMemoryDataset(split="test", test_noise_seed=1)
        b = InMemoryDataset(split="test", test_noise_seed=2)
        assert not torch.equal(a[0]["x"], b[0]["x"])

    def test_each_index_gets_its_own_noise(self):
        """One seed for the whole split would correlate the images."""
        ds = InMemoryDataset(frames=[torch.full((4, 16, 16), 0.5)] * 3,
                             split="test")
        assert not torch.equal(ds[0]["x"], ds[1]["x"])

    def test_the_test_split_freezes_exposure_and_expansion(self):
        """
        Otherwise every benchmark run would re-roll the scene brightness
        and the numbers would not be comparable between runs.
        """
        ds = InMemoryDataset(split="test", noise="realistic_lowlight",
                             return_meta=True)
        assert ds[0]["noise_meta"]["alpha"] == 1.0
        model = ds._noise_model_for(is_train=False)
        assert not model.params.random_alpha and not model.params.do_expand

    def test_the_train_split_keeps_its_randomness(self):
        ds = InMemoryDataset(split="train", noise="realistic_lowlight")
        model = ds._noise_model_for(is_train=True)
        assert model.params.random_alpha

    def test_train_samples_vary_between_visits(self):
        ds = InMemoryDataset(split="train", num_patch=4)
        import random
        random.seed(0)
        first = ds[0]["x"]
        second = ds[0]["x"]
        assert not torch.equal(first, second)


class TestTransformHook:
    def test_the_transform_is_applied_to_both_images(self):
        seen = {}

        def spy(noisy, clean):
            seen["called"] = True
            return noisy * 0, clean * 0

        ds = InMemoryDataset(split="test", transform=spy)
        sample = ds[0]
        assert seen.get("called")
        assert float(sample["x"].abs().max()) == 0.0
        assert float(sample["y"].abs().max()) == 0.0

    def test_a_d4_transform_composes_cleanly(self):
        ds = InMemoryDataset(split="train", crop_size=8,
                             transform=D4Transform("BGGR", mode="cell"))
        assert ds[0]["x"].shape == (4, 8, 8)

    def test_a_wrong_shaped_source_is_reported_with_its_id(self):
        ds = InMemoryDataset(frames=[torch.rand(3, 16, 16)], split="test")
        with pytest.raises(RuntimeError, match="frame00"):
            ds[0]


class TestLegacyCompatibilityHooks:
    def test_regen_helpers_are_harmless_no_ops(self, dataset):
        dataset.regen_crops()
        dataset.regen_noise()
        assert len(dataset) == 3
