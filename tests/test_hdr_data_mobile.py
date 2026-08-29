"""
MobileHDR through the new base class — `hdr_data.mobile_hdr`.

Why this file matters
─────────────────────
`MobileHDRPacked` reads the same corpus as the original
`HDR_Mobile_dataset.MobileHDRDataset` and is meant to be a drop-in for it.
"Meant to be" is not good enough: every trained checkpoint was fitted to
the original's exact output, so the two classes are asserted here to
produce *bit-identical* tensors on the deterministic test split. If that
ever diverges, benchmark numbers for existing checkpoints stop meaning
what they say.

The rest covers the parts the original does not have — noise presets,
`without_gt`, centre-cropping, metadata — and the file discovery, whose
sort order matters: the per-index test noise seed is tied to position in
the list, so an unstable order would quietly re-pair images with noise.

The `dataset`-marked tests exercise the real corpus and skip when it is
not on disk.
"""

import os

import pytest
import torch

from HDR_Mobile_dataset import MobileHDRDataset
from hdr_data.mobile_hdr import MobileHDRPacked
from helpers import make_dataset


@pytest.fixture
def corpus(tmp_path):
    """A synthetic on-disk tree in the MobileHDR layout."""
    return make_dataset(tmp_path / "mobile", n_train=4, n_test=3, h=64, w=64)


class TestEquivalenceWithTheOriginal:
    def test_test_split_is_bit_identical(self, corpus):
        old = MobileHDRDataset(base_dir=corpus, split="test")
        new = MobileHDRPacked(corpus, split="test")
        assert len(old) == len(new)
        for i in range(len(old)):
            a, b = old[i], new[i]
            assert torch.equal(a["x"], b["x"]), f"noisy differs at {i}"
            assert torch.equal(a["y"], b["y"]), f"clean differs at {i}"

    def test_the_file_lists_agree(self, corpus):
        old = MobileHDRDataset(base_dir=corpus, split="train")
        new = MobileHDRPacked(corpus, split="train")
        assert old.file_list == new.file_list

    def test_train_lengths_agree(self, corpus):
        old = MobileHDRDataset(base_dir=corpus, split="train", num_patch=6)
        new = MobileHDRPacked(corpus, split="train", num_patch=6)
        assert len(old) == len(new)

    def test_the_sample_dict_has_the_same_keys(self, corpus):
        old = MobileHDRDataset(base_dir=corpus, split="test")
        new = MobileHDRPacked(corpus, split="test")
        assert set(old[0]) == set(new[0])


class TestDiscovery:
    def test_train_recurses_into_subdirectories(self, corpus):
        ds = MobileHDRPacked(corpus, split="train")
        parents = {os.path.basename(os.path.dirname(p)) for p in ds.entries}
        assert parents == {"static", "dynamic"}

    def test_the_file_list_is_sorted(self, corpus):
        ds = MobileHDRPacked(corpus, split="train")
        assert ds.entries == sorted(ds.entries)

    def test_missing_data_names_the_directory_it_looked_in(self, tmp_path):
        with pytest.raises(RuntimeError, match="tensors"):
            MobileHDRPacked(str(tmp_path), split="train")

    def test_unknown_test_subset_is_rejected(self, corpus):
        with pytest.raises(ValueError, match="test_subset"):
            MobileHDRPacked(corpus, split="test", test_subset="maybe")

    def test_without_gt_reads_its_own_directory(self, corpus, tmp_path):
        target = os.path.join(corpus, "test", "tensors", "without_gt")
        os.makedirs(target, exist_ok=True)
        torch.save(torch.rand(4, 64, 64), os.path.join(target, "a.pt"))
        ds = MobileHDRPacked(corpus, split="test", test_subset="without_gt")
        assert len(ds) == 1 and ds.tensor_dir.endswith("without_gt")

    def test_frame_shape_reports_the_source_dimensions(self, corpus):
        ds = MobileHDRPacked(corpus, split="train")
        assert ds.frame_shape(0) == (4, 64, 64)


class TestNewCapabilities:
    def test_a_noise_preset_changes_the_noise_level(self, corpus):
        low = MobileHDRPacked(corpus, split="test", noise="low")
        high = MobileHDRPacked(corpus, split="test", noise="extreme")
        low_res = float((low[0]["x"] - low[0]["y"]).std())
        high_res = float((high[0]["x"] - high[0]["y"]).std())
        assert high_res > low_res

    def test_centre_cropping_on_the_test_split(self, corpus):
        ds = MobileHDRPacked(corpus, split="test", crop_size=32)
        assert ds[0]["x"].shape == (4, 32, 32)

    def test_metadata_carries_the_source_id(self, corpus):
        ds = MobileHDRPacked(corpus, split="test", return_meta=True)
        assert ds[0]["source_id"] == "test_000"

    def test_mmap_can_be_disabled(self, corpus):
        ds = MobileHDRPacked(corpus, split="test", mmap=False)
        assert ds[0]["x"].shape == (4, 64, 64)

    def test_repr_mentions_the_split_and_source_count(self, corpus):
        ds = MobileHDRPacked(corpus, split="test")
        assert "sources=3" in repr(ds)


@pytest.mark.dataset
class TestAgainstTheRealCorpus:
    def test_the_real_test_split_matches_the_original(self, real_dataset_dir):
        old = MobileHDRDataset(base_dir=real_dataset_dir, split="test")
        new = MobileHDRPacked(real_dataset_dir, split="test", crop_size=64)
        assert len(old) == len(new)
        # Crop the new one for speed; compare the file lists exactly.
        assert old.file_list == new.entries

    @pytest.mark.slow
    def test_a_real_sample_is_finite_and_normalised(self, real_dataset_dir):
        ds = MobileHDRPacked(real_dataset_dir, split="test", crop_size=64,
                             return_meta=True)
        sample = ds[0]
        for key in ("x", "y"):
            assert torch.isfinite(sample[key]).all()
            assert 0.0 <= float(sample[key].min())
            assert float(sample[key].max()) <= 1.0
        assert sample["source_id"]
