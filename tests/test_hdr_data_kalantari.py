"""
Kalantari 2017 bracketed scenes — `hdr_data.kalantari`.

Why this file matters
─────────────────────
Two things about this loader are worth pinning:

* **The imagery is not in this checkout.** `datasets/kalantari2017` holds
  74 train and 15 test scene *directories*, all empty. So every test here
  builds synthetic scenes in the documented layout — the one
  `convert_to_tfrecord.py` reads — and the loader is verified against
  those. Nothing here has seen the real files, and the empty-skeleton case
  is tested explicitly so the error a user hits is the informative one.

* **Exposures are EV stops, not times.** The dataset's files hold stops;
  a merge that treats them as seconds is off by orders of magnitude and
  produces a plausible-looking, entirely wrong HDR image. The conversion
  and its heuristic for already-converted files are tested directly.
"""

import os

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2", reason="OpenCV is needed to write fixtures")

from hdr_data.kalantari import (  # noqa: E402
    KalantariDataset,
    ldr_to_linear,
    merge_exposures,
    read_exposures,
    read_hdr,
    read_ldr,
)


def write_scene(directory, h=64, w=96, seed=0, style="train"):
    """One synthetic scene in the layout the reference converter expects."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    hdr = (rng.random((h, w, 3)) * 4.0).astype(np.float32)

    hdr_name = "ref_hdr_aligned.hdr" if style == "train" else "HDRImg.hdr"
    exp_name = "input_exp.txt" if style == "train" else "exposure.txt"
    cv2.imwrite(os.path.join(directory, hdr_name), hdr[:, :, ::-1])

    evs = [-2.0, 0.0, 2.0]
    for j, ev in enumerate(evs):
        ldr = np.clip(hdr * (2.0 ** ev) / 4.0, 0, 1) ** (1 / 2.2)
        name = (f"input_{j + 1}_aligned.tif" if style == "train"
                else f"{j + 1}.tif")
        cv2.imwrite(os.path.join(directory, name),
                    (ldr[:, :, ::-1] * 255).astype(np.uint8))
    with open(os.path.join(directory, exp_name), "w") as fh:
        fh.write("\n".join(str(e) for e in evs) + "\n")
    return hdr


@pytest.fixture
def scenes(tmp_path):
    """A root with two train scenes and one test scene."""
    root = tmp_path / "kal"
    for i in range(2):
        write_scene(root / "train" / f"scene_{i:02d}", seed=i)
    write_scene(root / "test" / "001", seed=9, style="test")
    return str(root)


class TestExposureFiles:
    def test_ev_stops_become_times(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("-2\n0\n2\n")
            path = fh.name
        assert read_exposures(path) == [0.25, 1.0, 4.0]
        os.unlink(path)

    def test_values_that_are_already_times_are_left_alone(self, tmp_path):
        path = tmp_path / "e.txt"
        path.write_text("100\n1000\n10000\n")
        assert read_exposures(str(path)) == [100.0, 1000.0, 10000.0]

    def test_a_short_file_is_reported(self, tmp_path):
        path = tmp_path / "e.txt"
        path.write_text("0\n1\n")
        with pytest.raises(ValueError, match="expected 3"):
            read_exposures(str(path), num_expected=3)

    def test_an_empty_file_is_reported(self, tmp_path):
        path = tmp_path / "e.txt"
        path.write_text("\n\n")
        with pytest.raises(ValueError, match="empty"):
            read_exposures(str(path))

    def test_non_numeric_content_is_reported(self, tmp_path):
        path = tmp_path / "e.txt"
        path.write_text("dark\nbright\n")
        with pytest.raises(ValueError, match="Non-numeric"):
            read_exposures(str(path))


class TestReaders:
    def test_hdr_reads_as_linear_rgb(self, tmp_path):
        d = tmp_path / "s"
        expected = write_scene(d)
        got = read_hdr(str(d / "ref_hdr_aligned.hdr"))
        assert got.shape == (3, 64, 96)
        # Radiance .hdr is RGBE: one 8-bit exponent SHARED by all three
        # channels plus an 8-bit mantissa each. So the quantisation step
        # of every channel is set by the *brightest* channel of that
        # pixel, and a dim channel beside a bright one loses a lot of
        # relative precision. The right tolerance is therefore per-pixel
        # and absolute, scaled by that pixel's maximum — anything tighter
        # would be testing the file format rather than the reader.
        want = torch.from_numpy(expected).permute(2, 0, 1)
        step = want.amax(dim=0, keepdim=True) / 256.0
        assert torch.all((got - want).abs() <= 2.0 * step + 1e-6)

    def test_hdr_preserves_the_channel_order(self, tmp_path):
        """OpenCV reads BGR; a missed swap would silently transpose R and B."""
        d = tmp_path / "s"
        os.makedirs(d, exist_ok=True)
        image = np.zeros((8, 8, 3), np.float32)
        image[:, :, 0] = 3.0          # red plane, in RGB order
        cv2.imwrite(str(d / "x.hdr"), image[:, :, ::-1])
        got = read_hdr(str(d / "x.hdr"))
        assert float(got[0].mean()) > 2.5
        assert float(got[1].mean()) < 0.1 and float(got[2].mean()) < 0.1

    def test_ldr_is_normalised_by_its_bit_depth(self, tmp_path):
        d = tmp_path / "s"
        write_scene(d)
        got = read_ldr(str(d / "input_1_aligned.tif"))
        assert got.shape == (3, 64, 96)
        assert 0.0 <= float(got.min()) and float(got.max()) <= 1.0

    def test_a_missing_file_is_reported_by_name(self, tmp_path):
        with pytest.raises(Exception, match="nope"):
            read_hdr(str(tmp_path / "nope.hdr"))


class TestExposureArithmetic:
    def test_linearisation_undoes_gamma_and_exposure(self):
        ldr = torch.full((3, 4, 4), 0.5)
        out = ldr_to_linear(ldr, exposure_time=2.0, gamma=2.2)
        assert torch.allclose(out, torch.full((3, 4, 4), 0.5 ** 2.2 / 2.0))

    def test_a_non_positive_exposure_is_rejected(self):
        with pytest.raises(ValueError, match="exposure_time"):
            ldr_to_linear(torch.rand(3, 4, 4), 0.0)

    def test_merging_recovers_a_known_radiance(self):
        """Well-exposed pixels in every frame must merge back to the truth."""
        radiance = torch.full((3, 8, 8), 0.3)
        times = [0.5, 1.0, 2.0]
        ldrs = [(radiance * t).clamp(0, 1) ** (1 / 2.2) for t in times]
        merged = merge_exposures(ldrs, times)
        assert torch.allclose(merged, radiance, atol=1e-2)

    def test_clipped_exposures_are_excluded_from_the_merge(self):
        """A blown-out frame must not drag the estimate down."""
        radiance = torch.full((3, 8, 8), 0.4)
        times = [1.0, 100.0]
        ldrs = [(radiance * times[0]).clamp(0, 1) ** (1 / 2.2),
                torch.ones(3, 8, 8)]              # fully saturated
        merged = merge_exposures(ldrs, times)
        assert torch.allclose(merged, radiance, atol=5e-2)

    def test_pixels_no_exposure_covers_fall_back_to_the_closest(self):
        """Every pixel must get a value, not a divide-by-zero."""
        ldrs = [torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)]
        merged = merge_exposures(ldrs, [1.0, 2.0])
        assert torch.isfinite(merged).all()

    def test_mismatched_counts_are_rejected(self):
        with pytest.raises(ValueError, match="exposure times"):
            merge_exposures([torch.rand(3, 4, 4)], [1.0, 2.0])

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="differing shapes"):
            merge_exposures([torch.rand(3, 4, 4), torch.rand(3, 4, 5)],
                            [1.0, 2.0])

    def test_an_empty_stack_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            merge_exposures([], [])


class TestDataset:
    def test_it_finds_scenes_in_the_training_layout(self, scenes):
        ds = KalantariDataset(scenes, split="train")
        assert ds.scene_ids == ["scene_00", "scene_01"]

    def test_it_finds_scenes_in_the_test_layout(self, scenes):
        """The released test scenes use HDRImg.hdr and exposure.txt."""
        ds = KalantariDataset(scenes, split="test")
        assert ds.scene_ids == ["001"]

    def test_samples_are_packed_cfa(self, scenes):
        ds = KalantariDataset(scenes, split="train")
        sample = ds[0]
        assert sample["x"].shape[0] == 4
        assert sample["x"].shape == sample["y"].shape
        assert torch.isfinite(sample["x"]).all()

    def test_dimensions_are_trimmed_for_the_encoder(self, scenes):
        ds = KalantariDataset(scenes, split="train")
        _, h, w = ds[0]["x"].shape
        assert (h * 2) % 16 == 0 and (w * 2) % 16 == 0

    def test_the_merge_source_works_without_a_reference_hdr(self, scenes, tmp_path):
        d = tmp_path / "nohdr" / "train" / "scene"
        write_scene(d)
        os.unlink(d / "ref_hdr_aligned.hdr")
        ds = KalantariDataset(str(tmp_path / "nohdr"), split="train",
                              source="merge")
        assert ds.num_sources == 1 and ds[0]["x"].shape[0] == 4

    def test_hdr_source_skips_scenes_without_a_reference(self, tmp_path):
        d = tmp_path / "nohdr" / "train" / "scene"
        write_scene(d)
        os.unlink(d / "ref_hdr_aligned.hdr")
        with pytest.raises(RuntimeError, match="reference HDR"):
            KalantariDataset(str(tmp_path / "nohdr"), split="train")

    def test_the_ldr_stack_can_be_returned(self, scenes):
        ds = KalantariDataset(scenes, split="train", return_ldr=True)
        sample = ds[0]
        assert sample["ldr"].shape == (3, 3, 64, 96)
        assert sample["exposure_times"] == [0.25, 1.0, 4.0]

    def test_reference_files_are_not_mistaken_for_inputs(self, scenes):
        """`ref_*` matches the bare *.tif fallback and must be excluded."""
        scene = os.path.join(scenes, "train", "scene_00")
        cv2.imwrite(os.path.join(scene, "ref_1_aligned.tif"),
                    np.zeros((64, 96, 3), np.uint8))
        found = KalantariDataset.find_ldrs(scene)
        assert all("ref_" not in os.path.basename(p) for p in found)

    def test_max_side_downscales(self, scenes):
        ds = KalantariDataset(scenes, split="train", max_side=48)
        _, h, w = ds[0]["x"].shape
        assert max(h * 2, w * 2) <= 48

    def test_unknown_source_is_rejected(self, scenes):
        with pytest.raises(ValueError, match="source must be"):
            KalantariDataset(scenes, source="guess")

    def test_a_tiny_max_side_is_rejected(self, scenes):
        with pytest.raises(ValueError, match="max_side"):
            KalantariDataset(scenes, max_side=8)


class TestEmptySkeleton:
    """The state this checkout is actually in."""

    def test_empty_scene_directories_give_an_explanatory_error(self, tmp_path):
        for i in range(3):
            (tmp_path / "train" / f"{i:03d}").mkdir(parents=True)
        with pytest.raises(RuntimeError) as excinfo:
            KalantariDataset(str(tmp_path), split="train")
        message = str(excinfo.value)
        assert "3 scene directories" in message
        assert "not been downloaded" in message

    def test_a_missing_split_directory_says_so(self, tmp_path):
        with pytest.raises(RuntimeError, match="does not exist"):
            KalantariDataset(str(tmp_path), split="train")

    @pytest.mark.skipif(
        not os.path.isdir("/lus/eagle/projects/lighthouse-purdue/ryanchen/"
                          "datasets/kalantari2017/train"),
        reason="the real kalantari2017 tree is not present")
    def test_the_real_tree_is_still_empty(self):
        """
        Documents the checkout's state: if this ever fails, the imagery
        has been downloaded and the loader can finally be run for real.
        """
        root = ("/lus/eagle/projects/lighthouse-purdue/ryanchen/datasets/"
                "kalantari2017")
        with pytest.raises(RuntimeError, match="none containing"):
            KalantariDataset(root, split="train")
