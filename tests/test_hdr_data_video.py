"""
The i2-2kfps video corpus — `hdr_data.video_i2`.

Why this file matters
─────────────────────
This loader turns decoded RGB video into a virtual CFA sensor, and two of
its jobs are easy to get silently wrong:

* **Linearisation.** The noise model is stated in linear photons. Feeding
  it gamma-encoded frames puts the noise in the wrong places — plausible
  looking output, wrong training distribution. `srgb_to_linear` is checked
  against the standard's own anchor points and for exact round-tripping.

* **Refusing a broken decoder.** On the Polaris login nodes OpenCV's
  ffmpeg cannot initialise its scaling context, and every `read()` then
  returns a frame-shaped buffer of garbage *while reporting success*.
  Training on that silently would be far worse than failing, so
  `probe_decoder` write-decodes known levels and the dataset refuses to
  index mp4 clips until it passes. The probe is asserted here to detect
  the failure by its signature (every frame identical), which is what it
  actually returns on the affected hosts.

The frame-directory path needs no decoder, so it carries the behavioural
tests; the mp4 path is exercised only where decoding demonstrably works.
"""

import os

import numpy as np
import pytest
import torch

from hdr_data.video_i2 import (
    DecoderUnavailable,
    I2VideoDataset,
    linear_to_srgb,
    probe_decoder,
    srgb_to_linear,
)


def _decoder_works():
    """True only where OpenCV can really decode, not merely open, a clip."""
    try:
        return bool(probe_decoder()["ok"])
    except Exception:
        return False


requires_decoder = pytest.mark.skipif(
    not _decoder_works(),
    reason="OpenCV cannot decode video on this host (see probe_decoder)")


@pytest.fixture
def frames_root(tmp_path):
    """A directory of pre-extracted .npy frames in the expected layout."""
    root = tmp_path / "i2"
    rng = np.random.default_rng(0)
    for split, clips in (("train", 2), ("test", 1)):
        for c in range(clips):
            d = root / split / f"clip_{c:02d}"
            d.mkdir(parents=True)
            for f in range(6):
                np.save(d / f"frame_{f:05d}.npy",
                        (rng.random((48, 64, 3)) * 255).astype(np.float32))
    return str(root)


class TestColourTransfer:
    def test_srgb_round_trip_is_exact(self):
        x = torch.linspace(0, 1, 257)
        assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-6)

    def test_the_standard_anchor_points(self):
        """0 -> 0, 1 -> 1, and the linear segment below the knee."""
        assert float(srgb_to_linear(torch.tensor(0.0))) == 0.0
        assert abs(float(srgb_to_linear(torch.tensor(1.0))) - 1.0) < 1e-6
        # Below 0.04045 the transfer function is a straight line of slope
        # 1/12.92; a gamma-2.2 approximation gets this region badly wrong.
        x = torch.tensor(0.02)
        assert abs(float(srgb_to_linear(x)) - 0.02 / 12.92) < 1e-9

    def test_it_darkens_the_midtones(self):
        mid = torch.tensor(0.5)
        assert float(srgb_to_linear(mid)) < 0.25

    def test_negative_input_is_clamped_not_nan(self):
        out = srgb_to_linear(torch.tensor([-0.5, 0.5]))
        assert torch.isfinite(out).all()


class TestDecoderProbe:
    def test_the_probe_returns_a_verdict_and_a_reason(self):
        result = probe_decoder()
        assert set(result) >= {"ok", "reason", "levels"}
        assert isinstance(result["ok"], bool)
        if not result["ok"]:
            assert result["reason"]

    def test_a_failing_probe_can_raise_instead(self):
        result = probe_decoder()
        if result["ok"]:
            pytest.skip("decoder works here; nothing to raise about")
        with pytest.raises(DecoderUnavailable):
            probe_decoder(raise_on_failure=True)

    def test_the_broken_decoder_signature_is_recognised(self):
        """
        The failure this guards against returns identical frames for
        different inputs. Where that is happening, the probe must say so
        rather than report success.
        """
        result = probe_decoder()
        if result["ok"] or not result["levels"]:
            pytest.skip("decoder works here")
        levels = result["levels"]
        if len(levels) == 6 and max(levels) - min(levels) < 1.0:
            assert "same mean level" in result["reason"]


class TestFrameDirectoryLoader:
    def test_it_indexes_every_clip_and_frame(self, frames_root):
        ds = I2VideoDataset(frames_root=frames_root, split="train",
                            frame_stride=1)
        assert ds.num_sources == 12 and not ds.uses_video

    def test_frame_stride_subsamples(self, frames_root):
        ds = I2VideoDataset(frames_root=frames_root, split="train",
                            frame_stride=3)
        assert ds.num_sources == 4

    def test_max_frames_caps_a_long_clip(self, frames_root):
        ds = I2VideoDataset(frames_root=frames_root, split="train",
                            frame_stride=1, max_frames_per_clip=2)
        assert ds.num_sources == 4

    def test_samples_are_packed_cfa_in_range(self, frames_root):
        ds = I2VideoDataset(frames_root=frames_root, split="train")
        sample = ds[0]
        assert sample["x"].shape[0] == 4
        assert torch.isfinite(sample["x"]).all()
        assert 0.0 <= float(sample["x"].min()) and float(sample["x"].max()) <= 1.0

    def test_frames_are_trimmed_to_a_multiple_of_sixteen(self, frames_root):
        """
        The encoder's three PixelUnshuffle(2) stages need packed dims
        divisible by 8, i.e. sensor dims divisible by 16.
        """
        ds = I2VideoDataset(frames_root=frames_root, split="train")
        _, h, w = ds[0]["x"].shape
        assert (h * 2) % 16 == 0 and (w * 2) % 16 == 0

    def test_the_source_id_names_clip_and_frame(self, frames_root):
        ds = I2VideoDataset(frames_root=frames_root, split="train",
                            frame_stride=1, return_meta=True)
        assert ds[0]["source_id"] == "clip_00_f00000"

    def test_linearisation_can_be_disabled(self, frames_root):
        lin = I2VideoDataset(frames_root=frames_root, split="test",
                             linearise=True)
        raw = I2VideoDataset(frames_root=frames_root, split="test",
                             linearise=False)
        assert not torch.equal(lin[0]["y"], raw[0]["y"])

    def test_scale_brightens_the_signal(self, frames_root):
        dim = I2VideoDataset(frames_root=frames_root, split="test",
                             noise="clean")
        bright = I2VideoDataset(frames_root=frames_root, split="test",
                                scale=4.0, noise="clean")
        assert float(bright[0]["y"].mean()) >= float(dim[0]["y"].mean())

    def test_an_empty_root_names_where_it_looked(self, tmp_path):
        with pytest.raises(RuntimeError, match="Looked in"):
            I2VideoDataset(frames_root=str(tmp_path), split="train")

    def test_cropping_and_noise_presets_compose(self, frames_root):
        ds = I2VideoDataset(frames_root=frames_root, split="train",
                            crop_size=8, noise="realistic")
        assert ds[0]["x"].shape == (4, 8, 8)


class TestValidation:
    def test_one_of_base_dir_or_frames_root_is_required(self):
        with pytest.raises(ValueError, match="base_dir"):
            I2VideoDataset(split="train")

    def test_zero_stride_is_rejected(self, frames_root):
        with pytest.raises(ValueError, match="frame_stride"):
            I2VideoDataset(frames_root=frames_root, frame_stride=0)

    def test_zero_max_frames_is_rejected(self, frames_root):
        with pytest.raises(ValueError, match="max_frames_per_clip"):
            I2VideoDataset(frames_root=frames_root, max_frames_per_clip=0)

    def test_mp4_indexing_refuses_a_broken_decoder(self, tmp_path):
        """The point of the guard: no silent training on garbage frames."""
        if _decoder_works():
            pytest.skip("decoder works here; the guard cannot trigger")
        clips = tmp_path / "train"
        clips.mkdir(parents=True)
        (clips / "a.mp4").write_bytes(b"not really a video")
        with pytest.raises(DecoderUnavailable, match="not usable"):
            I2VideoDataset(str(tmp_path), split="train")


@requires_decoder
class TestVideoLoader:
    """Only runs where OpenCV can actually decode."""

    @pytest.fixture
    def video_root(self, tmp_path):
        import cv2
        root = tmp_path / "clips"
        (root / "train").mkdir(parents=True)
        path = str(root / "train" / "clip.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 30.0, (64, 48))
        for i in range(8):
            writer.write(np.full((48, 64, 3), 30 + i * 20, np.uint8))
        writer.release()
        return str(root)

    def test_it_indexes_and_decodes(self, video_root):
        ds = I2VideoDataset(video_root, split="train", frame_stride=2,
                            cache_index=False)
        assert ds.uses_video and ds.num_sources >= 1
        assert ds[0]["x"].shape[0] == 4
        ds.close()

    def test_the_frame_count_cache_is_written(self, video_root):
        ds = I2VideoDataset(video_root, split="train", cache_index=True)
        assert os.path.exists(os.path.join(video_root,
                                           I2VideoDataset.INDEX_CACHE_NAME))
        ds.close()
