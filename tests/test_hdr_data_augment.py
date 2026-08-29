"""
CFA-aware augmentation — `hdr_data.augment`.

Why this file matters
─────────────────────
`train_A100_MoE_two_phase.make_d4_transform` is the only place the
training pipeline reorders Bayer channels, and its behaviour is pinned by
tests/test_d4_augmentation.py as "the packed transform equals a true
mosaic flip". That is exactly what it does — and a true mosaic flip moves
the CFA phase: flipping a BGGR readout horizontally produces a GBRG one.
The transform does not track that, so on roughly half of augmented
samples the model is handed GBRG data labelled BGGR, and the matching GT
is demosaiced by the BGGR-only GBTF.

This file pins three things:

* `make_d4_transform_compat` is **bit-identical** to the legacy transform
  under a shared RNG seed, so existing runs stay reproducible;
* the legacy/pixel mode really does reach all four phases, which is the
  concrete statement of the issue above;
* the new `mode="cell"` default gives all eight D4 symmetries with the
  phase **preserved**, which is what makes it safe to use with a
  pattern-specific demosaicer.
"""

import random

import pytest
import torch

from hdr_data.augment import (
    CenterCropPacked,
    Compose,
    D4Transform,
    RandomCropPacked,
    identity_transform,
    make_d4_transform_compat,
)
from hdr_data.bayer import pack, unpack
from train_A100_MoE_two_phase import make_d4_transform


@pytest.fixture
def pair():
    """An unbatched (noisy, clean) pair with distinct values everywhere."""
    g = torch.Generator().manual_seed(5)
    return (torch.rand((4, 16, 16), generator=g),
            torch.rand((4, 16, 16), generator=g))


class TestLegacyCompatibility:
    @pytest.mark.parametrize("allow_transpose", [True, False])
    @pytest.mark.parametrize("seed", range(8))
    def test_bit_identical_to_make_d4_transform(self, pair, allow_transpose, seed):
        noisy, clean = pair
        legacy = make_d4_transform(allow_transpose)
        new = make_d4_transform_compat(allow_transpose)

        random.seed(seed)
        want_n, want_c = legacy(noisy.clone(), clean.clone())
        random.seed(seed)
        got_n, got_c = new(noisy.clone(), clean.clone())

        assert torch.equal(want_n, got_n)
        assert torch.equal(want_c, got_c)

    def test_the_same_draw_is_shared_by_both_images(self, pair):
        """Noisy and clean must stay pixel-aligned or supervision breaks."""
        noisy, clean = pair
        tf = D4Transform("BGGR", rng=random.Random(0))
        n, c = tf(noisy.clone(), noisy.clone())
        assert torch.equal(n, c)

    def test_transpose_is_skipped_when_disallowed(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", allow_transpose=False, rng=random.Random(1))
        for _ in range(30):
            tf(noisy.clone(), clean.clone())
            assert "transpose" not in tf.last_ops


class TestPhaseBehaviour:
    def test_pixel_mode_reaches_every_phase(self, pair):
        """The concrete form of the issue: the phase is not preserved."""
        noisy, clean = pair
        tf = D4Transform("BGGR", mode="pixel", rng=random.Random(2))
        seen = set()
        for _ in range(200):
            tf(noisy.clone(), clean.clone())
            seen.add(tf.last_pattern)
        assert seen == {"BGGR", "GBRG", "GRBG", "RGGB"}

    def test_legacy_mode_moves_the_phase_the_same_way(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", mode="legacy", rng=random.Random(2))
        seen = set()
        for _ in range(200):
            tf(noisy.clone(), clean.clone())
            seen.add(tf.last_pattern)
        assert len(seen) > 1

    def test_cell_mode_never_moves_the_phase(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", mode="cell", rng=random.Random(3))
        for _ in range(200):
            tf(noisy.clone(), clean.clone())
            assert tf.last_pattern == "BGGR"

    def test_cell_mode_still_reaches_all_eight_symmetries(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", mode="cell", rng=random.Random(4))
        seen = set()
        for _ in range(300):
            tf(noisy.clone(), clean.clone())
            seen.add(tf.last_ops)
        assert len(seen) == 8

    def test_restricting_to_the_pattern_keeps_it_fixed(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", mode="pixel", restrict_to_pattern=True,
                         rng=random.Random(5))
        for _ in range(100):
            tf(noisy.clone(), clean.clone())
            assert tf.last_pattern == "BGGR"

    def test_restriction_still_allows_the_transpose_for_bggr(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", mode="pixel", restrict_to_pattern=True,
                         rng=random.Random(6))
        ops = set()
        for _ in range(200):
            tf(noisy.clone(), clean.clone())
            ops |= set(tf.last_ops)
        assert ops == {"transpose"}

    def test_cell_mode_output_is_a_valid_mosaic_permutation(self, pair):
        """No values invented or lost — just moved."""
        noisy, clean = pair
        tf = D4Transform("BGGR", mode="cell", rng=random.Random(7))
        out, _ = tf(noisy.clone(), clean.clone())
        assert torch.equal(torch.sort(out.flatten()).values,
                           torch.sort(noisy.flatten()).values)


class TestValidation:
    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            D4Transform("BGGR", mode="affine")

    def test_probability_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match="p must be"):
            D4Transform("BGGR", p=1.5)

    def test_mismatched_shapes_are_rejected(self):
        tf = D4Transform("BGGR")
        with pytest.raises(ValueError, match="same shape"):
            tf(torch.rand(4, 8, 8), torch.rand(4, 8, 9))

    def test_batched_input_is_rejected(self):
        """The transform runs per sample, inside the dataset."""
        tf = D4Transform("BGGR")
        with pytest.raises(ValueError, match=r"\[4, h, w\]"):
            tf(torch.rand(2, 4, 8, 8), torch.rand(2, 4, 8, 8))

    def test_non_square_input_never_transposes_outside_legacy_mode(self):
        tf = D4Transform("BGGR", mode="cell", rng=random.Random(8))
        n, c = torch.rand(4, 8, 12), torch.rand(4, 8, 12)
        for _ in range(50):
            out, _ = tf(n.clone(), c.clone())
            assert out.shape == n.shape

    def test_p_zero_never_applies_anything(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", p=0.0, rng=random.Random(9))
        for _ in range(20):
            out, _ = tf(noisy.clone(), clean.clone())
            assert torch.equal(out, noisy) and tf.last_ops == ()

    def test_p_one_always_applies_everything(self, pair):
        noisy, clean = pair
        tf = D4Transform("BGGR", p=1.0, rng=random.Random(10))
        tf(noisy.clone(), clean.clone())
        assert set(tf.last_ops) == {"hflip", "vflip", "transpose"}


class TestCrops:
    def test_random_crop_returns_the_requested_size(self, pair):
        noisy, clean = pair
        n, c = RandomCropPacked(8, rng=random.Random(0))(noisy, clean)
        assert n.shape == (4, 8, 8) and c.shape == (4, 8, 8)

    def test_random_crop_takes_the_same_window_from_both(self, pair):
        noisy, _ = pair
        n, c = RandomCropPacked(8, rng=random.Random(1))(noisy, noisy)
        assert torch.equal(n, c)

    def test_random_crop_moves_between_calls(self, pair):
        noisy, clean = pair
        crop = RandomCropPacked(8, rng=random.Random(2))
        outs = {crop(noisy, clean)[0].sum().item() for _ in range(20)}
        assert len(outs) > 1

    def test_centre_crop_is_deterministic_and_centred(self):
        x = torch.arange(4 * 16 * 16, dtype=torch.float32).view(4, 16, 16)
        a, _ = CenterCropPacked(8)(x, x)
        b, _ = CenterCropPacked(8)(x, x)
        assert torch.equal(a, b)
        assert torch.equal(a, x[:, 4:12, 4:12])

    @pytest.mark.parametrize("cls", [RandomCropPacked, CenterCropPacked])
    def test_crop_larger_than_the_image_is_rejected(self, cls, pair):
        noisy, clean = pair
        with pytest.raises(ValueError, match="Cannot take"):
            cls(64)(noisy, clean)

    @pytest.mark.parametrize("cls", [RandomCropPacked, CenterCropPacked])
    def test_non_positive_size_is_rejected(self, cls):
        with pytest.raises(ValueError, match="positive"):
            cls(0)


class TestCompose:
    def test_transforms_run_left_to_right(self, pair):
        noisy, clean = pair
        tf = Compose([RandomCropPacked(8, rng=random.Random(0)),
                      D4Transform("BGGR", rng=random.Random(0))])
        n, c = tf(noisy, clean)
        assert n.shape == (4, 8, 8) and len(tf) == 2

    def test_empty_compose_is_the_identity(self, pair):
        noisy, clean = pair
        n, c = Compose([])(noisy, clean)
        assert torch.equal(n, noisy) and torch.equal(c, clean)

    def test_repr_lists_the_stages(self):
        tf = Compose([CenterCropPacked(4), D4Transform("BGGR")])
        assert "CenterCropPacked" in repr(tf) and "D4Transform" in repr(tf)

    def test_identity_transform_passes_through(self, pair):
        noisy, clean = pair
        n, c = identity_transform(noisy, clean)
        assert n is noisy and c is clean
