"""
CFA pattern algebra — `hdr_data.bayer`.

Why this file matters
─────────────────────
Every dataset, baseline and metric in the new packages routes its CFA
handling through this module, so a wrong permutation here is not a local
bug: it silently colour-scrambles training data, ground truth and every
comparison at once. Nothing downstream would crash.

The ground truth used throughout is therefore never the packed tensor —
it is the SENSOR MOSAIC. With S(p) = pixel_shuffle(p, 2) the contract for
a true (`FlipMode.PIXEL`) symmetry is

    S(hflip(p))  == hflip(S(p))
    S(vflip(p))  == vflip(S(p))
    S(transpose(p)) == S(p).transpose(-2, -1)

as bit-exact equalities on hand-built mosaics with unique per-pixel
values, so a one-cell mis-permutation is caught rather than averaged away.

The second contract is the *phase*: a true flip of a BGGR mosaic produces
a GBRG one, and `pattern_after` must say so. That is the fact the older
`make_d4_transform` docstring gets wrong (see test_hdr_data_augment.py),
and getting it right here is what lets the new augmentation offer a
phase-preserving mode at all.
"""

import pytest
import torch
import torch.nn.functional as F

from hdr_data import bayer as B


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept here on purpose — shared helpers.py is BGGR-specific)
# ─────────────────────────────────────────────────────────────────────────────

def unique_mosaic(h=6, w=8):
    """A [1, 1, h, w] mosaic whose every pixel value is distinct."""
    return torch.arange(h * w, dtype=torch.float32).view(1, 1, h, w)


ALL_PATTERNS = tuple(B.CFA_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

class TestPatternNames:
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_canonical_accepts_any_case(self, pattern):
        assert B.canonical_pattern(pattern.lower()) == pattern
        assert B.canonical_pattern(f"  {pattern}  ") == pattern

    def test_unknown_pattern_names_the_alternatives(self):
        with pytest.raises(ValueError, match="BGGR"):
            B.canonical_pattern("XYZW")

    def test_non_string_pattern_is_a_type_error(self):
        with pytest.raises(TypeError):
            B.canonical_pattern(4)

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_every_pattern_has_two_greens_one_red_one_blue(self, pattern):
        colors = B.pattern_colors(pattern)
        assert sorted(colors) == ["B", "G", "G", "R"]

    def test_channel_names_disambiguate_the_greens(self):
        assert B.packed_channel_names("BGGR") == ("B", "G1", "G2", "R")
        assert B.packed_channel_names("GRBG") == ("G1", "R", "B", "G2")

    def test_color_index_maps_to_rgb_planes(self):
        assert B.color_index("BGGR") == (2, 1, 1, 0)
        assert B.color_index("RGGB") == (0, 1, 1, 2)

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_green_channels_are_the_green_positions(self, pattern):
        colors = B.pattern_colors(pattern)
        for idx in B.green_channels(pattern):
            assert colors[idx] == "G"


# ─────────────────────────────────────────────────────────────────────────────
# Pack / unpack
# ─────────────────────────────────────────────────────────────────────────────

class TestPackUnpack:
    def test_round_trip_is_exact(self):
        m = unique_mosaic()
        assert torch.equal(B.unpack(B.pack(m)), m)

    def test_pack_matches_pixel_unshuffle(self):
        m = unique_mosaic()
        assert torch.equal(B.pack(m), F.pixel_unshuffle(m, 2))

    def test_channel_c_holds_cell_offset_c(self):
        """Channel c must be the sample at intra-cell offset (c//2, c%2)."""
        m = unique_mosaic(4, 4)
        p = B.pack(m)
        for c in range(4):
            r, s = c // 2, c % 2
            assert torch.equal(p[0, c], m[0, 0, r::2, s::2])

    def test_unbatched_input_stays_unbatched(self):
        m = unique_mosaic()[0]                       # [1, H, W]
        p = B.pack(m)
        assert p.shape == (4, 3, 4)
        assert torch.equal(B.unpack(p), m)

    def test_odd_mosaic_dims_are_rejected(self):
        with pytest.raises(ValueError, match="even"):
            B.pack(torch.zeros(1, 1, 5, 4))

    def test_wrong_channel_count_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[\.\.\., 4, h, w\]"):
            B.unpack(torch.zeros(1, 3, 4, 4))

    def test_batched_pack_preserves_the_batch(self):
        m = torch.rand(3, 1, 8, 8)
        assert B.pack(m).shape == (3, 4, 4, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Symmetries: the mosaic-equality contract
# ─────────────────────────────────────────────────────────────────────────────

class TestPixelModeMatchesMosaicSymmetry:
    """PIXEL mode must be indistinguishable from flipping the raw readout."""

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_hflip(self, pattern):
        m = unique_mosaic()
        out, _ = B.hflip(B.pack(m), pattern)
        assert torch.equal(B.unpack(out), torch.flip(m, dims=(-1,)))

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_vflip(self, pattern):
        m = unique_mosaic()
        out, _ = B.vflip(B.pack(m), pattern)
        assert torch.equal(B.unpack(out), torch.flip(m, dims=(-2,)))

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_rot180(self, pattern):
        m = unique_mosaic()
        out, _ = B.rot180(B.pack(m), pattern)
        assert torch.equal(B.unpack(out), torch.flip(m, dims=(-2, -1)))

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_transpose(self, pattern):
        m = unique_mosaic(8, 8)
        out, _ = B.transpose(B.pack(m), pattern)
        assert torch.equal(B.unpack(out), m.transpose(-2, -1))

    def test_permutation_constants_are_the_documented_ones(self):
        """The training script's literals, which existing tests pin."""
        assert B.PERM_HFLIP == (1, 0, 3, 2)
        assert B.PERM_VFLIP == (2, 3, 0, 1)
        assert B.PERM_TRANSPOSE == (0, 2, 1, 3)
        assert B.PERM_ROT180 == (3, 2, 1, 0)


class TestPhaseTracking:
    """A true flip moves the CFA phase; `pattern_after` must report it."""

    def test_hflip_of_bggr_is_gbrg(self):
        assert B.pattern_after("BGGR", "hflip") == "GBRG"

    def test_vflip_of_bggr_is_grbg(self):
        assert B.pattern_after("BGGR", "vflip") == "GRBG"

    def test_rot180_of_bggr_is_rggb(self):
        assert B.pattern_after("BGGR", "rot180") == "RGGB"

    def test_transpose_preserves_only_the_diagonal_phases(self):
        """
        Transpose reflects the cell about its main diagonal, so it fixes
        whichever colours sit on that diagonal and swaps the pair on the
        anti-diagonal. BGGR and RGGB carry R and B on the main diagonal
        and survive unchanged — which is what makes the training script's
        transpose augmentation safe on square patches. GRBG and GBRG carry
        them on the anti-diagonal and swap into each other.
        """
        assert B.pattern_after("BGGR", "transpose") == "BGGR"
        assert B.pattern_after("RGGB", "transpose") == "RGGB"
        assert B.pattern_after("GRBG", "transpose") == "GBRG"
        assert B.pattern_after("GBRG", "transpose") == "GRBG"

    def test_identity_is_identity(self):
        for pattern in ALL_PATTERNS:
            assert B.pattern_after(pattern, "identity") == pattern

    @pytest.mark.parametrize("op", ["hflip", "vflip", "rot180", "transpose"])
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_every_symmetry_is_an_involution_on_the_phase(self, op, pattern):
        once = B.pattern_after(pattern, op)
        assert B.pattern_after(once, op) == pattern

    def test_reported_phase_matches_the_actual_colour_layout(self):
        """
        Cross-check the bookkeeping against the pixels: build an RGB image,
        mosaic it, flip it, and confirm the reported phase is the one that
        recovers the correct colours.
        """
        rgb = torch.rand(1, 3, 8, 8)
        packed = B.rgb_to_packed(rgb, "BGGR")
        flipped, new_pattern = B.hflip(packed, "BGGR")

        # The flipped image should equal mosaicking the flipped RGB in the
        # NEW pattern — not in BGGR.
        expected = B.rgb_to_packed(torch.flip(rgb, dims=(-1,)), new_pattern)
        assert torch.equal(flipped, expected)
        assert not torch.equal(
            flipped, B.rgb_to_packed(torch.flip(rgb, dims=(-1,)), "BGGR"))

    def test_unknown_op_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown op"):
            B.pattern_after("BGGR", "shear")


class TestCellMode:
    """CELL mode trades a half-cell shift for a phase that never moves."""

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    @pytest.mark.parametrize("fn", [B.hflip, B.vflip, B.rot180])
    def test_phase_is_preserved(self, pattern, fn):
        p = B.pack(unique_mosaic())
        _, new_pattern = fn(p, pattern, mode=B.FlipMode.CELL)
        assert new_pattern == pattern

    @pytest.mark.parametrize("fn", [B.hflip, B.vflip, B.rot180])
    def test_applying_twice_is_the_identity(self, fn):
        p = B.pack(unique_mosaic())
        once, pat = fn(p, "BGGR", mode=B.FlipMode.CELL)
        twice, _ = fn(once, pat, mode=B.FlipMode.CELL)
        assert torch.equal(twice, p)

    def test_cells_move_but_their_contents_do_not(self):
        p = B.pack(unique_mosaic())
        out, _ = B.hflip(p, "BGGR", mode=B.FlipMode.CELL)
        assert torch.equal(out, torch.flip(p, dims=(-1,)))

    def test_cell_and_pixel_modes_differ(self):
        p = B.pack(unique_mosaic())
        cell, _ = B.hflip(p, "BGGR", mode=B.FlipMode.CELL)
        pixel, _ = B.hflip(p, "BGGR", mode=B.FlipMode.PIXEL)
        assert not torch.equal(cell, pixel)


class TestShiftPhase:
    def test_shifting_by_one_column_advances_the_phase(self):
        m = unique_mosaic(8, 8)
        out, pattern = B.shift_phase(m, "BGGR", 0, 1)
        assert pattern == "GBRG"
        assert out.shape[-2:] == (8, 6)

    def test_shifting_by_one_row_advances_the_phase(self):
        m = unique_mosaic(8, 8)
        _, pattern = B.shift_phase(m, "BGGR", 1, 0)
        assert pattern == "GRBG"

    def test_shifting_both_gives_the_diagonal_phase(self):
        m = unique_mosaic(8, 8)
        _, pattern = B.shift_phase(m, "BGGR", 1, 1)
        assert pattern == "RGGB"

    def test_zero_shift_is_a_no_op(self):
        m = unique_mosaic(8, 8)
        out, pattern = B.shift_phase(m, "BGGR", 0, 0)
        assert torch.equal(out, m) and pattern == "BGGR"

    def test_result_keeps_even_dimensions(self):
        """Odd dimensions could not be packed again."""
        m = unique_mosaic(8, 8)
        for dy in (0, 1):
            for dx in (0, 1):
                out, _ = B.shift_phase(m, "BGGR", dy, dx)
                assert out.shape[-2] % 2 == 0 and out.shape[-1] % 2 == 0

    def test_a_flip_plus_a_shift_restores_the_original_phase(self):
        """The documented way to keep a true flip in its original phase."""
        m = unique_mosaic(8, 8)
        flipped = torch.flip(m, dims=(-1,))
        restored, pattern = B.shift_phase(flipped, B.pattern_after("BGGR", "hflip"),
                                          0, 1)
        assert pattern == "BGGR"

    def test_shift_of_more_than_one_is_rejected(self):
        with pytest.raises(ValueError, match="0 or 1"):
            B.shift_phase(unique_mosaic(8, 8), "BGGR", 2, 0)


# ─────────────────────────────────────────────────────────────────────────────
# RGB <-> CFA
# ─────────────────────────────────────────────────────────────────────────────

class TestRGBConversion:
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_mosaic_samples_the_right_plane_at_each_position(self, pattern):
        rgb = torch.rand(2, 3, 8, 10)
        mosaic = B.rgb_to_mosaic(rgb, pattern)
        colors = B.pattern_colors(pattern)
        plane = {"R": 0, "G": 1, "B": 2}
        for c in range(4):
            r, s = c // 2, c % 2
            expected = rgb[:, plane[colors[c]], r::2, s::2]
            assert torch.equal(mosaic[:, 0, r::2, s::2], expected)

    def test_rgb_to_packed_is_mosaic_then_pack(self):
        rgb = torch.rand(1, 3, 8, 8)
        assert torch.equal(B.rgb_to_packed(rgb, "BGGR"),
                           B.pack(B.rgb_to_mosaic(rgb, "BGGR")))

    def test_odd_rgb_dims_are_rejected(self):
        with pytest.raises(ValueError, match="even"):
            B.rgb_to_mosaic(torch.rand(1, 3, 7, 8))

    def test_wrong_channel_count_is_rejected(self):
        with pytest.raises(ValueError, match="RGB"):
            B.rgb_to_mosaic(torch.rand(1, 4, 8, 8))

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_sparse_rgb_keeps_measured_samples_and_zeroes_the_rest(self, pattern):
        rgb = torch.rand(1, 3, 8, 8) + 0.1     # strictly positive
        packed = B.rgb_to_packed(rgb, pattern)
        sparse = B.packed_to_sparse_rgb(packed, pattern)
        masks = B.cfa_masks(pattern, 8, 8)
        for plane, color in enumerate(("R", "G", "B")):
            mask = masks[color][0]
            assert torch.equal(sparse[0, plane][mask], rgb[0, plane][mask])
            assert float(sparse[0, plane][~mask].abs().max()) == 0.0

    def test_half_rgb_averages_the_two_greens(self):
        packed = torch.zeros(1, 4, 2, 2)
        packed[:, 1] = 0.2        # G1 for BGGR
        packed[:, 2] = 0.6        # G2
        packed[:, 0] = 0.5        # B
        packed[:, 3] = 0.9        # R
        half = B.packed_to_half_rgb(packed, "BGGR")
        assert torch.allclose(half[:, 0], torch.full((1, 2, 2), 0.9))
        assert torch.allclose(half[:, 1], torch.full((1, 2, 2), 0.4))
        assert torch.allclose(half[:, 2], torch.full((1, 2, 2), 0.5))

    def test_half_rgb_halves_the_resolution(self):
        packed = torch.rand(2, 4, 6, 8)
        assert B.packed_to_half_rgb(packed).shape == (2, 3, 6, 8)


class TestCFAMasks:
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_masks_partition_the_image(self, pattern):
        masks = B.cfa_masks(pattern, 8, 8)
        total = masks["R"].int() + masks["G"].int() + masks["B"].int()
        assert torch.equal(total, torch.ones_like(total))

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_green_covers_half_and_red_blue_a_quarter_each(self, pattern):
        masks = B.cfa_masks(pattern, 8, 8)
        assert float(masks["G"].float().mean()) == 0.5
        assert float(masks["R"].float().mean()) == 0.25
        assert float(masks["B"].float().mean()) == 0.25

    def test_float_dtype_gives_ones_not_true(self):
        masks = B.cfa_masks("BGGR", 4, 4, dtype=torch.float32)
        assert masks["R"].dtype == torch.float32
        assert float(masks["R"].sum()) == 4.0

    def test_odd_dims_are_rejected(self):
        with pytest.raises(ValueError, match="even"):
            B.cfa_masks("BGGR", 5, 4)


class TestDeviceAndDtype:
    """Nothing here may silently promote or move a tensor."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_survives_the_round_trip(self, dtype):
        m = unique_mosaic().to(dtype)
        assert B.unpack(B.pack(m)).dtype == dtype
        assert B.hflip(B.pack(m), "BGGR")[0].dtype == dtype

    def test_rgb_conversion_preserves_dtype(self):
        rgb = torch.rand(1, 3, 8, 8, dtype=torch.float64)
        assert B.rgb_to_packed(rgb, "BGGR").dtype == torch.float64
