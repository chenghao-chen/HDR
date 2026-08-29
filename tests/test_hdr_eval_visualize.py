"""
Figures — `hdr_eval.visualize`.

Why this file matters
─────────────────────
These functions write the artefacts a human actually looks at, and they
do it with only torch and PIL because matplotlib is not installed in this
environment. That constraint makes a few things easy to get wrong in ways
no metric would catch: writing a linear-HDR tensor straight to 8 bits (a
black rectangle), mismatched panel sizes silently stretched, or a
colormap that inverts.

The tests assert the file is written, is a real image of the expected
size, and that the tone curve is applied — a linear write of a dark HDR
frame is the specific bug being guarded against.
"""

import os

import pytest
import torch

from hdr_eval.visualize import (
    colorize,
    save_comparison,
    save_crop_comparison,
    save_error_heatmap,
    save_gate_map,
    save_image,
    to_display,
)

Image = pytest.importorskip("PIL.Image", reason="PIL is required")


@pytest.fixture
def rgb():
    g = torch.Generator().manual_seed(0)
    return torch.rand((1, 3, 64, 64), generator=g)


def image_size(path):
    with Image.open(path) as im:
        return im.size          # (width, height)


class TestToDisplay:
    def test_it_applies_the_tone_curve(self):
        """A dark HDR frame must not be written out as black."""
        dark = torch.full((1, 3, 8, 8), 0.01)
        assert float(to_display(dark).mean()) > 0.3

    def test_it_accepts_unbatched_input(self):
        assert to_display(torch.rand(3, 8, 8)).shape == (1, 3, 8, 8)

    def test_it_takes_only_the_first_image_of_a_batch(self):
        assert to_display(torch.rand(4, 3, 8, 8)).shape == (1, 3, 8, 8)

    def test_scaling_shrinks_the_output(self, rgb):
        assert to_display(rgb, scale=0.5).shape == (1, 3, 32, 32)

    def test_an_out_of_range_scale_is_rejected(self, rgb):
        with pytest.raises(ValueError, match="scale"):
            to_display(rgb, scale=2.0)

    def test_a_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[C, H, W\]"):
            to_display(torch.rand(8, 8))

    def test_the_result_is_within_the_display_range(self, rgb):
        out = to_display(rgb)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


class TestSaveImage:
    def test_it_writes_a_png(self, tmp_path, rgb):
        path = save_image(str(tmp_path / "a.png"), rgb[0])
        assert os.path.exists(path) and image_size(path) == (64, 64)

    def test_it_writes_a_jpeg(self, tmp_path, rgb):
        path = save_image(str(tmp_path / "a.jpg"), rgb[0])
        with Image.open(path) as im:
            assert im.format == "JPEG"

    def test_it_creates_missing_directories(self, tmp_path, rgb):
        path = save_image(str(tmp_path / "deep" / "a.png"), rgb[0])
        assert os.path.exists(path)

    def test_a_single_channel_image_is_expanded_to_grey(self, tmp_path):
        path = save_image(str(tmp_path / "g.png"), torch.rand(1, 16, 16))
        with Image.open(path) as im:
            assert im.mode == "RGB"

    def test_an_impossible_channel_count_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="channels"):
            save_image(str(tmp_path / "x.png"), torch.rand(5, 8, 8))


class TestComparison:
    def test_it_writes_a_strip_of_the_expected_width(self, tmp_path, rgb):
        path = save_comparison(str(tmp_path / "c.jpg"),
                               {"a": rgb, "b": rgb, "c": rgb},
                               scale=1.0, separator=4, label=False)
        width, _ = image_size(path)
        assert width == 64 * 3 + 4 * 2

    def test_labels_add_a_caption_bar(self, tmp_path, rgb):
        plain = save_comparison(str(tmp_path / "p.jpg"), {"a": rgb},
                                scale=1.0, label=False)
        labelled = save_comparison(str(tmp_path / "l.jpg"), {"a": rgb},
                                   scale=1.0, label=True)
        assert image_size(labelled)[1] > image_size(plain)[1]

    def test_mismatched_panel_sizes_are_rejected(self, tmp_path, rgb):
        """Stretching them silently would misrepresent the comparison."""
        with pytest.raises(ValueError, match="matching sizes"):
            save_comparison(str(tmp_path / "c.jpg"),
                            {"a": rgb, "b": torch.rand(1, 3, 32, 32)},
                            scale=1.0)

    def test_no_panels_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="at least one panel"):
            save_comparison(str(tmp_path / "c.jpg"), {})

    def test_a_single_panel_works(self, tmp_path, rgb):
        path = save_comparison(str(tmp_path / "c.jpg"), {"only": rgb},
                               scale=1.0, label=False)
        assert image_size(path) == (64, 64)


class TestColorize:
    def test_it_returns_three_channels(self):
        out = colorize(torch.rand(1, 1, 8, 8))
        assert out.shape == (3, 8, 8)

    def test_the_ends_of_the_scale_differ(self):
        field = torch.linspace(0, 1, 16).view(1, 1, 1, 16)
        out = colorize(field)
        assert not torch.allclose(out[:, 0, 0], out[:, 0, -1])

    def test_a_constant_field_does_not_divide_by_zero(self):
        out = colorize(torch.full((1, 1, 4, 4), 0.5))
        assert torch.isfinite(out).all()

    def test_explicit_limits_are_honoured(self):
        field = torch.tensor([[[[0.5]]]])
        low = colorize(field, vmin=0.0, vmax=1.0)
        high = colorize(field, vmin=0.4, vmax=0.6)
        assert torch.isfinite(low).all() and torch.isfinite(high).all()

    def test_output_stays_in_the_display_range(self):
        out = colorize(torch.randn(1, 1, 8, 8))
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


class TestErrorHeatmap:
    def test_it_writes_an_image(self, tmp_path, rgb):
        path = save_error_heatmap(str(tmp_path / "e.png"), rgb * 0.8, rgb,
                                  scale=1.0)
        assert image_size(path) == (64, 64)

    def test_mismatched_inputs_are_rejected(self, tmp_path, rgb):
        with pytest.raises(ValueError, match="must match"):
            save_error_heatmap(str(tmp_path / "e.png"), rgb,
                               torch.rand(1, 3, 32, 32))


class TestGateMap:
    def test_up_to_three_experts_compose_into_one_rgb(self, tmp_path):
        gates = torch.rand(1, 3, 32, 32)
        path = save_gate_map(str(tmp_path / "g.jpg"), gates, scale=1.0)
        assert image_size(path) == (32, 32)

    def test_two_experts_leave_the_third_channel_dark(self, tmp_path):
        gates = torch.rand(1, 2, 16, 16)
        path = save_gate_map(str(tmp_path / "g.jpg"), gates, scale=1.0)
        assert os.path.exists(path)

    def test_more_than_three_experts_become_a_strip(self, tmp_path):
        gates = torch.rand(1, 5, 16, 16)
        path = save_gate_map(str(tmp_path / "g.jpg"), gates, scale=1.0)
        width, _ = image_size(path)
        assert width >= 16 * 5

    def test_a_wrong_rank_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"\[B, K, H, W\]"):
            save_gate_map(str(tmp_path / "g.jpg"), torch.rand(3, 16, 16))


class TestCropComparison:
    def test_it_zooms_the_requested_window(self, tmp_path, rgb):
        path = save_crop_comparison(str(tmp_path / "z.jpg"),
                                    {"a": rgb, "b": rgb},
                                    box=(8, 8, 16, 16), zoom=4)
        width, _ = image_size(path)
        assert width >= 16 * 4 * 2

    def test_a_crop_outside_the_image_is_rejected(self, tmp_path, rgb):
        with pytest.raises(ValueError, match="does not fit"):
            save_crop_comparison(str(tmp_path / "z.jpg"), {"a": rgb},
                                 box=(60, 60, 16, 16))

    def test_a_degenerate_box_is_rejected(self, tmp_path, rgb):
        with pytest.raises(ValueError, match="positive"):
            save_crop_comparison(str(tmp_path / "z.jpg"), {"a": rgb},
                                 box=(0, 0, 0, 16))

    def test_a_zoom_below_one_is_rejected(self, tmp_path, rgb):
        with pytest.raises(ValueError, match="zoom"):
            save_crop_comparison(str(tmp_path / "z.jpg"), {"a": rgb},
                                 box=(0, 0, 8, 8), zoom=0)
