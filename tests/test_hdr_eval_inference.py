"""
Whole-frame inference — `hdr_eval.inference`.

Why this file matters
─────────────────────
Tiled inference is only trustworthy if it produces the same picture as
running the model once on the whole frame. This file asserts exactly that,
for a model whose output depends only on its input locally: tiled and full
must agree to floating-point noise, at several tile sizes and overlaps,
including sizes that are not multiples of the tile.

That equality is what catches the two failure modes the tiling in
test_dual_MoE_two_phase.py has:

* its accumulation window is uniform, so neighbouring tiles are averaged
  with equal weight right up to the tile edge and any tile-to-tile
  difference lands as a seam;
* it iterates ``range(0, Hp - patch + 1, stride)``, which drops the last
  partial tile whenever the padded size is not an exact multiple of the
  stride, leaving a strip of the image reconstructed from nothing.

Both would show up here as a mismatch against full-frame inference.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from hdr_eval.inference import (
    InferenceConfig,
    ModelOutput,
    autocast_context,
    blend_window,
    call_model,
    infer,
    infer_full,
    infer_tiled,
)


def local_model(x, snr):
    """A purely local model: every output pixel depends on one input cell."""
    rgb = F.interpolate(x[:, :3], scale_factor=2, mode="nearest")
    experts = rgb.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    gates = torch.full((x.shape[0], 2, x.shape[2] * 2, x.shape[3] * 2), 0.5)
    return rgb, experts, gates


def tensor_model(x, snr):
    """A model with no routing, as the classical baselines behave."""
    return F.interpolate(x[:, :3], scale_factor=2, mode="nearest")


def snr_fn(x):
    from HDR_model_hybrid_Teacher import estimate_local_snr_map
    return estimate_local_snr_map(x, window_size=5)


@pytest.fixture
def packed():
    g = torch.Generator().manual_seed(1)
    return torch.rand((1, 4, 40, 56), generator=g)


class TestConfigValidation:
    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            InferenceConfig(mode="sliding")

    def test_a_tile_that_breaks_the_encoder_alignment_is_rejected(self):
        """Three PixelUnshuffle(2) stages need packed dims divisible by 8."""
        with pytest.raises(ValueError, match="divisible by"):
            InferenceConfig(mode="tiled", tile=100)

    def test_overlap_at_or_above_the_tile_is_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            InferenceConfig(mode="tiled", tile=64, overlap=64)

    def test_a_non_positive_tile_is_rejected(self):
        with pytest.raises(ValueError, match="tile must be positive"):
            InferenceConfig(tile=0)


class TestModelOutput:
    def test_a_triple_is_unpacked(self, packed):
        out = call_model(local_model, packed, snr_fn(packed))
        assert out.num_experts == 2 and out.gates is not None

    def test_a_bare_tensor_becomes_a_single_output(self, packed):
        out = call_model(tensor_model, packed, snr_fn(packed))
        assert out.num_experts == 0 and out.experts is None

    def test_a_model_output_passes_through(self, packed):
        original = ModelOutput(torch.rand(1, 3, 8, 8))
        assert call_model(lambda a, b: original, packed, None) is original

    def test_an_unexpected_tuple_length_is_reported(self, packed):
        with pytest.raises(ValueError, match="expected 1 or 3"):
            call_model(lambda a, b: (a, a), packed, None)

    def test_an_unexpected_type_is_reported(self, packed):
        with pytest.raises(TypeError, match="expected a tensor"):
            call_model(lambda a, b: "hello", packed, None)

    def test_detach_cpu_moves_everything(self, packed):
        out = call_model(local_model, packed, snr_fn(packed)).detach_cpu()
        assert out.blended.device.type == "cpu"
        assert out.experts.device.type == "cpu"


class TestFullInference:
    def test_output_is_at_sensor_resolution(self, packed):
        out = infer_full(local_model, packed, snr_fn,
                         InferenceConfig(mode="full", amp_dtype=None))
        assert out.blended.shape == (1, 3, 80, 112)

    def test_it_pads_and_crops_back_to_the_true_size(self):
        """Encoder alignment must not change the output dimensions."""
        x = torch.rand(1, 4, 30, 42)          # neither dim divisible by 8
        out = infer_full(local_model, x, snr_fn,
                         InferenceConfig(mode="full", amp_dtype=None))
        assert out.blended.shape == (1, 3, 60, 84)

    def test_experts_and_gates_are_cropped_consistently(self):
        x = torch.rand(1, 4, 30, 42)
        out = infer_full(local_model, x, snr_fn,
                         InferenceConfig(mode="full", amp_dtype=None))
        assert out.experts.shape[-2:] == out.blended.shape[-2:]
        assert out.gates.shape[-2:] == out.blended.shape[-2:]

    def test_it_runs_on_the_cpu(self, packed):
        """
        The test script hardcodes autocast(device_type='cuda'), which
        throws on a login node. This path must not.
        """
        out = infer_full(local_model, packed, snr_fn)
        assert torch.isfinite(out.blended).all()


class TestTiledMatchesFull:
    @pytest.mark.parametrize("tile,overlap", [(16, 4), (16, 8), (24, 8),
                                              (32, 8), (64, 16)])
    def test_tiled_reproduces_full_frame_inference(self, packed, tile, overlap):
        full = infer_full(local_model, packed, snr_fn,
                          InferenceConfig(mode="full", amp_dtype=None))
        tiled = infer_tiled(local_model, packed, snr_fn,
                            InferenceConfig(mode="tiled", tile=tile,
                                            overlap=overlap, amp_dtype=None))
        assert torch.allclose(tiled.blended, full.blended, atol=1e-5)

    @pytest.mark.parametrize("tile,overlap", [(16, 4), (24, 8)])
    def test_experts_and_gates_also_match(self, packed, tile, overlap):
        full = infer_full(local_model, packed, snr_fn,
                          InferenceConfig(mode="full", amp_dtype=None))
        tiled = infer_tiled(local_model, packed, snr_fn,
                            InferenceConfig(mode="tiled", tile=tile,
                                            overlap=overlap, amp_dtype=None))
        assert torch.allclose(tiled.experts, full.experts, atol=1e-5)
        assert torch.allclose(tiled.gates, full.gates, atol=1e-5)

    @pytest.mark.parametrize("h,w", [(40, 56), (33, 41), (17, 19), (8, 8)])
    def test_every_pixel_is_covered_for_awkward_sizes(self, h, w):
        """
        The last tile must land flush with the far edge. A stride-based
        loop that stops early leaves a strip at zero, which shows up as
        both a shape mismatch and a non-finite division.
        """
        x = torch.rand(1, 4, h, w)
        out = infer_tiled(local_model, x, snr_fn,
                          InferenceConfig(mode="tiled", tile=16, overlap=4,
                                          amp_dtype=None))
        assert out.blended.shape == (1, 3, h * 2, w * 2)
        assert torch.isfinite(out.blended).all()

    def test_a_tile_larger_than_the_image_still_works(self):
        x = torch.rand(1, 4, 16, 16)
        out = infer_tiled(local_model, x, snr_fn,
                          InferenceConfig(mode="tiled", tile=64, overlap=16,
                                          amp_dtype=None))
        assert out.blended.shape == (1, 3, 32, 32)

    def test_zero_overlap_still_covers_the_image(self, packed):
        out = infer_tiled(local_model, packed, snr_fn,
                          InferenceConfig(mode="tiled", tile=16, overlap=0,
                                          amp_dtype=None))
        assert out.blended.shape == (1, 3, 80, 112)

    def test_a_batch_is_refused_rather_than_silently_wrong(self):
        x = torch.rand(2, 4, 32, 32)
        with pytest.raises(ValueError, match="one image at a time"):
            infer_tiled(local_model, x, snr_fn,
                        InferenceConfig(mode="tiled", tile=16, overlap=4))

    def test_a_model_without_routing_tiles_too(self, packed):
        out = infer_tiled(tensor_model, packed, snr_fn,
                          InferenceConfig(mode="tiled", tile=16, overlap=4,
                                          amp_dtype=None))
        assert out.experts is None and out.gates is None


class TestBlendWindow:
    def test_the_interior_is_flat(self):
        w = blend_window(16, 4)
        assert float(w[0, 0, 8, 8]) == 1.0

    def test_it_tapers_toward_a_shared_edge(self):
        w = blend_window(16, 4)[0, 0, 8]
        assert float(w[0]) < float(w[1]) < float(w[3]) < float(w[8])

    def test_the_taper_is_disabled_on_image_boundaries(self):
        """
        Otherwise the border pixels are reconstructed from a near-zero
        weight, and normalisation amplifies whatever is there.
        """
        w = blend_window(16, 4, top=True, left=True)
        assert float(w[0, 0, 0, 0]) == 1.0

    def test_it_is_strictly_positive_everywhere(self):
        """A zero weight anywhere is a division waiting to happen."""
        assert float(blend_window(16, 4).min()) > 0.0

    def test_overlapping_windows_sum_to_a_constant_in_the_interior(self):
        """The partition-of-unity property the seamless blend relies on."""
        size, taper, stride = 16, 8, 8
        total = torch.zeros(48)
        for start in range(0, 48 - size + 1, stride):
            w = blend_window(size, taper)[0, 0, 0]
            total[start:start + size] += w
        interior = total[size:48 - size]
        assert float(interior.max() - interior.min()) < 1e-5

    def test_an_impossible_taper_is_rejected(self):
        with pytest.raises(ValueError, match="taper"):
            blend_window(8, 8)

    def test_a_non_positive_size_is_rejected(self):
        with pytest.raises(ValueError, match="size"):
            blend_window(0, 0)


class TestAutocast:
    def test_it_is_a_no_op_on_the_cpu(self):
        with autocast_context(torch.device("cpu")):
            assert (torch.rand(4, 4) @ torch.rand(4, 4)).dtype == torch.float32

    def test_none_dtype_disables_it(self):
        ctx = autocast_context(torch.device("cpu"), None)
        with ctx:
            pass

    @pytest.mark.gpu
    def test_it_engages_on_a_gpu(self, gpu_device):
        device = gpu_device
        with autocast_context(device, torch.bfloat16):
            out = torch.rand(8, 8, device=device) @ torch.rand(8, 8, device=device)
        assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("kind", ["cuda", "xpu"])
    def test_it_names_the_devices_own_backend(self, kind, monkeypatch):
        """
        The device_type handed to torch.amp.autocast has to be the backend
        the tensor is on. A literal 'cuda' raises on Aurora's Intel GPUs, and
        no CI host has both backends, so the request itself is recorded here
        rather than the effect.
        """
        seen = []

        def record(*args, **kwargs):
            seen.append(kwargs.get("device_type", args[0] if args else None))
            return contextlib.nullcontext()

        monkeypatch.setattr(torch.amp, "autocast", record)
        with autocast_context(torch.device(f"{kind}:0"), torch.bfloat16):
            pass
        assert seen == [kind]


class TestDispatch:
    def test_infer_selects_the_configured_mode(self, packed):
        a = infer(local_model, packed, snr_fn,
                  InferenceConfig(mode="full", amp_dtype=None))
        b = infer(local_model, packed, snr_fn,
                  InferenceConfig(mode="tiled", tile=16, overlap=4,
                                  amp_dtype=None))
        assert torch.allclose(a.blended, b.blended, atol=1e-5)

    def test_the_default_config_runs(self, packed):
        assert infer(local_model, packed, snr_fn).blended.shape == (1, 3, 80, 112)
