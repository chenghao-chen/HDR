"""
Classical demosaicing — `hdr_baselines.demosaic`.

Why this file matters
─────────────────────
A broken baseline is worse than no baseline: it makes the model under
test look good for free, and nothing about the output looks wrong. So
these are pinned by properties only a correct implementation has.

The sharpest of them: **bilinear and Malvar must reproduce a linear ramp
exactly.** Both are exact for a signal that is locally linear, so any
mis-indexed kernel, wrong mask or misaligned CFA phase shows up
immediately as a finite PSNR where 60+ dB is expected. It also catches
border handling, which is the part that was wrong first: padding the
*sparse* colour planes replicates their structural zeros, so every edge
pixel interpolates from invented black. The mosaic must be padded
instead, by an even number, so the CFA phase survives the pad.
"""

import math

import pytest
import torch

from hdr_data.bayer import CFA_PATTERNS, rgb_to_packed
from hdr_baselines.demosaic import (
    BilinearDemosaic,
    GBTFDemosaic,
    MalvarDemosaic,
    NearestDemosaic,
    demosaic_bilinear,
    demosaic_malvar,
)
from hdr_eval.metrics import psnr

ALL_PATTERNS = tuple(CFA_PATTERNS)
CLASSICAL = (NearestDemosaic, BilinearDemosaic, MalvarDemosaic)


def grid(h=64, w=64):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w),
                            indexing="ij")
    return yy, xx


@pytest.fixture
def ramp():
    """A linear ramp: bilinear and Malvar are exact on this."""
    yy, xx = grid()
    return torch.stack([xx, yy, (xx + yy) / 2]).unsqueeze(0)


@pytest.fixture
def smooth():
    yy, xx = grid()
    return torch.stack([
        0.5 + 0.4 * torch.sin(6 * math.pi * xx),
        0.5 + 0.4 * torch.cos(5 * math.pi * yy),
        0.5 + 0.3 * torch.sin(4 * math.pi * (xx + yy)),
    ]).unsqueeze(0).clamp(0, 1)


@pytest.fixture
def chirp():
    """High frequency: the content demosaicing actually exists for."""
    yy, xx = grid()
    out = torch.stack([0.5 + 0.45 * torch.sin(60 * math.pi * xx ** 2)] * 3)
    out[1] = 0.5 + 0.45 * torch.sin(60 * math.pi * xx ** 2 + 0.5)
    return out.unsqueeze(0)


class TestExactness:
    @pytest.mark.parametrize("fn", [demosaic_bilinear, demosaic_malvar])
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_a_constant_image_is_reproduced_exactly(self, fn, pattern):
        flat = torch.full((1, 3, 32, 32), 0.4)
        packed = rgb_to_packed(flat, pattern)
        from hdr_data.bayer import unpack
        out = fn(unpack(packed), pattern)
        assert torch.allclose(out, flat, atol=1e-5)

    @pytest.mark.parametrize("fn", [demosaic_bilinear, demosaic_malvar])
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_a_linear_ramp_is_reproduced_nearly_exactly(self, fn, pattern, ramp):
        """
        Both methods are exact for a locally linear signal. Anything below
        ~50 dB here means a kernel, a mask or the CFA phase is wrong.
        """
        from hdr_data.bayer import unpack
        packed = rgb_to_packed(ramp, pattern)
        out = fn(unpack(packed), pattern)
        assert psnr(out, ramp) > 50.0

    def test_the_border_is_not_reconstructed_from_invented_zeros(self, smooth):
        """
        The bug this guards: padding the sparse planes rather than the
        mosaic makes every edge pixel interpolate against black. Interior
        and full-frame scores then diverge sharply.
        """
        from hdr_data.bayer import unpack
        packed = rgb_to_packed(smooth, "BGGR")
        out = demosaic_bilinear(unpack(packed), "BGGR")
        full = psnr(out, smooth)
        interior = psnr(out[:, :, 8:-8, 8:-8], smooth[:, :, 8:-8, 8:-8])
        assert full > interior - 10.0


class TestOrdering:
    def test_malvar_beats_bilinear_on_high_frequency_content(self, chirp):
        """
        Malvar's gradient correction buys nothing on a smooth signal and
        several dB on a hard one, which is the whole claim of the method.
        """
        from hdr_data.bayer import unpack
        packed = unpack(rgb_to_packed(chirp, "BGGR"))
        assert psnr(demosaic_malvar(packed, "BGGR"), chirp) > \
               psnr(demosaic_bilinear(packed, "BGGR"), chirp) + 1.0

    def test_gbtf_beats_malvar_on_high_frequency_content(self, chirp):
        packed = rgb_to_packed(chirp, "BGGR")
        gbtf = GBTFDemosaic()(packed)[0]
        malvar = MalvarDemosaic()(packed)[0]
        assert psnr(gbtf, chirp) > psnr(malvar, chirp)

    def test_nearest_is_the_floor_on_high_frequency_content(self, chirp):
        packed = rgb_to_packed(chirp, "BGGR")
        near = psnr(NearestDemosaic()(packed)[0], chirp)
        for cls in (BilinearDemosaic, MalvarDemosaic, GBTFDemosaic):
            assert psnr(cls()(packed)[0], chirp) > near


class TestModelInterface:
    @pytest.mark.parametrize("cls", CLASSICAL + (GBTFDemosaic,))
    def test_the_shapes_match_the_project_contract(self, cls):
        model = cls()
        packed = torch.rand(2, 4, 16, 16)
        blended, experts, gates = model(packed, None)
        assert blended.shape == (2, 3, 32, 32)
        assert experts.shape == (2, 1, 3, 32, 32)
        assert gates.shape == (2, 1, 32, 32)

    @pytest.mark.parametrize("cls", CLASSICAL + (GBTFDemosaic,))
    def test_a_single_expert_reports_a_unit_gate(self, cls):
        """Keeps the CSV schema uniform across routed and unrouted models."""
        _, experts, gates = cls()(torch.rand(1, 4, 16, 16), None)
        assert float(gates.min()) == 1.0 and experts.shape[1] == 1

    @pytest.mark.parametrize("cls", CLASSICAL + (GBTFDemosaic,))
    def test_output_is_finite_and_in_range(self, cls):
        out, _, _ = cls()(torch.rand(1, 4, 32, 32), None)
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    @pytest.mark.parametrize("cls", CLASSICAL)
    def test_they_are_marked_untrainable(self, cls):
        assert cls().trainable is False

    def test_a_wrongly_shaped_input_is_rejected(self):
        with pytest.raises(ValueError, match=r"packed \[B, 4, h, w\]"):
            BilinearDemosaic()(torch.rand(1, 3, 16, 16), None)

    @pytest.mark.parametrize("cls", CLASSICAL)
    def test_they_work_at_every_cfa_phase(self, cls):
        for pattern in ALL_PATTERNS:
            out, _, _ = cls(pattern=pattern)(torch.rand(1, 4, 16, 16), None)
            assert out.shape == (1, 3, 32, 32)

    def test_gbtf_refuses_a_pattern_it_cannot_handle(self):
        """
        DifferentiableGBTF_BGGR hardcodes the phase; accepting another one
        would silently swap red and blue.
        """
        with pytest.raises(ValueError, match="BGGR-only"):
            GBTFDemosaic(pattern="RGGB")


class TestFunctionalInterface:
    def test_a_wrong_mosaic_shape_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
            demosaic_bilinear(torch.rand(1, 3, 16, 16))

    def test_malvar_rejects_a_wrong_mosaic_shape(self):
        with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
            demosaic_malvar(torch.rand(1, 3, 16, 16))

    def test_output_dtype_follows_the_input(self):
        from hdr_data.bayer import unpack
        packed = torch.rand(1, 4, 8, 8, dtype=torch.float64)
        assert demosaic_bilinear(unpack(packed)).dtype == torch.float64

    def test_batches_are_handled_independently(self, smooth):
        from hdr_data.bayer import unpack
        packed = unpack(rgb_to_packed(torch.cat([smooth, smooth * 0.5]), "BGGR"))
        out = demosaic_malvar(packed, "BGGR")
        assert out.shape[0] == 2
        assert not torch.allclose(out[0], out[1])


class TestGBTFDeviceHandling:
    """
    GBTF is the only demosaicer here that is a Module rather than a function.
    bilinear and Malvar build their kernels on ``mosaic.device`` every call, so
    they follow the input for free; GBTF's kernels are registered buffers that
    live wherever the instance was built — and the registry in
    hdr_baselines.pipelines caches exactly one instance, lazily, on the CPU.

    Feeding that cached instance a CUDA tensor used to raise "Input type
    (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the
    same". It was not a corner case: BenchmarkRunner demosaics the on-device
    ground truth with gt_demosaic="gbtf" by default, so every GPU benchmark
    run died on its first frame.
    """

    def test_dtype_follows_the_input(self):
        from hdr_baselines.demosaic import GBTFDemosaic
        m = GBTFDemosaic()
        out = m.demosaic(torch.rand(1, 1, 16, 16, dtype=torch.float64))
        assert torch.isfinite(out).all()

    @pytest.mark.gpu
    def test_the_cached_instance_follows_the_input_to_the_gpu(self, gpu_device):
        """
        The registry path, which is the one the benchmark actually uses: a
        CPU-built cached module must accept a GPU tensor and return a GPU
        result matching the CPU one.
        """
        from hdr_baselines.pipelines import DEMOSAIC_FUNCTIONS
        fn = DEMOSAIC_FUNCTIONS["gbtf"]
        mosaic = torch.rand(1, 1, 32, 32)

        cpu_out = fn(mosaic, "BGGR")
        gpu_out = fn(mosaic.to(gpu_device), "BGGR")

        assert gpu_out.device.type == gpu_device.type
        assert torch.allclose(cpu_out, gpu_out.cpu(), atol=1e-5)

    @pytest.mark.gpu
    def test_it_still_works_on_the_cpu_after_a_gpu_call(self, gpu_device):
        """
        The instance is shared and moves in place, so a GPU call must not
        strand it: the next CPU frame has to keep working. Mixed-device
        evaluation in one process is the normal case, not an exotic one.
        """
        from hdr_baselines.pipelines import DEMOSAIC_FUNCTIONS
        fn = DEMOSAIC_FUNCTIONS["gbtf"]
        mosaic = torch.rand(1, 1, 32, 32)

        fn(mosaic.to(gpu_device), "BGGR")
        back = fn(mosaic, "BGGR")
        assert back.device.type == "cpu"
        assert torch.isfinite(back).all()
