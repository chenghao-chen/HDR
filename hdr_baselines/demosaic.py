"""
hdr_baselines/demosaic.py — classical demosaicing references
=============================================================

Three reference demosaicers, in ascending order of how hard they are to
beat:

``NearestDemosaic``  Replicate each cell's samples. Useless as a method,
    invaluable as a floor: any learned model that does not clear it by a
    wide margin is broken.
``BilinearDemosaic``  Separable interpolation of each sparse plane. The
    textbook baseline; produces the zippering and false colour that every
    later method exists to remove.
``MalvarDemosaic``  Malvar-He-Cutler gradient-corrected linear
    interpolation (ICASSP 2004). Still linear, still five lines of
    convolution, and typically 3-5 dB better than bilinear — this is the
    honest classical bar.
``GBTFDemosaic``  Wraps the project's own DifferentiableGBTF_BGGR so the
    directional method already in the training loop can be scored on the
    same axis as the rest.

The first three are pattern-generic: they read the CFA layout from
``hdr_data.bayer`` rather than assuming BGGR, so they can be pointed at
the video and Kalantari loaders' virtual sensors too. GBTF is BGGR-only
by construction and raises if asked for anything else.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from hdr_data.bayer import (
    canonical_pattern,
    cfa_masks,
    mosaic_to_sparse_rgb,
    pattern_colors,
    unpack,
)

from .interface import BaselineModel

__all__ = [
    "NearestDemosaic",
    "BilinearDemosaic",
    "MalvarDemosaic",
    "GBTFDemosaic",
    "demosaic_bilinear",
    "demosaic_malvar",
]


#: Mosaic padding used before interpolation. Must be EVEN: reflect-padding
#: a mosaic by an even number maps column -1 onto column 1, which carries
#: the same colour filter, so the CFA phase survives the pad. Pad by an odd
#: amount (or pad the sparse planes instead of the mosaic, which replicates
#: the structural zeros) and every border pixel is interpolated from the
#: wrong colours.
_MOSAIC_PAD = 2


def _pad_mosaic(mosaic: torch.Tensor, pad: int = _MOSAIC_PAD) -> torch.Tensor:
    """Reflect-pad a mosaic by an even amount, preserving the CFA phase."""
    if pad % 2:
        raise ValueError(f"Mosaic padding must be even, got {pad}")
    if pad == 0:
        return mosaic
    h, w = mosaic.shape[-2], mosaic.shape[-1]
    mode = "reflect" if (pad < h and pad < w) else "replicate"
    return F.pad(mosaic, (pad, pad, pad, pad), mode=mode)


def _conv_valid(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Single-channel convolution with no padding of its own."""
    return F.conv2d(x, kernel)


def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Crop x back to (h, w) about its centre."""
    top = (x.shape[-2] - h) // 2
    left = (x.shape[-1] - w) // 2
    return x[..., top:top + h, left:left + w]


def _k(values, device, dtype) -> torch.Tensor:
    """Build a [1, 1, k, k] kernel from nested lists."""
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Functional cores
# ─────────────────────────────────────────────────────────────────────────────

def demosaic_bilinear(mosaic: torch.Tensor,
                      pattern: str = "BGGR") -> torch.Tensor:
    """
    Bilinear demosaicing. [B, 1, H, W] -> [B, 3, H, W].

    Each sparse plane is interpolated by the kernel that exactly averages
    its own sampling lattice: a 4-neighbour cross for green (half
    density), a full 3x3 tent for red and blue (quarter density).
    """
    if mosaic.dim() != 4 or mosaic.shape[1] != 1:
        raise ValueError(
            f"Expected [B, 1, H, W] mosaic, got {tuple(mosaic.shape)}")
    device, dtype = mosaic.device, mosaic.dtype
    h, w = mosaic.shape[-2], mosaic.shape[-1]
    padded = _pad_mosaic(mosaic)
    sparse = mosaic_to_sparse_rgb(padded, pattern)

    k_g = _k([[0., 1., 0.], [1., 4., 1.], [0., 1., 0.]], device, dtype) / 4.0
    k_rb = _k([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device, dtype) / 4.0

    out = []
    for idx, kernel in ((0, k_rb), (1, k_g), (2, k_rb)):
        conv = _conv_valid(sparse[:, idx:idx + 1], kernel)
        out.append(_center_crop(conv, h, w))
    return torch.cat(out, dim=1)


def demosaic_malvar(mosaic: torch.Tensor,
                    pattern: str = "BGGR") -> torch.Tensor:
    """
    Malvar-He-Cutler gradient-corrected linear demosaicing.

    The idea in one line: interpolate a missing colour bilinearly, then
    correct it by a fraction of the *Laplacian of the channel that was
    measured at that pixel*. Edges are correlated across colour channels,
    so the measured channel's second derivative is a good estimate of the
    missing one's — which is why five fixed 5x5 kernels get most of the
    way to the directional methods.

    [B, 1, H, W] -> [B, 3, H, W], any CFA phase.
    """
    if mosaic.dim() != 4 or mosaic.shape[1] != 1:
        raise ValueError(
            f"Expected [B, 1, H, W] mosaic, got {tuple(mosaic.shape)}")
    pattern = canonical_pattern(pattern)
    device, dtype = mosaic.device, mosaic.dtype
    out_h, out_w = mosaic.shape[-2], mosaic.shape[-1]
    mosaic = _pad_mosaic(mosaic)
    h, w = mosaic.shape[-2], mosaic.shape[-1]

    # The five Malvar kernels, all scaled by 1/8.
    k_g_at_rb = _k([
        [0., 0., -1., 0., 0.],
        [0., 0., 2., 0., 0.],
        [-1., 2., 4., 2., -1.],
        [0., 0., 2., 0., 0.],
        [0., 0., -1., 0., 0.],
    ], device, dtype) / 8.0

    # R at a green pixel whose row also carries red (and symmetrically B).
    k_rb_at_g_same_row = _k([
        [0., 0., 0.5, 0., 0.],
        [0., -1., 0., -1., 0.],
        [-1., 4., 5., 4., -1.],
        [0., -1., 0., -1., 0.],
        [0., 0., 0.5, 0., 0.],
    ], device, dtype) / 8.0

    # The same, rotated: the green pixel's *column* carries red.
    k_rb_at_g_same_col = k_rb_at_g_same_row.transpose(-2, -1).contiguous()

    # R at a blue pixel (and B at a red pixel).
    k_rb_at_br = _k([
        [0., 0., -1.5, 0., 0.],
        [0., 2., 0., 2., 0.],
        [-1.5, 0., 6., 0., -1.5],
        [0., 2., 0., 2., 0.],
        [0., 0., -1.5, 0., 0.],
    ], device, dtype) / 8.0

    # The kernels are 5x5 and the mosaic carries a 2px pad, so a valid
    # convolution comes back at exactly the pre-pad size; re-padding it
    # keeps every intermediate on the padded grid, and the final crop
    # discards those synthetic border pixels.
    def conv(kernel):
        return F.pad(_conv_valid(mosaic, kernel), (2, 2, 2, 2), mode="replicate")

    masks = cfa_masks(pattern, h, w, device=device, dtype=torch.bool)
    colors = pattern_colors(pattern)
    # Cell row occupied by red; a green pixel on that row shares it.
    red_cell = colors.index("R")
    red_row = red_cell // 2

    g_full = torch.where(masks["G"].unsqueeze(0), mosaic, conv(k_g_at_rb))

    # Split the two greens by whether they sit on the red row or the blue one.
    rows = torch.arange(h, device=device).view(1, 1, h, 1)
    on_red_row = (rows % 2) == red_row
    g_red_row = masks["G"].unsqueeze(0) & on_red_row
    g_blue_row = masks["G"].unsqueeze(0) & ~on_red_row

    conv_same_row = conv(k_rb_at_g_same_row)
    conv_same_col = conv(k_rb_at_g_same_col)
    conv_cross = conv(k_rb_at_br)

    r_full = torch.where(masks["R"].unsqueeze(0), mosaic, torch.zeros_like(mosaic))
    r_full = torch.where(g_red_row, conv_same_row, r_full)
    r_full = torch.where(g_blue_row, conv_same_col, r_full)
    r_full = torch.where(masks["B"].unsqueeze(0), conv_cross, r_full)

    b_full = torch.where(masks["B"].unsqueeze(0), mosaic, torch.zeros_like(mosaic))
    # Blue varies along the row that carries blue, which is the other one.
    b_full = torch.where(g_blue_row, conv_same_row, b_full)
    b_full = torch.where(g_red_row, conv_same_col, b_full)
    b_full = torch.where(masks["R"].unsqueeze(0), conv_cross, b_full)

    rgb = torch.cat([r_full, g_full, b_full], dim=1)
    return _center_crop(rgb, out_h, out_w)


# ─────────────────────────────────────────────────────────────────────────────
# Model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _DemosaicOnly(BaselineModel):
    """Shared plumbing: unpack to a mosaic, demosaic, clamp."""

    def predict_rgb(self, x: torch.Tensor,
                    snr_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        mosaic = unpack(x)
        return self.demosaic(mosaic).clamp(0.0, 1.0)

    def demosaic(self, mosaic: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NearestDemosaic(_DemosaicOnly):
    """
    Replicate each 2x2 cell's samples across the cell. The floor.

    Every output pixel in a cell gets that cell's red, its blue, and the
    mean of its two greens — so the result is a correct-looking image at
    half the true resolution, which is exactly what makes it a useful
    lower bound.
    """

    name = "nearest"

    def demosaic(self, mosaic: torch.Tensor) -> torch.Tensor:
        from hdr_data.bayer import pack, packed_to_half_rgb
        half = packed_to_half_rgb(pack(mosaic), self.pattern)   # [B, 3, h, w]
        return F.interpolate(half, scale_factor=2, mode="nearest")


class BilinearDemosaic(_DemosaicOnly):
    """Bilinear interpolation of each sparse colour plane."""

    name = "bilinear"

    def demosaic(self, mosaic: torch.Tensor) -> torch.Tensor:
        return demosaic_bilinear(mosaic, self.pattern)


class MalvarDemosaic(_DemosaicOnly):
    """Malvar-He-Cutler gradient-corrected linear interpolation."""

    name = "malvar"

    def demosaic(self, mosaic: torch.Tensor) -> torch.Tensor:
        return demosaic_malvar(mosaic, self.pattern)


class GBTFDemosaic(_DemosaicOnly):
    """
    The project's own differentiable GBTF, as a scorable baseline.

    GBTF is BGGR-only — the kernels and the checkerboard sign pattern are
    hardcoded for that phase — so this refuses any other pattern rather
    than silently producing colour-swapped output.
    """

    name = "gbtf"

    def __init__(self, pattern: str = "BGGR", **kwargs):
        if canonical_pattern(pattern) != "BGGR":
            raise ValueError(
                f"GBTFDemosaic is BGGR-only (DifferentiableGBTF_BGGR "
                f"hardcodes the phase); got '{pattern}'. Use MalvarDemosaic "
                f"for other patterns.")
        super().__init__(pattern="BGGR", **kwargs)
        from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR
        self.gbtf = DifferentiableGBTF_BGGR()
        self.gbtf.eval()
        for p in self.gbtf.parameters():
            p.requires_grad_(False)

    def demosaic(self, mosaic: torch.Tensor) -> torch.Tensor:
        # Follow the input's device. Unlike the bilinear/Malvar functions,
        # which build their kernels on mosaic.device every call, GBTF is a
        # Module whose buffers live wherever it was constructed — the CPU,
        # since the registry caches one instance lazily. Feeding it a CUDA
        # tensor otherwise raises "Input type (torch.cuda.FloatTensor) and
        # weight type (torch.FloatTensor) should be the same", which is what
        # every GPU benchmark run hit: BenchmarkRunner demosaics the on-device
        # ground truth with gt_demosaic="gbtf" by default.
        buf = next(self.gbtf.buffers())
        if buf.device != mosaic.device:
            self.gbtf.to(mosaic.device)
            buf = next(self.gbtf.buffers())
        return self.gbtf(mosaic.to(buf.dtype))
