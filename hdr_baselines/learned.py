"""
hdr_baselines/learned.py — trainable reference architectures
=============================================================

Four small joint denoise+demosaic networks, all with the project's
signature, all trainable from scratch. They exist so that "the MoE
teacher is better" can be a measured claim against comparable networks
rather than only against classical filters — a 21M-parameter model
beating a bilateral filter says very little.

    ``DnCNNJDD``        A residual conv stack (Zhang et al.'s DnCNN,
        adapted to output RGB at 2x resolution). No downsampling: the
        receptive field is depth-limited, which is the point — it shows
        how much of the teacher's advantage comes from the U-Net's global
        context rather than its capacity.
    ``UNetJDD``         A three-level U-Net with skip connections. The
        standard restoration workhorse and the fairest architectural
        comparison to the teacher's trunk.
    ``DemosaicNetJDD``  Gharbi et al.'s arrangement: process at packed
        resolution, then re-inject the *measured* samples through a
        sparse-RGB skip so the network only has to predict what the sensor
        did not see. Notably strong on colour fidelity for its size.
    ``RestormerLiteJDD``The project's own RestormerBlock at a fraction of
        the teacher's width and depth, to separate "transformer blocks
        help" from "19M parameters help".

Every one takes ``(x, snr_map)`` and ignores ``snr_map`` unless
``use_snr=True``, in which case it is concatenated to the input — an
ablation on whether the routing signal is useful outside a routed model.

None of these ship with weights. Scoring one straight from its
initialisation is a shape and throughput smoke test, not a result; the
runner marks untrained models in its output so the distinction survives
into the report.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdr_data.bayer import packed_to_sparse_rgb

from .interface import BaselineModel

__all__ = [
    "LearnedBaseline",
    "DnCNNJDD",
    "UNetJDD",
    "DemosaicNetJDD",
    "RestormerLiteJDD",
]

#: Matches HDR_model_hybrid_Teacher._CLAMP_EPS: outputs stay strictly
#: positive so a downstream log or mu-law curve never sees a zero.
_CLAMP_EPS = 1.0 / (2 ** 20 - 1)


class LearnedBaseline(BaselineModel):
    """
    Shared plumbing for the trainable references.

    Subclasses implement :meth:`body`, taking the (possibly
    SNR-augmented) input at packed resolution and returning
    ``out_channels * 4`` channels, which this class pixel-shuffles to RGB
    at sensor resolution.
    """

    trainable = True

    def __init__(self, name: Optional[str] = None, pattern: str = "BGGR",
                 use_snr: bool = False, out_channels: int = 3):
        super().__init__(name=name, pattern=pattern)
        self.use_snr = bool(use_snr)
        self.out_channels = int(out_channels)
        self.in_channels = 5 if self.use_snr else 4

    @property
    def num_parameters(self) -> int:
        """Total parameter count — the number a comparison has to state."""
        return sum(p.numel() for p in self.parameters())

    def body(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_channels, h, w] -> [B, out_channels*4, h, w]."""
        raise NotImplementedError

    def predict_rgb(self, x: torch.Tensor,
                    snr_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_snr:
            if snr_map is None:
                raise ValueError(
                    f"{self.name} was built with use_snr=True but no SNR map "
                    f"was passed.")
            x = torch.cat([x, snr_map], dim=1)
        out = F.pixel_shuffle(self.body(x), 2)
        out = out.clamp(min=_CLAMP_EPS)
        if not self.training:
            out = out.clamp(max=1.0)
        return out

    def extra_repr(self) -> str:
        return (f"name='{self.name}', params={self.num_parameters/1e6:.2f}M, "
                f"use_snr={self.use_snr}")


def _conv(in_ch: int, out_ch: int, k: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2)


class DnCNNJDD(LearnedBaseline):
    """
    Flat residual conv stack — depth-limited receptive field, no pooling.

    ``depth`` 3x3 layers give a receptive field of ``2*depth + 1`` packed
    pixels, so the default 12 layers see 25x25 packed (50x50 sensor).
    That is deliberately less context than the U-Net: comparing the two
    isolates how much the global context is worth.
    """

    name = "dncnn"

    def __init__(self, width: int = 64, depth: int = 12, **kwargs):
        super().__init__(**kwargs)
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        layers = [_conv(self.in_channels, width), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [_conv(width, width),
                       nn.BatchNorm2d(width),
                       nn.ReLU(inplace=True)]
        layers += [_conv(width, self.out_channels * 4)]
        self.net = nn.Sequential(*layers)
        self.width, self.depth = width, depth

    def body(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UNetBlock(nn.Module):
    """Two 3x3 convolutions with GELU — one U-Net rung."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv(in_ch, out_ch), nn.GELU(),
            _conv(out_ch, out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetJDD(LearnedBaseline):
    """
    Three-level encoder/decoder with skip connections.

    Downsampling is PixelUnshuffle rather than striding or pooling, which
    is lossless — the same choice the teacher makes, and it matters for a
    task where the input is a sub-sampled lattice to begin with.
    """

    name = "unet"

    def __init__(self, base: int = 32, levels: int = 3, **kwargs):
        super().__init__(**kwargs)
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        if base < 4:
            raise ValueError(f"base must be >= 4, got {base}")

        self.levels = levels
        self.stem = _conv(self.in_channels, base)

        widths = [base * (2 ** i) for i in range(levels + 1)]
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(levels):
            self.encoders.append(_UNetBlock(widths[i], widths[i]))
            # PixelUnshuffle(2) multiplies channels by 4; a 1x1 halves
            # them back to the usual doubling.
            self.downs.append(nn.Conv2d(widths[i] * 4, widths[i + 1], 1))

        self.middle = _UNetBlock(widths[levels], widths[levels])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(levels)):
            self.ups.append(nn.Conv2d(widths[i + 1], widths[i] * 4, 1))
            self.decoders.append(_UNetBlock(widths[i] * 2, widths[i]))

        self.head = _conv(base, self.out_channels * 4)
        self.multiple_of = 2 ** levels

    def body(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        m = self.multiple_of
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

        z = self.stem(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            z = enc(z)
            skips.append(z)
            z = down(F.pixel_unshuffle(z, 2))

        z = self.middle(z)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            z = F.pixel_shuffle(up(z), 2)
            z = dec(torch.cat([z, skip], dim=1))

        out = self.head(z)
        return out[..., :h, :w]


class DemosaicNetJDD(LearnedBaseline):
    """
    Gharbi-style: predict only what the sensor did not measure.

    The measured samples are re-injected as a sparse-RGB skip just before
    the head, so the network spends its capacity on interpolation and
    noise rather than on relearning the identity for the two thirds of
    pixels it was handed. On small budgets this is worth several dB.
    """

    name = "demosaicnet"

    def __init__(self, width: int = 64, depth: int = 10, **kwargs):
        super().__init__(**kwargs)
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        layers = [_conv(self.in_channels, width), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [_conv(width, width), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        # +12: the sparse RGB skip, pixel-unshuffled back to packed res.
        self.head = _conv(width + 12, self.out_channels * 4)
        self.width, self.depth = width, depth

    def body(self, x: torch.Tensor) -> torch.Tensor:
        # The skip is built from the CFA channels only, never the SNR map.
        sparse = packed_to_sparse_rgb(x[:, :4], self.pattern)   # [B,3,2h,2w]
        skip = F.pixel_unshuffle(sparse, 2)                     # [B,12,h,w]
        z = self.trunk(x)
        return self.head(torch.cat([z, skip], dim=1))


class RestormerLiteJDD(LearnedBaseline):
    """
    The project's RestormerBlock, at a fraction of the teacher's size.

    One PixelUnshuffle stage, a few transformer blocks at the bottleneck,
    one PixelShuffle back. Same block, same attention, ~2% of the
    parameters — so a gap between this and the teacher is about scale,
    and a gap between this and :class:`UNetJDD` is about the block.
    """

    name = "restormer_lite"

    def __init__(self, dim: int = 32, num_blocks: int = 4, heads: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        from blocks_Restormer import RestormerBlock

        self.stem = _conv(self.in_channels, dim)
        # PixelUnshuffle(2) -> dim*4 at half resolution.
        self.blocks = nn.Sequential(*[
            RestormerBlock(dim=dim * 4, num_heads=heads)
            for _ in range(num_blocks)
        ])
        self.fuse = nn.Conv2d(dim * 4, dim * 4, kernel_size=1)
        self.head = _conv(dim, self.out_channels * 4)
        self.dim = dim

    def body(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        pad_h, pad_w = (-h) % 2, (-w) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        z = self.stem(x)
        z = F.pixel_unshuffle(z, 2)
        z = self.fuse(self.blocks(z)) + z
        z = F.pixel_shuffle(z, 2)
        return self.head(z)[..., :h, :w]
