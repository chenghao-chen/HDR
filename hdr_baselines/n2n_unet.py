"""
hdr_baselines/n2n_unet.py — the Noise2Noise U-Net, as a calibration reference
=============================================================================

Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean
Data", ICML 2018, Appendix A.1. This is the network behind the supervised
("N2C") raw-RGB SIDD baseline that later papers report -- 51.19 dB on the
SIDD validation blocks in Neighbor2Neighbor (Huang et al., CVPR 2021,
Table 2), where it is cited via Laine et al. 2019.

It is here for one purpose: to check that this project's SIDD pipeline (data
loading, normalisation, schedule, evaluation) can reproduce a published
number before any of our own models' SIDD results are compared against the
literature. If this network lands near 51.2 dB in our pipeline, the pipeline
is calibrated; if it does not, the gap is in the pipeline, not the models.

Faithful to the paper's table: 48-channel encoder with five 2x2 max-pools,
96-channel decoder with nearest-neighbour upsampling and skip
concatenation, LeakyReLU(0.1) everywhere except the linear output layer.
Laine et al.'s variant appends three 1x1 convolutions for its blind-spot
experiments; that change is specific to self-supervision and is omitted.

Five pools mean the input's packed height and width must be divisible by 32.

Two knobs extend it into a family for the efficiency study; at their defaults
(width=48, residual=False) the network is exactly the paper's, so calibration
checkpoints load unchanged:

* ``width`` scales every layer proportionally (encoder w, decoder 2w, final
  stages 4w/3 and 2w/3, matching 48/96/64/32 at w=48). FLOPs go roughly as w^2.
* ``residual`` adds the input to the output, so the layers model the noise
  rather than the image -- the DnCNN formulation, which costs one addition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["N2NUNet"]


def _conv(cin, cout, act=True):
    layers = [nn.Conv2d(cin, cout, 3, padding=1)]
    if act:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


class N2NUNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 width: int = 48, residual: bool = False, **_):
        super().__init__()
        self.num_experts = 1
        if residual and in_channels != out_channels:
            raise ValueError("residual needs in_channels == out_channels")
        self.residual = bool(residual)
        n, m, w = in_channels, out_channels, int(width)
        d, a, b = 2 * w, (4 * w) // 3, (2 * w) // 3
        self.enc0 = nn.Sequential(_conv(n, w), _conv(w, w))
        self.enc = nn.ModuleList([_conv(w, w) for _ in range(5)])  # conv2..6
        self.dec5 = nn.Sequential(_conv(2 * w, d), _conv(d, d))
        self.dec4 = nn.Sequential(_conv(d + w, d), _conv(d, d))
        self.dec3 = nn.Sequential(_conv(d + w, d), _conv(d, d))
        self.dec2 = nn.Sequential(_conv(d + w, d), _conv(d, d))
        self.dec1 = nn.Sequential(_conv(d + n, a), _conv(a, b),
                                  _conv(b, m, act=False))

    def forward(self, x, snr_map=None):
        skips = [x]
        h = F.max_pool2d(self.enc0(x), 2)          # pool1
        skips.append(h)
        for i in range(4):                           # conv2..5 + pool2..5
            h = F.max_pool2d(self.enc[i](h), 2)
            skips.append(h)
        h = self.enc[4](h)                           # conv6
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        h = self.dec5(torch.cat([up(h), skips[4]], 1))
        h = self.dec4(torch.cat([up(h), skips[3]], 1))
        h = self.dec3(torch.cat([up(h), skips[2]], 1))
        h = self.dec2(torch.cat([up(h), skips[1]], 1))
        out = self.dec1(torch.cat([up(h), skips[0]], 1))
        if self.residual:
            out = out + x
        if not self.training:
            out = out.clamp(0.0, 1.0)
        gates = torch.ones_like(out[:, :1])
        return out, out.unsqueeze(1), gates
