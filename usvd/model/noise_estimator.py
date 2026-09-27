"""Per-pixel noise standard deviation estimator."""
from __future__ import annotations

import torch.nn as nn

__all__ = ["NoiseEstimator"]


class NoiseEstimator(nn.Module):
    """
    Lightweight CNN predicting a per-pixel sigma map in (0, inf).

    Softplus rather than ReLU on the output: the map is divided by in the
    data-fidelity weight W = 1/sigma^2, so a hard zero would make W infinite.
    Softplus is strictly positive and keeps a usable gradient near zero.
    """

    def __init__(self, in_channels: int = 1, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width, 1, 1), nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)
