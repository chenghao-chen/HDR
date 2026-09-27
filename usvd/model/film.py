"""Feature-wise linear modulation conditioned on the noise level."""
from __future__ import annotations

import torch.nn as nn

__all__ = ["FiLM"]


class FiLM(nn.Module):
    """
    Scale-and-shift features from a globally pooled sigma map.

    The sigma map is pooled to a single scalar per image, so this conditions
    on the overall noise level rather than per-pixel structure; the spatial
    variation is carried by the data-fidelity weight in the unrolled step.
    """

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(True), nn.Linear(hidden, 2 * channels)
        )
        self.channels = channels
        # Start near an identity transform (gamma ~ 0 in the 1+gamma form,
        # beta ~ 0) so a fresh model passes features through rather than
        # rescaling them at every one of the K iterations. Small std rather
        # than exact zero: a zero final weight would also zero the gradient
        # reaching the hidden layer through it, leaving this MLP untrainable
        # on the first step.
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features, sigma_map):
        s = self.pool(sigma_map).flatten(1)               # (B, 1)
        gamma, beta = self.mlp(s).chunk(2, dim=1)         # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * features + beta
