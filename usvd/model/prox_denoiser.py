"""One unrolled proximal step: a local branch and a non-local branch, fused."""
from __future__ import annotations

import torch
import torch.nn as nn

from .film import FiLM
from .nonlocal_attn import SparseNonLocalAttention

__all__ = ["ProxDenoiser"]


class ProxDenoiser(nn.Module):
    """
    The learned proximal operator applied after each gradient step.

    Local branch is a depthwise-separable block conditioned on the noise
    level; non-local branch aggregates similar patches from elsewhere in the
    image. The two are concatenated and mixed, normalised, and added back as
    a residual -- so at initialisation the step is close to the identity and
    the unrolled iteration starts from the data-fidelity solution rather
    than from noise.
    """

    def __init__(self, channels: int, hidden_mult: int = 2, patch_size: int = 8,
                 top_k: int = 16, d_model: int = 256, chunk_size: int = 1024):
        super().__init__()
        hidden = hidden_mult * channels

        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.film = FiLM(channels)

        self.nonlocal_attn = SparseNonLocalAttention(
            channels, patch_size=patch_size, top_k=top_k,
            d_model=d_model, chunk_size=chunk_size)
        self.nl_proj = nn.Conv2d(channels, channels, 1)

        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        # GroupNorm(1, C) is LayerNorm over the channel dim for NCHW data.
        self.norm = nn.GroupNorm(1, channels)

        # Near-identity at init, but deliberately NOT exactly zero. An exact
        # zero here makes the block a perfect identity -- and also makes
        # d(loss)/d(local) and d(loss)/d(nonlocal) exactly zero, so every
        # parameter in both branches receives no gradient at all on the
        # first step and can only start learning once fuse itself moves off
        # zero. This project has twice paid for that pattern (see ExpertHead
        # and NoiseGate in HDR_model_hybrid_Teacher.py). A small std keeps
        # the residual quiet across K iterations while giving every
        # parameter gradient from step 0.
        nn.init.normal_(self.fuse.weight, std=1e-3)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x, sigma_map):
        local = self.pw2(self.act(self.pw1(self.dwconv(x))))
        local = self.film(local, sigma_map)

        nonlocal_ = self.nl_proj(self.nonlocal_attn(x))

        fused = self.fuse(torch.cat([local, nonlocal_], dim=1))
        return self.norm(fused) + x
