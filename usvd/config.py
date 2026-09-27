"""Hyperparameters for the unrolled denoiser."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["USVDConfig"]


@dataclass
class USVDConfig:
    in_channels: int = 1        # 4 for this project's packed Bayer
    num_iters: int = 8          # unrolled steps; try 4 / 8 / 12
    share_weights: bool = False
    patch_size: int = 8
    top_k: int = 16
    d_model: int = 256
    hidden_mult: int = 2
    alpha_init: float = 0.1
    # Queries scored per block in the non-local similarity. Lower it when the
    # input is large: peak memory there scales with chunk_size x N.
    chunk_size: int = 1024
    # Recompute proximal steps in the backward pass instead of storing them.
    # Worth enabling past ~8 iterations.
    checkpoint: bool = False
