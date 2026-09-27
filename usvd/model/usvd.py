"""USVD — unrolled spatially-varying denoiser."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .noise_estimator import NoiseEstimator
from .prox_denoiser import ProxDenoiser

__all__ = ["USVD"]


class USVD(nn.Module):
    """
    Flat, iterative denoiser built by unrolling proximal gradient descent.

    There is no encoder-decoder: every iteration runs at full input
    resolution. Each step takes a gradient step on the data-fidelity term
    weighted per pixel by 1/sigma^2 -- which is what makes it
    spatially varying -- and then applies a learned proximal operator.

        x^(0) = y
        r       = W * (x^(k) - y)
        x^(k+1) = prox_k(x^(k) - alpha_k * r, sigma)

    share_weights ties one ProxDenoiser across all iterations, which is the
    classical unrolling formulation and costs K times less memory in
    parameters; unshared gives each step its own operator.
    """

    def __init__(self, in_channels: int = 1, num_iters: int = 8,
                 share_weights: bool = False, patch_size: int = 8,
                 top_k: int = 16, d_model: int = 256, hidden_mult: int = 2,
                 alpha_init: float = 0.1, chunk_size: int = 1024,
                 checkpoint: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_iters = num_iters
        self.share_weights = share_weights
        self.checkpoint = checkpoint

        self.noise_estimator = NoiseEstimator(in_channels)

        def make():
            return ProxDenoiser(in_channels, hidden_mult=hidden_mult,
                                patch_size=patch_size, top_k=top_k,
                                d_model=d_model, chunk_size=chunk_size)

        if share_weights:
            shared = make()
            self.denoisers = nn.ModuleList([shared] * num_iters)
        else:
            self.denoisers = nn.ModuleList([make() for _ in range(num_iters)])

        # Stored raw and passed through softplus in forward, so the step size
        # stays positive however the optimiser moves it.
        inv = math_softplus_inverse(alpha_init)
        self.alpha_raw = nn.Parameter(torch.full((num_iters,), inv))

    def forward(self, y, sigma_map=None):
        if sigma_map is None:
            sigma_map = self.noise_estimator(y)

        weight = 1.0 / (sigma_map ** 2 + 1e-6)
        alpha = F.softplus(self.alpha_raw)

        x = y
        for k in range(self.num_iters):
            residual = weight * (x - y)
            x_grad = x - alpha[k] * residual
            if self.checkpoint and self.training:
                x = _checkpoint(self.denoisers[k], x_grad, sigma_map,
                                use_reentrant=False)
            else:
                x = self.denoisers[k](x_grad, sigma_map)
        return x, sigma_map


def math_softplus_inverse(value: float) -> float:
    """Raw parameter whose softplus equals `value`."""
    import math
    return math.log(math.expm1(value))
