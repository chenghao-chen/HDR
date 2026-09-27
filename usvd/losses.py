"""Losses for the unrolled denoiser."""
from __future__ import annotations

import torch.nn.functional as F

__all__ = ["usvd_loss"]


def usvd_loss(pred, target, sigma_map, pred_sigma=None, true_sigma=None,
              l1_weight: float = 0.1, sigma_weight: float = 0.5):
    """
    Noise-weighted reconstruction, plus an L1 term and optional sigma
    supervision.

    The sigma map is detached in the weight: it is a *weighting* on the
    reconstruction term, and leaving it attached would let the model shrink
    the loss by inflating its own sigma estimate rather than by denoising
    better.
    """
    weight = 1.0 / (sigma_map.detach() ** 2 + 1e-6)
    recon = (weight * (pred - target) ** 2).mean()
    l1 = F.l1_loss(pred, target)

    total = recon + l1_weight * l1
    parts = {"recon": float(recon), "l1": float(l1)}

    if pred_sigma is not None and true_sigma is not None:
        sigma_term = F.mse_loss(pred_sigma, true_sigma)
        total = total + sigma_weight * sigma_term
        parts["sigma"] = float(sigma_term)

    return total, parts
