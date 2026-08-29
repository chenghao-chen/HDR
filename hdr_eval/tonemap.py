"""
hdr_eval/tonemap.py — tone curves for HDR metrics and display
==============================================================

Every number this project reports depends on the curve applied first.
PSNR on linear HDR is dominated by the highlights — a 1% error at
radiance 10 costs as much as a 100% error at radiance 0.1 — which is
exactly backwards from how the result looks. The literature standard is
therefore to measure through a mu-law curve, and this project uses
mu = 5000 (Kalantari, SIGGRAPH 2017) in training, in the test script and
here. Changing mu changes every PSNR-mu number, so the value is threaded
explicitly through the API rather than left to a default in three places.

The registry at the bottom lets the CLI take ``--tonemap reinhard`` and
get a curve without importing anything by name.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch

__all__ = [
    "MU_DEFAULT",
    "mu_law",
    "mu_law_inverse",
    "reinhard",
    "reinhard_extended",
    "gamma_encode",
    "gamma_decode",
    "log_encode",
    "identity",
    "TONEMAPS",
    "get_tonemap",
    "list_tonemaps",
    "tonemap",
]

#: The mu the training script, the test script and every reported number use.
MU_DEFAULT = 5000.0


def mu_law(x: torch.Tensor, mu: float = MU_DEFAULT) -> torch.Tensor:
    """
    mu-law compression: ``log1p(mu * x) / log1p(mu)``.

    Maps [0, 1] linear HDR onto a perceptually far more uniform [0, 1].
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    return torch.log1p(mu * x) / math.log1p(mu)


def mu_law_inverse(y: torch.Tensor, mu: float = MU_DEFAULT) -> torch.Tensor:
    """Exact inverse of :func:`mu_law`."""
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    return torch.expm1(y * math.log1p(mu)) / mu


def reinhard(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Global Reinhard operator, ``x / (1 + x)``.

    Cheaper than mu-law and never saturates, but compresses the midtones
    much harder — useful as a second opinion when a result looks like a
    tone-curve artefact rather than a real difference.
    """
    return x / (1.0 + x.clamp(min=0.0) + eps)


def reinhard_extended(x: torch.Tensor, white_point: float = 1.0,
                      eps: float = 1e-8) -> torch.Tensor:
    """
    Reinhard with a white point: values at or above `white_point` map to 1.

    ``x * (1 + x / w^2) / (1 + x)``.
    """
    if white_point <= 0:
        raise ValueError(f"white_point must be positive, got {white_point}")
    x = x.clamp(min=0.0)
    return (x * (1.0 + x / (white_point ** 2))) / (1.0 + x + eps)


def gamma_encode(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Display encoding ``x ** (1/gamma)``, clamped at zero first."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return x.clamp(min=0.0) ** (1.0 / gamma)


def gamma_decode(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Inverse of :func:`gamma_encode`."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return x.clamp(min=0.0) ** gamma


def log_encode(x: torch.Tensor, eps: float = 1e-4,
               max_value: float = 1.0) -> torch.Tensor:
    """
    Normalised log encoding, ``log(x + eps)`` rescaled to [0, 1].

    Closest of these curves to how a display-referred log profile behaves,
    and the most aggressive about lifting deep shadows — which makes it
    the harshest test of a denoiser's low-light behaviour.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    lo = math.log(eps)
    hi = math.log(max_value + eps)
    return (torch.log(x.clamp(min=0.0) + eps) - lo) / (hi - lo)


def identity(x: torch.Tensor) -> torch.Tensor:
    """No tone curve — measure in linear light."""
    return x


#: Name -> curve. Curves take (tensor, **kwargs) and return a tensor.
TONEMAPS: Dict[str, Callable[..., torch.Tensor]] = {
    "mu": mu_law,
    "mu_law": mu_law,
    "reinhard": reinhard,
    "reinhard_extended": reinhard_extended,
    "gamma": gamma_encode,
    "log": log_encode,
    "linear": identity,
    "none": identity,
}


def list_tonemaps() -> list:
    """Registered curve names, sorted."""
    return sorted(TONEMAPS)


def get_tonemap(name: str) -> Callable[..., torch.Tensor]:
    """Look up a curve by name."""
    if name not in TONEMAPS:
        raise KeyError(
            f"Unknown tonemap '{name}'. Available: {list_tonemaps()}")
    return TONEMAPS[name]


def tonemap(x: torch.Tensor, name: str = "mu", **kwargs) -> torch.Tensor:
    """
    Apply a named curve.

        tonemap(img, "mu", mu=5000)
        tonemap(img, "gamma", gamma=2.2)
    """
    return get_tonemap(name)(x, **kwargs)
