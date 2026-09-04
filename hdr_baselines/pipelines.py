"""
hdr_baselines/pipelines.py — denoise + demosaic, composed
==========================================================

The model this project trains does denoising and demosaicing jointly. The
classical alternative does them in sequence, and *which* sequence is a
real decision with a known trade-off:

``denoise_first`` (the usual choice)
    Denoise on the CFA lattice, then demosaic. The denoiser sees raw
    sensor values, so the noise is still the simple signal-dependent kind
    the model assumes, and every sample is a real measurement. The cost is
    that each packed channel is a sub-lattice at half resolution, so a
    spatial filter's footprint covers twice the scene distance it appears
    to, and fine detail is oversmoothed.

``demosaic_first``
    Demosaic, then denoise the RGB. The denoiser sees full-resolution
    images with real spatial correlation, which helps it — but
    demosaicing has by then correlated the noise across pixels and
    channels, turning white noise into structured blotches that a
    white-noise denoiser cannot remove. It also amplifies noise at edges,
    where the interpolation weights are largest.

Both are provided because "which order" is exactly the sort of thing a
benchmark should answer with numbers rather than assert.

``auto_sigma`` re-parameterises the denoiser per image from an estimate of
that image's noise level. Without it a single set of constants has to
serve every exposure in the test set, and the classical baselines look
worse than they are — which would flatter the learned model for the wrong
reason.
"""

from __future__ import annotations

from typing import Optional

import torch

from hdr_data.bayer import unpack

from .demosaic import demosaic_bilinear, demosaic_malvar
from .denoise import Denoiser, estimate_noise_sigma
from .interface import BaselineModel

__all__ = ["ClassicalPipeline", "DEMOSAIC_FUNCTIONS"]


def _demosaic_gbtf(mosaic: torch.Tensor, pattern: str) -> torch.Tensor:
    """Lazily-built GBTF, so importing this module does not build a module."""
    from .demosaic import GBTFDemosaic
    if not hasattr(_demosaic_gbtf, "_cached"):
        _demosaic_gbtf._cached = GBTFDemosaic(pattern=pattern)
    # demosaic() moves the cached module onto the input's device and matches
    # its dtype, so the one shared instance follows whatever it is fed.
    return _demosaic_gbtf._cached.demosaic(mosaic)


def _demosaic_nearest(mosaic: torch.Tensor, pattern: str) -> torch.Tensor:
    import torch.nn.functional as F
    from hdr_data.bayer import pack, packed_to_half_rgb
    half = packed_to_half_rgb(pack(mosaic), pattern)
    return F.interpolate(half, scale_factor=2, mode="nearest")


#: Name -> ``(mosaic, pattern) -> rgb`` demosaicing function.
DEMOSAIC_FUNCTIONS = {
    "nearest": _demosaic_nearest,
    "bilinear": demosaic_bilinear,
    "malvar": demosaic_malvar,
    "gbtf": _demosaic_gbtf,
}


class ClassicalPipeline(BaselineModel):
    """
    A classical denoise + demosaic pipeline as a scorable model.

    Parameters
    ──────────
    denoiser
        Any :class:`~hdr_baselines.denoise.Denoiser`, or None for
        demosaicing alone.
    demosaic
        A key of :data:`DEMOSAIC_FUNCTIONS`.
    order
        ``"denoise_first"`` or ``"demosaic_first"``; see the module
        docstring.
    auto_sigma
        Re-parameterise the denoiser per image from an estimate of that
        image's noise level, using ``type(denoiser).for_noise_sigma``.
    pattern
        CFA phase. ``gbtf`` requires BGGR.
    """

    trainable = False

    def __init__(self, denoiser: Optional[Denoiser] = None,
                 demosaic: str = "malvar", order: str = "denoise_first",
                 auto_sigma: bool = True, pattern: str = "BGGR",
                 name: Optional[str] = None):
        if demosaic not in DEMOSAIC_FUNCTIONS:
            raise ValueError(
                f"Unknown demosaic '{demosaic}'. Available: "
                f"{sorted(DEMOSAIC_FUNCTIONS)}")
        if order not in ("denoise_first", "demosaic_first"):
            raise ValueError(
                f"order must be 'denoise_first' or 'demosaic_first', "
                f"got '{order}'")
        if denoiser is not None and not isinstance(denoiser, Denoiser):
            raise TypeError(
                f"denoiser must be a Denoiser or None, got "
                f"{type(denoiser).__name__}")
        if demosaic == "gbtf" and pattern.upper() != "BGGR":
            raise ValueError(
                f"The gbtf demosaicer is BGGR-only, got pattern '{pattern}'.")

        auto_name = (demosaic if denoiser is None
                     else f"{denoiser.name}+{demosaic}"
                          f"{'' if order == 'denoise_first' else '_post'}")
        super().__init__(name=name or auto_name, pattern=pattern)
        self.denoiser = denoiser
        self.demosaic_name = demosaic
        self.order = order
        self.auto_sigma = bool(auto_sigma)

    def _denoise(self, x: torch.Tensor) -> torch.Tensor:
        if self.denoiser is None:
            return x
        denoiser = self.denoiser
        if self.auto_sigma:
            sigma = float(estimate_noise_sigma(x).mean())
            # A degenerate estimate (a flat or already-clean patch) means
            # there is nothing to tune to; keep the configured denoiser.
            if sigma > 1e-6:
                denoiser = denoiser.retuned(sigma).to(x.device)
        return denoiser(x)

    def predict_rgb(self, x: torch.Tensor,
                    snr_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        fn = DEMOSAIC_FUNCTIONS[self.demosaic_name]
        if self.order == "denoise_first":
            rgb = fn(unpack(self._denoise(x)), self.pattern)
        else:
            rgb = self._denoise(fn(unpack(x), self.pattern))
        return rgb.clamp(0.0, 1.0)

    def extra_repr(self) -> str:
        d = "none" if self.denoiser is None else self.denoiser.name
        return (f"name='{self.name}', denoise={d}, "
                f"demosaic='{self.demosaic_name}', order='{self.order}', "
                f"auto_sigma={self.auto_sigma}")
