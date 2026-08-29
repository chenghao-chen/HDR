"""
hdr_baselines — comparison models for the HDR benchmark
========================================================

Everything here presents the project's own model signature —
``blended, experts, gates = model(x, snr_map)`` — so a classical
pipeline, a small learned network and the MoE teacher can all be scored
by the same loop.

    from hdr_baselines import build_baseline

    model = build_baseline("wavelet+gbtf")     # denoise, then demosaic
    model = build_baseline("unet")             # a trainable reference

See :func:`~hdr_baselines.registry.describe_baselines` for the accepted
names.
"""

from .demosaic import (
    BilinearDemosaic,
    GBTFDemosaic,
    MalvarDemosaic,
    NearestDemosaic,
    demosaic_bilinear,
    demosaic_malvar,
)
from .denoise import (
    BilateralDenoise,
    Denoiser,
    GaussianDenoise,
    GuidedFilterDenoise,
    MedianDenoise,
    NLMDenoise,
    VarianceStabilised,
    WaveletDenoise,
    estimate_noise_sigma,
)
from .interface import BaselineModel
from .learned import DemosaicNetJDD, DnCNNJDD, RestormerLiteJDD, UNetJDD
from .pipelines import ClassicalPipeline
from .registry import (
    DEFAULT_LINEUP,
    build_baseline,
    build_denoiser,
    describe_baselines,
    list_baselines,
    list_demosaicers,
    list_denoisers,
)

__all__ = [
    "BaselineModel",
    "NearestDemosaic", "BilinearDemosaic", "MalvarDemosaic", "GBTFDemosaic",
    "demosaic_bilinear", "demosaic_malvar",
    "Denoiser", "GaussianDenoise", "MedianDenoise", "BilateralDenoise",
    "GuidedFilterDenoise", "NLMDenoise", "WaveletDenoise",
    "VarianceStabilised", "estimate_noise_sigma",
    "ClassicalPipeline",
    "DnCNNJDD", "UNetJDD", "DemosaicNetJDD", "RestormerLiteJDD",
    "build_baseline", "build_denoiser", "list_baselines", "list_denoisers",
    "list_demosaicers", "describe_baselines", "DEFAULT_LINEUP",
]
