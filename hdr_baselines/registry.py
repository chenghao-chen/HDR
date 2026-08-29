"""
hdr_baselines/registry.py — baselines by name
==============================================

``--baselines bilinear,gaussian+malvar,unet`` has to become a list of
constructed models. This is that lookup.

Names follow the pipeline convention: a bare demosaicer (``malvar``), a
denoiser and demosaicer joined by ``+`` (``wavelet+gbtf``), optionally
suffixed ``_post`` for demosaic-then-denoise (``wavelet+gbtf_post``), or
the name of a learned architecture (``unet``). Anything of the first three
forms is constructed on the fly, so the useful combinations do not have to
be enumerated here — :func:`list_baselines` shows the named ones and
:func:`build_baseline` accepts any valid combination.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .demosaic import (
    BilinearDemosaic,
    GBTFDemosaic,
    MalvarDemosaic,
    NearestDemosaic,
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
)
from .interface import BaselineModel
from .learned import DemosaicNetJDD, DnCNNJDD, RestormerLiteJDD, UNetJDD
from .pipelines import DEMOSAIC_FUNCTIONS, ClassicalPipeline

__all__ = [
    "DENOISERS",
    "DEMOSAICERS",
    "LEARNED",
    "list_baselines",
    "list_denoisers",
    "list_demosaicers",
    "build_baseline",
    "build_denoiser",
    "describe_baselines",
]

#: Denoiser name -> zero-argument factory.
DENOISERS: Dict[str, Callable[[], Denoiser]] = {
    "gaussian": GaussianDenoise,
    "median": MedianDenoise,
    "bilateral": BilateralDenoise,
    "guided": GuidedFilterDenoise,
    "nlm": NLMDenoise,
    "wavelet": WaveletDenoise,
}

#: Demosaicer name -> BaselineModel factory (demosaicing alone).
DEMOSAICERS: Dict[str, Callable[..., BaselineModel]] = {
    "nearest": NearestDemosaic,
    "bilinear": BilinearDemosaic,
    "malvar": MalvarDemosaic,
    "gbtf": GBTFDemosaic,
}

#: Learned architecture name -> factory.
LEARNED: Dict[str, Callable[..., BaselineModel]] = {
    "dncnn": DnCNNJDD,
    "unet": UNetJDD,
    "demosaicnet": DemosaicNetJDD,
    "restormer_lite": RestormerLiteJDD,
}

#: A reasonable default lineup for a first comparison run.
DEFAULT_LINEUP = (
    "bilinear",
    "malvar",
    "gbtf",
    "gaussian+gbtf",
    "wavelet+gbtf",
    "guided+malvar",
)


def list_denoisers() -> List[str]:
    """Registered denoiser names, sorted."""
    return sorted(DENOISERS)


def list_demosaicers() -> List[str]:
    """Registered demosaicer names, sorted."""
    return sorted(DEMOSAICERS)


def list_baselines() -> List[str]:
    """Every directly-named baseline (not the ``denoise+demosaic`` combinations)."""
    return sorted(set(DEMOSAICERS) | set(LEARNED))


def describe_baselines() -> str:
    """A help string listing what ``--baselines`` accepts."""
    return (
        "Demosaic only:   " + ", ".join(list_demosaicers()) + "\n"
        "Denoise+demosaic: <denoiser>+<demosaicer>, e.g. wavelet+gbtf\n"
        "  denoisers:     " + ", ".join(list_denoisers()) + "\n"
        "  add '_post' to denoise after demosaicing: wavelet+gbtf_post\n"
        "  prefix a denoiser with 'vst_' for variance-stabilised: "
        "vst_wavelet+gbtf\n"
        "Learned (untrained unless a checkpoint is given): "
        + ", ".join(sorted(LEARNED))
    )


def build_denoiser(name: str, **kwargs) -> Denoiser:
    """
    Construct a denoiser by name.

    A ``vst_`` prefix wraps it in the generalised Anscombe transform, so
    ``vst_wavelet`` is wavelet shrinkage applied to variance-stabilised
    data.
    """
    if name.startswith("vst_"):
        inner = build_denoiser(name[4:], **kwargs)
        return VarianceStabilised(inner)
    if name not in DENOISERS:
        raise KeyError(
            f"Unknown denoiser '{name}'. Available: {list_denoisers()} "
            f"(optionally prefixed 'vst_').")
    return DENOISERS[name](**kwargs)


def build_baseline(name: str, *, pattern: str = "BGGR",
                   auto_sigma: bool = True, **kwargs) -> BaselineModel:
    """
    Construct a baseline from its name.

    Accepted forms::

        malvar                 demosaicing alone
        wavelet+gbtf           denoise on the CFA, then demosaic
        wavelet+gbtf_post      demosaic, then denoise the RGB
        vst_wavelet+malvar     variance-stabilised denoise, then demosaic
        unet                   a learned architecture (untrained)
    """
    if not name:
        raise ValueError("Baseline name must not be empty.")

    if name in LEARNED:
        return LEARNED[name](pattern=pattern, **kwargs)

    if "+" in name:
        den_name, dem_name = name.split("+", 1)
        order = "denoise_first"
        if dem_name.endswith("_post"):
            dem_name = dem_name[: -len("_post")]
            order = "demosaic_first"
        if dem_name not in DEMOSAIC_FUNCTIONS:
            raise KeyError(
                f"Unknown demosaicer '{dem_name}' in baseline '{name}'. "
                f"Available: {sorted(DEMOSAIC_FUNCTIONS)}")
        return ClassicalPipeline(
            build_denoiser(den_name), dem_name, order=order,
            auto_sigma=auto_sigma, pattern=pattern, name=name, **kwargs)

    if name in DEMOSAICERS:
        return DEMOSAICERS[name](pattern=pattern, **kwargs)

    raise KeyError(
        f"Unknown baseline '{name}'.\n{describe_baselines()}")
