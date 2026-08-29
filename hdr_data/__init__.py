"""
hdr_data — datasets, sensor simulation and CFA handling for the HDR project
===========================================================================

Three things live here:

``bayer``
    The CFA algebra: packing, mosaicking, phase-aware flips, and RGB ->
    sensor conversion. Everything that used to assume "BGGR, hardcoded"
    goes through this.

``noise``
    A configurable Poisson-Gaussian sensor model — bit-exact with the
    original ``add_photon_noise`` on its legacy preset, extended with
    per-channel gain, PRNU, row/column banding and hot pixels, plus
    :func:`~hdr_data.noise.calibrate_from_pairs` to fit the model to real
    captures rather than assume it.

datasets
    :class:`~hdr_data.mobile_hdr.MobileHDRPacked` (the existing corpus),
    :class:`~hdr_data.video_i2.I2VideoDataset` (high-speed video through a
    virtual sensor) and :class:`~hdr_data.kalantari.KalantariDataset`
    (bracketed multi-exposure scenes), all sharing one base class so a
    crop, a noise preset or an augmentation means the same thing in each.

Typical use:

    from hdr_data import build_dataset, D4Transform

    ds = build_dataset("mobile_hdr", split="train", crop_size=512,
                       noise="realistic", num_patch=8,
                       transform=D4Transform("BGGR", mode="cell"))
"""

from .augment import (
    CenterCropPacked,
    Compose,
    D4Transform,
    RandomCropPacked,
    make_d4_transform_compat,
)
from .base import BaseBayerDataset, resolve_noise_model
from .bayer import (
    CFA_PATTERNS,
    FlipMode,
    cfa_masks,
    pack,
    packed_to_half_rgb,
    pattern_after,
    rgb_to_packed,
    unpack,
)
from .kalantari import KalantariDataset
from .mobile_hdr import MobileHDRPacked
from .noise import (
    NOISE_PRESETS,
    NoiseModel,
    NoiseParams,
    calibrate_from_pairs,
    get_preset,
    list_presets,
)
from .registry import DATASETS, build_dataset, list_datasets, resolve_root
from .video_i2 import I2VideoDataset, probe_decoder

__all__ = [
    # bayer
    "CFA_PATTERNS", "FlipMode", "cfa_masks", "pack", "unpack",
    "pattern_after", "rgb_to_packed", "packed_to_half_rgb",
    # noise
    "NoiseModel", "NoiseParams", "NOISE_PRESETS", "get_preset",
    "list_presets", "calibrate_from_pairs",
    # augment
    "Compose", "D4Transform", "RandomCropPacked", "CenterCropPacked",
    "make_d4_transform_compat",
    # datasets
    "BaseBayerDataset", "resolve_noise_model", "MobileHDRPacked",
    "I2VideoDataset", "KalantariDataset", "probe_decoder",
    # registry
    "DATASETS", "build_dataset", "list_datasets", "resolve_root",
]
