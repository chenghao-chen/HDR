"""
hdr_data/registry.py — datasets by name
========================================

The evaluation CLI takes ``--dataset mobile_hdr`` and has to turn that into
a constructed Dataset without knowing which module it lives in. This is
that lookup, plus the default-root resolution: each dataset declares the
environment variables and filesystem locations it will look in, so a run
on a different cluster only needs the env var set.

    from hdr_data.registry import build_dataset, list_datasets

    ds = build_dataset("mobile_hdr", split="test", noise="high")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

__all__ = [
    "DatasetSpec",
    "DATASETS",
    "list_datasets",
    "get_spec",
    "resolve_root",
    "build_dataset",
]

#: Where the project's data lives on this machine, used as the last resort
#: when no environment variable is set.
DEFAULT_DATA_ROOT = "/lus/eagle/projects/lighthouse-purdue/ryanchen/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    """
    Everything needed to construct one dataset by name.

    Attributes
    ──────────
    name
        Registry key, as typed on the command line.
    factory
        Callable taking ``(root, **kwargs)`` and returning a Dataset.
    env_vars
        Environment variables consulted, in order, for the data root.
    default_subdir
        Appended to :data:`DEFAULT_DATA_ROOT` when no env var is set.
    description
        One line, shown by ``--list-datasets``.
    needs_root
        False for datasets that can be built without a path (none today,
        but synthetic generators plug in here).
    """

    name: str
    factory: Callable[..., Dataset]
    env_vars: Tuple[str, ...]
    default_subdir: str
    description: str
    needs_root: bool = True


def _mobile_hdr(root: str, **kwargs) -> Dataset:
    from .mobile_hdr import MobileHDRPacked
    return MobileHDRPacked(root, **kwargs)


def _i2_video(root: str, **kwargs) -> Dataset:
    from .video_i2 import I2VideoDataset
    return I2VideoDataset(root, **kwargs)


def _kalantari(root: str, **kwargs) -> Dataset:
    from .kalantari import KalantariDataset
    return KalantariDataset(root, **kwargs)


#: The name -> spec table the CLI reads.
DATASETS: Dict[str, DatasetSpec] = {
    "mobile_hdr": DatasetSpec(
        name="mobile_hdr",
        factory=_mobile_hdr,
        env_vars=("HDR_DATASET_DIR", "MOBILE_HDR_DIR"),
        default_subdir="Mobile-HDR",
        description="MobileHDR packed-BGGR frames (the training corpus).",
    ),
    "i2_video": DatasetSpec(
        name="i2_video",
        factory=_i2_video,
        env_vars=("I2_DATASET_DIR",),
        default_subdir="i2-2kfps_v1",
        description="i2-2kfps high-speed clips, mosaicked to a virtual sensor.",
    ),
    "kalantari": DatasetSpec(
        name="kalantari",
        factory=_kalantari,
        env_vars=("KALANTARI_DATASET_DIR",),
        default_subdir="kalantari2017",
        description="Kalantari 2017 bracketed scenes (needs the imagery downloaded).",
    ),
}


def list_datasets() -> List[str]:
    """Registered dataset names, sorted."""
    return sorted(DATASETS)


def get_spec(name: str) -> DatasetSpec:
    """Look up a spec, with the available names in the error."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return DATASETS[name]


def resolve_root(name: str, root: Optional[str] = None) -> str:
    """
    Data root for `name`: explicit argument, then each environment
    variable in order, then the project default.

    The path is returned whether or not it exists — the dataset's own
    "no samples found" error is more informative than one raised here.
    """
    if root:
        return str(root)
    spec = get_spec(name)
    for var in spec.env_vars:
        value = os.environ.get(var)
        if value:
            return value
    return os.path.join(DEFAULT_DATA_ROOT, spec.default_subdir)


def build_dataset(name: str, *, root: Optional[str] = None, **kwargs) -> Dataset:
    """
    Construct a registered dataset.

    Keyword arguments pass straight through to the dataset class, so
    ``build_dataset("mobile_hdr", split="test", noise="high",
    crop_size=512)`` works as expected.
    """
    spec = get_spec(name)
    resolved = resolve_root(name, root)
    if spec.needs_root:
        return spec.factory(resolved, **kwargs)
    return spec.factory(**kwargs)
