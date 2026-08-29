"""
hdr_data/mobile_hdr.py — the MobileHDR packed-Bayer dataset
============================================================

The same corpus ``HDR_Mobile_dataset.MobileHDRDataset`` reads, expressed on
top of :class:`hdr_data.base.BaseBayerDataset` so it shares the crop,
cache, augmentation and noise machinery with the video and multi-exposure
loaders.

On-disk layout
──────────────
    <root>/train/tensors/**/*.pt          (4, H, W) float32 packed BGGR
    <root>/test/tensors/with_gt/*.pt      same, one level deep
    <root>/test/tensors/without_gt/*.pt   inputs with no reference

Values are unnormalised HDR; the packed channel order is (B, G1, G2, R).

Differences from the original class
───────────────────────────────────
* Any noise preset, not just the hardcoded high-noise recipe.
* ``with_gt`` / ``without_gt`` selectable for the test split.
* Optional metadata in the sample dict (source id, realised alpha and read
  variance) for stratified evaluation.
* Centre-cropping on the test split, so a benchmark can run on fixed-size
  tiles without the full-frame memory cost.

With default arguments and the legacy noise preset it produces exactly the
same tensors as the original for the same seed — that equivalence is
pinned by tests/test_data_mobile_hdr.py.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Tuple

import torch

from .base import BaseBayerDataset

__all__ = ["MobileHDRPacked"]


class MobileHDRPacked(BaseBayerDataset):
    """
    Packed-BGGR MobileHDR frames with synthetic sensor noise.

    Parameters
    ──────────
    base_dir
        Dataset root containing ``train/`` and ``test/``.
    test_subset
        ``"with_gt"`` (default) or ``"without_gt"``. The latter has no
        reference, so the "clean" tensor it pairs with is the un-noised
        capture itself — useful for qualitative runs, not for metrics.
    mmap
        Memory-map the .pt files so a crop only faults in the pages it
        touches (torch >= 2.1). Falls back automatically when unsupported.

    All other arguments are :class:`BaseBayerDataset`'s.
    """

    def __init__(self, base_dir: str, *, test_subset: str = "with_gt",
                 mmap: bool = True, **kwargs):
        if test_subset not in ("with_gt", "without_gt"):
            raise ValueError(
                f"test_subset must be 'with_gt' or 'without_gt', "
                f"got '{test_subset}'")
        self.base_dir = str(base_dir)
        self.test_subset = test_subset
        self.mmap = bool(mmap)
        kwargs.setdefault("pattern", "BGGR")
        super().__init__(**kwargs)

    # ── index ────────────────────────────────────────────────────────────
    @property
    def tensor_dir(self) -> str:
        """Directory the current split reads from."""
        if self.split == "train":
            return os.path.join(self.base_dir, "train", "tensors")
        return os.path.join(self.base_dir, "test", "tensors", self.test_subset)

    def _build_index(self) -> Sequence[str]:
        # Sorted: glob order is filesystem-dependent, and an unstable order
        # would make the per-index test noise seed point at a different
        # image between runs.
        if self.split == "train":
            pattern = os.path.join(self.tensor_dir, "**", "*.pt")
            return sorted(glob.glob(pattern, recursive=True))
        return sorted(glob.glob(os.path.join(self.tensor_dir, "*.pt")))

    def _empty_index_hint(self) -> str:
        return (f"Looked in {self.tensor_dir}. Set base_dir to the dataset "
                f"root (the directory holding train/ and test/), or check "
                f"HDR_DATASET_DIR.")

    # ── loading ──────────────────────────────────────────────────────────
    def _load_clean(self, entry: str) -> torch.Tensor:
        if self.mmap:
            try:
                return torch.load(entry, weights_only=True, mmap=True)
            except (TypeError, RuntimeError):
                # Older torch, or a file the mmap path cannot take.
                pass
        return torch.load(entry, weights_only=True)

    # ── convenience ──────────────────────────────────────────────────────
    @property
    def file_list(self) -> List[str]:
        """Alias for ``entries`` matching the original class's attribute."""
        return self.entries

    def frame_shape(self, idx: int = 0) -> Tuple[int, int, int]:
        """Shape of one source tensor, without keeping it in memory."""
        t = self._load_clean(self.entries[idx])
        return tuple(t.shape)
