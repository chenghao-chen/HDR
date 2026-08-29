"""
hdr_data/kalantari.py — Kalantari & Ramamoorthi 2017 multi-exposure scenes
===========================================================================

The classic dynamic-scene HDR benchmark: per scene, three bracketed LDR
exposures (aligned) plus a reference HDR image. This project's model is a
*joint denoise + demosaic* network taking packed CFA and emitting RGB, so
the reference HDR is the natural supervision target: it is mosaicked into
a virtual sensor exactly as the video loader does, and the noise model
does the rest.

The bracketed stack is not thrown away — pass ``return_ldr=True`` and each
sample carries the three exposures and their times, which is what a
merge-then-denoise comparison needs.

Layout
──────
The converter shipped with the dataset (``convert_to_tfrecord.py``) fixes
the training-set names, and the released test scenes use a shorter set;
both are accepted, tried in order:

    input_[123]_aligned.tif  +  input_exp.txt  +  ref_hdr_aligned.hdr
    *.tif                    +  exposure.txt   +  HDRImg.hdr

Exposure files hold one value per line. The dataset convention is EV
stops, so exposure time is ``2 ** ev``; a file whose values are already
times (all positive, spanning orders of magnitude) is detected and used
as-is.

Note on this checkout
─────────────────────
``datasets/kalantari2017`` here contains the directory skeleton only — 74
train and 15 test scene folders, all empty. This loader is written to the
layout above and covered by tests against synthetic scenes; it has not
been run against the real imagery, because the real imagery is not
present. Populate the scene folders and it will pick them up.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .base import BaseBayerDataset
from .bayer import rgb_to_packed

__all__ = [
    "KalantariDataset",
    "read_exposures",
    "read_hdr",
    "read_ldr",
    "ldr_to_linear",
    "merge_exposures",
    "MissingImageIO",
]

#: Candidate filenames for the reference HDR, most specific first.
HDR_CANDIDATES = ("ref_hdr_aligned.hdr", "HDRImg.hdr", "ref_hdr.hdr", "*.hdr")
#: Candidate filenames for the exposure list.
EXPOSURE_CANDIDATES = ("input_exp.txt", "exposure.txt", "exposures.txt")
#: Glob patterns for the bracketed LDR inputs, most specific first.
LDR_PATTERNS = ("input_*_aligned.tif", "input_*.tif", "*.tif")


class MissingImageIO(RuntimeError):
    """Raised when no backend can read .hdr / .tif imagery."""


def _import_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:                       # pragma: no cover
        raise MissingImageIO(
            "OpenCV (cv2) is required to read Radiance .hdr and .tif "
            "images. Install opencv-python."
        ) from exc
    return cv2


# ─────────────────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────────────────

def read_exposures(path: str, num_expected: Optional[int] = None) -> List[float]:
    """
    Read an exposure file as a list of *exposure times*.

    Values are EV stops by convention, so a time is ``2 ** ev``. A file
    whose values are already times is detected heuristically: EV lists in
    this dataset are small signed numbers, so any value above 32 or any
    non-integral positive value spanning decades is taken as a time.
    """
    with open(path) as fh:
        raw = [line.strip() for line in fh if line.strip()]
    if not raw:
        raise ValueError(f"Exposure file is empty: {path}")
    try:
        values = [float(v) for v in raw]
    except ValueError as exc:
        raise ValueError(f"Non-numeric entry in {path}: {exc}") from exc

    if num_expected is not None:
        values = values[:num_expected]
        if len(values) < num_expected:
            raise ValueError(
                f"{path} holds {len(values)} exposures, expected "
                f"{num_expected}.")

    looks_like_times = all(v > 0 for v in values) and max(values) > 32.0
    if looks_like_times:
        return values
    return [float(2.0 ** v) for v in values]


def read_hdr(path: str) -> torch.Tensor:
    """
    Read a Radiance .hdr file as [3, H, W] float32 RGB, linear light.

    OpenCV returns BGR and (for .hdr) raw float values, which is what the
    reference converter relies on too.
    """
    cv2 = _import_cv2()
    import numpy as np

    arr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise MissingImageIO(f"Could not read HDR image: {path}")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr[:, :, ::-1]).astype("float32")  # BGR -> RGB
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def read_ldr(path: str) -> torch.Tensor:
    """
    Read one bracketed LDR exposure as [3, H, W] float32 RGB in [0, 1].

    8- and 16-bit files are both accepted and normalised by their depth.
    """
    cv2 = _import_cv2()
    import numpy as np

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise MissingImageIO(f"Could not read LDR image: {path}")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr[:, :, ::-1])                    # BGR -> RGB

    if arr.dtype == np.uint8:
        scale = 255.0
    elif arr.dtype == np.uint16:
        scale = 65535.0
    else:
        scale = 1.0
    t = torch.from_numpy(arr.astype("float32") / scale)
    return t.permute(2, 0, 1).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Exposure arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def ldr_to_linear(ldr: torch.Tensor, exposure_time: float,
                  gamma: float = 2.2) -> torch.Tensor:
    """
    Undo the camera response and the exposure: LDR [0,1] -> linear radiance.

    The dataset's own baseline assumes a gamma camera response, so
    ``radiance = ldr ** gamma / exposure_time``.
    """
    if exposure_time <= 0:
        raise ValueError(f"exposure_time must be positive, got {exposure_time}")
    return ldr.clamp(min=0.0) ** gamma / exposure_time


def merge_exposures(ldrs: Sequence[torch.Tensor],
                    exposure_times: Sequence[float],
                    gamma: float = 2.2,
                    low: float = 0.05, high: float = 0.95,
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Debevec-style weighted merge of a bracketed stack into linear HDR.

    Each exposure votes with a hat weight that falls to zero at the ends
    of its usable range, so under- and over-exposed pixels are excluded
    rather than averaged in. Pixels outside every exposure's usable range
    (deep shadow in the shortest, blown highlight in the longest) fall
    back to the single exposure whose value is closest to mid-grey.

    Returns [3, H, W] linear radiance.
    """
    if len(ldrs) != len(exposure_times):
        raise ValueError(
            f"{len(ldrs)} images but {len(exposure_times)} exposure times.")
    if not ldrs:
        raise ValueError("Need at least one exposure to merge.")
    shapes = {tuple(t.shape) for t in ldrs}
    if len(shapes) != 1:
        raise ValueError(f"Exposures have differing shapes: {sorted(shapes)}")

    num = torch.zeros_like(ldrs[0])
    den = torch.zeros_like(ldrs[0])
    for img, t in zip(ldrs, exposure_times):
        x = img.clamp(0.0, 1.0)
        # Hat weight, zero outside [low, high], peaking at mid-grey.
        w = 1.0 - (2.0 * x - 1.0).abs()
        w = torch.where((x >= low) & (x <= high), w, torch.zeros_like(w))
        num = num + w * ldr_to_linear(x, t, gamma)
        den = den + w

    merged = num / (den + eps)

    unused = den <= eps
    if bool(unused.any()):
        best = None
        best_dist = None
        for img, t in zip(ldrs, exposure_times):
            x = img.clamp(0.0, 1.0)
            dist = (x - 0.5).abs()
            lin = ldr_to_linear(x, t, gamma)
            if best is None:
                best, best_dist = lin, dist
            else:
                take = dist < best_dist
                best = torch.where(take, lin, best)
                best_dist = torch.where(take, dist, best_dist)
        merged = torch.where(unused, best, merged)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KalantariDataset(BaseBayerDataset):
    """
    Kalantari 2017 scenes as packed-CFA denoise/demosaic pairs.

    Parameters
    ──────────
    base_dir
        Root holding ``train/`` and ``test/`` scene directories.
    source
        ``"hdr"`` (default) takes the reference HDR as the clean signal;
        ``"merge"`` merges the bracketed LDR stack instead, which is
        useful when a scene has no reference file.
    return_ldr
        Add ``ldr`` ([N, 3, H, W]) and ``exposure_times`` to each sample.
    gamma
        Camera response exponent used by the LDR linearisation.
    max_side
        Downscale so the longer side is at most this many pixels. The
        released images are ~1500x1000, which at full resolution is a slow
        way to smoke-test a pipeline.

    All other arguments are :class:`BaseBayerDataset`'s.
    """

    def __init__(self, base_dir: str, *, source: str = "hdr",
                 return_ldr: bool = False, gamma: float = 2.2,
                 max_side: Optional[int] = None, **kwargs):
        if source not in ("hdr", "merge"):
            raise ValueError(f"source must be 'hdr' or 'merge', got '{source}'")
        if max_side is not None and max_side < 32:
            raise ValueError(f"max_side must be >= 32, got {max_side}")
        self.base_dir = str(base_dir)
        self.source = source
        self.return_ldr = bool(return_ldr)
        self.gamma = float(gamma)
        self.max_side = max_side
        kwargs.setdefault("pattern", "BGGR")
        super().__init__(**kwargs)

    # ── scene discovery ──────────────────────────────────────────────────
    @property
    def split_dir(self) -> str:
        return os.path.join(self.base_dir, self.split)

    @staticmethod
    def _first_match(scene: str, candidates: Sequence[str]) -> Optional[str]:
        """First existing file among literal names or glob patterns."""
        for name in candidates:
            if any(ch in name for ch in "*?["):
                hits = sorted(glob.glob(os.path.join(scene, name)))
                if hits:
                    return hits[0]
            else:
                path = os.path.join(scene, name)
                if os.path.isfile(path):
                    return path
        return None

    @classmethod
    def find_hdr(cls, scene: str) -> Optional[str]:
        """Path to the scene's reference HDR, or None."""
        return cls._first_match(scene, HDR_CANDIDATES)

    @classmethod
    def find_exposures(cls, scene: str) -> Optional[str]:
        """Path to the scene's exposure list, or None."""
        return cls._first_match(scene, EXPOSURE_CANDIDATES)

    @classmethod
    def find_ldrs(cls, scene: str) -> List[str]:
        """Sorted bracketed LDR paths — sorted order is exposure order."""
        for pattern in LDR_PATTERNS:
            hits = sorted(glob.glob(os.path.join(scene, pattern)))
            # ref_* files match the bare "*.tif" fallback; drop them so a
            # scene laid out the training way does not mix inputs and refs.
            hits = [h for h in hits
                    if not os.path.basename(h).startswith("ref_")]
            if hits:
                return hits
        return []

    def _scene_is_usable(self, scene: str) -> bool:
        if self.source == "hdr":
            return self.find_hdr(scene) is not None
        return bool(self.find_ldrs(scene)) and self.find_exposures(scene) is not None

    def _build_index(self) -> Sequence[str]:
        d = self.split_dir
        if not os.path.isdir(d):
            return []
        scenes = sorted(
            os.path.join(d, name) for name in os.listdir(d)
            if os.path.isdir(os.path.join(d, name)))
        return [s for s in scenes if self._scene_is_usable(s)]

    def _empty_index_hint(self) -> str:
        d = self.split_dir
        if not os.path.isdir(d):
            return f"{d} does not exist."
        total = sum(1 for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))
        want = ("a reference HDR (one of "
                f"{', '.join(HDR_CANDIDATES)})" if self.source == "hdr"
                else "bracketed *.tif exposures and an exposure list")
        return (f"{total} scene directories under {d}, none containing {want}. "
                f"In this checkout the Kalantari scene folders are empty — the "
                f"imagery has not been downloaded.")

    def _entry_id(self, entry: str) -> str:
        return os.path.basename(entry.rstrip("/"))

    # ── loading ──────────────────────────────────────────────────────────
    def _resize(self, rgb: torch.Tensor) -> torch.Tensor:
        """Optionally shrink so the longer side is at most ``max_side``."""
        if self.max_side is None:
            return rgb
        h, w = rgb.shape[-2], rgb.shape[-1]
        longest = max(h, w)
        if longest <= self.max_side:
            return rgb
        import torch.nn.functional as F
        factor = self.max_side / float(longest)
        new_h = max(16, int(round(h * factor)))
        new_w = max(16, int(round(w * factor)))
        return F.interpolate(rgb.unsqueeze(0), size=(new_h, new_w),
                             mode="area").squeeze(0)

    def load_scene_rgb(self, scene: str) -> torch.Tensor:
        """The scene's clean linear RGB, before mosaicking. [3, H, W]."""
        if self.source == "hdr":
            path = self.find_hdr(scene)
            if path is None:
                raise RuntimeError(f"No reference HDR in {scene}")
            rgb = read_hdr(path)
        else:
            paths = self.find_ldrs(scene)
            exp_path = self.find_exposures(scene)
            if not paths or exp_path is None:
                raise RuntimeError(f"No usable LDR stack in {scene}")
            times = read_exposures(exp_path, num_expected=len(paths))
            rgb = merge_exposures([read_ldr(p) for p in paths], times,
                                  gamma=self.gamma)
        return self._resize(rgb)

    def load_ldr_stack(self, scene: str) -> Tuple[torch.Tensor, List[float]]:
        """The bracketed stack as ([N, 3, H, W], exposure_times)."""
        paths = self.find_ldrs(scene)
        exp_path = self.find_exposures(scene)
        if not paths:
            raise RuntimeError(f"No LDR exposures in {scene}")
        times = (read_exposures(exp_path, num_expected=len(paths))
                 if exp_path is not None else [1.0] * len(paths))
        stack = torch.stack([self._resize(read_ldr(p)) for p in paths])
        return stack, times

    def _load_clean(self, entry: str) -> torch.Tensor:
        rgb = self.load_scene_rgb(entry)
        h, w = rgb.shape[-2], rgb.shape[-1]
        # Trim to a multiple of 16 sensor pixels: even for the mosaic, and
        # divisible by 8 once packed for the encoder's three unshuffles.
        h -= h % 16
        w -= w % 16
        if h < 16 or w < 16:
            raise RuntimeError(
                f"{self._entry_id(entry)}: {rgb.shape[-2]}x{rgb.shape[-1]} is "
                f"too small after trimming to a multiple of 16.")
        return rgb_to_packed(rgb[..., :h, :w], self.pattern)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        if self.return_ldr:
            scene = self.entries[idx % len(self.entries)]
            stack, times = self.load_ldr_stack(scene)
            sample["ldr"] = stack
            sample["exposure_times"] = times
        return sample

    @property
    def scene_ids(self) -> List[str]:
        """Scene directory names, in index order."""
        return [self._entry_id(e) for e in self.entries]
