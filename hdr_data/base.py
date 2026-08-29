"""
hdr_data/base.py — the shared packed-Bayer dataset skeleton
============================================================

Every dataset in this package produces the same thing: a noisy/clean pair
of packed CFA tensors in [0, 1], cropped and augmented consistently, with
noise synthesised by a :class:`hdr_data.noise.NoiseModel`. Only *where the
clean signal comes from* differs — a .pt tensor on disk, a decoded video
frame, a stack of aligned exposures — so that is the only thing a subclass
has to implement:

    class MyDataset(BaseBayerDataset):
        def _build_index(self):    return [...]     # one entry per sample
        def _load_clean(self, entry): return tensor # [4, h, w], unnormalised

Everything else — the virtual-epoch repeat factor, cropping *before* noise
synthesis, the per-file range cache, deterministic test noise, the
augmentation hook and the output dict — is handled once, here.

Three details are load-bearing and easy to get wrong:

* **Crop before noise.** The random crop is taken from the clean tensor and
  noise is synthesised only for the crop. For 512^2 crops out of ~2K x 1.5K
  frames that is ~12x less dataloader CPU than noising the full frame.

* **Normalise by the full frame's range, not the crop's.** A crop carries
  its absolute brightness only if it is scaled by the range of the image it
  came from; per-crop normalisation would make every dark crop look
  mid-grey and destroy the low-light supervision signal. Subclasses report
  that range through :meth:`_clean_range`, which the base class caches.

* **Deterministic test noise.** The test split seeds a per-index generator,
  so every benchmark run sees the exact same noisy inputs and two
  checkpoints are compared on identical data.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from .bayer import canonical_pattern
from .noise import NoiseModel, NoiseParams, get_preset

__all__ = ["BaseBayerDataset", "resolve_noise_model"]

NoiseSpec = Union[str, NoiseParams, NoiseModel, None]


def resolve_noise_model(spec: NoiseSpec) -> NoiseModel:
    """
    Accept a preset name, a :class:`NoiseParams`, a ready
    :class:`NoiseModel`, or None (meaning the legacy defaults).
    """
    if spec is None:
        return NoiseModel(NoiseParams())
    if isinstance(spec, NoiseModel):
        return spec
    if isinstance(spec, NoiseParams):
        return NoiseModel(spec)
    if isinstance(spec, str):
        return NoiseModel(get_preset(spec))
    raise TypeError(
        f"noise must be a preset name, NoiseParams, NoiseModel or None; "
        f"got {type(spec).__name__}")


class BaseBayerDataset(Dataset):
    """
    Common machinery for packed-CFA denoise/demosaic datasets.

    Parameters
    ──────────
    split
        ``"train"`` or ``"test"``. The test split disables the random
        exposure factor and the highlight expansion, seeds noise per
        index, and takes a centre crop rather than a random one.
    transform
        Optional pair-transform ``(noisy, clean) -> (noisy, clean)``,
        applied after noise synthesis. See hdr_data.augment.
    num_patch
        Virtual repeats per item per epoch (train only): each visit
        re-crops and re-noises, so N repeats give N different samples.
    crop_size
        Packed-domain crop size, or None for whole frames. Applied to the
        clean tensor *before* noise.
    noise
        Preset name, NoiseParams, NoiseModel, or None for the legacy model.
    pattern
        CFA phase of the tensors this dataset yields.
    test_noise_seed
        Base seed for the deterministic test-split generator.
    return_meta
        Include the realised noise parameters and the source identifier in
        each sample dict. Useful for stratified evaluation; adds
        non-tensor entries, so pair it with a collate that tolerates them.
    """

    #: Subclasses may override when their tensors are not 4-channel packed.
    expected_channels: int = 4

    def __init__(self, *, split: str = "train", transform=None,
                 num_patch: int = 1, crop_size: Optional[int] = None,
                 noise: NoiseSpec = None, pattern: str = "BGGR",
                 test_noise_seed: int = 2025, return_meta: bool = False):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split '{split}' (use train | test)")
        if num_patch < 1:
            raise ValueError(f"num_patch must be >= 1, got {num_patch}")
        if crop_size is not None and crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}")

        self.split = split
        self.transform = transform
        self.num_patch = int(num_patch) if split == "train" else 1
        self.crop_size = crop_size
        self.pattern = canonical_pattern(pattern)
        self.test_noise_seed = int(test_noise_seed)
        self.return_meta = bool(return_meta)
        self.noise_model = resolve_noise_model(noise)

        self.entries: List[Any] = list(self._build_index())
        if not self.entries:
            raise RuntimeError(
                f"{type(self).__name__}: no samples found for split "
                f"'{split}'. {self._empty_index_hint()}")

        # Per-entry (min, max) of the FULL frame. Filled lazily, and per
        # DataLoader worker — workers do not share memory, so each pays the
        # first-touch cost once for the entries it happens to draw.
        self._range_cache: Dict[int, Tuple[float, float]] = {}

    # ── subclass hooks ───────────────────────────────────────────────────
    def _build_index(self) -> Sequence[Any]:
        """Return one entry per source item (a path, a (path, frame) pair...)."""
        raise NotImplementedError

    def _load_clean(self, entry: Any) -> torch.Tensor:
        """Load one entry as an unnormalised packed tensor [4, h, w]."""
        raise NotImplementedError

    def _entry_id(self, entry: Any) -> str:
        """A short human-readable identifier, used in reports and filenames."""
        if isinstance(entry, str):
            return os.path.splitext(os.path.basename(entry))[0]
        return str(entry)

    def _empty_index_hint(self) -> str:
        """Extra guidance appended to the "no samples" error."""
        return ""

    def _clean_range(self, index: int, tensor: torch.Tensor) -> Tuple[float, float]:
        """
        (min, max) of the FULL frame `index` came from, cached per entry.

        Subclasses that crop inside ``_load_clean`` must override this to
        report the uncropped range, or crops lose their absolute brightness.
        """
        if index not in self._range_cache:
            self._range_cache[index] = (float(tensor.min()), float(tensor.max()))
        return self._range_cache[index]

    # ── Dataset protocol ─────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.entries) * self.num_patch

    @property
    def num_sources(self) -> int:
        """Number of distinct source items, ignoring virtual repeats."""
        return len(self.entries)

    def source_id(self, idx: int) -> str:
        """Identifier of the source behind sample `idx`."""
        return self._entry_id(self.entries[idx % len(self.entries)])

    # Kept as no-ops for compatibility with older training scripts that
    # called these between epochs; crops and noise are regenerated on
    # every __getitem__, so there is nothing to regenerate here.
    def regen_crops(self) -> None:
        pass

    def regen_noise(self) -> None:
        pass

    def _crop(self, clean: torch.Tensor, is_train: bool) -> torch.Tensor:
        """Crop in packed coordinates: random for train, centred for test."""
        cs = self.crop_size
        if cs is None:
            return clean
        _, h, w = clean.shape
        if h < cs or w < cs:
            raise RuntimeError(
                f"Source ({h}x{w} packed) is smaller than crop_size={cs}.")
        if is_train:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top, left = (h - cs) // 2, (w - cs) // 2
        return clean[:, top:top + cs, left:left + cs]

    def _noise_model_for(self, is_train: bool) -> NoiseModel:
        """
        The test split freezes the stochastic *scene* parameters — exposure
        factor and highlight expansion — so the only thing separating two
        benchmark runs is the model under test.
        """
        if is_train:
            return self.noise_model
        p = self.noise_model.params
        if not p.random_alpha and not p.do_expand:
            return self.noise_model
        return NoiseModel(p.replace(random_alpha=False, do_expand=False))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(
                f"index {idx} out of range for {len(self)} samples")

        actual = idx % len(self.entries)
        entry = self.entries[actual]
        is_train = self.split == "train"

        clean_hdr = self._load_clean(entry)
        if clean_hdr.dim() != 3 or clean_hdr.shape[0] != self.expected_channels:
            raise RuntimeError(
                f"{self._entry_id(entry)}: expected "
                f"[{self.expected_channels}, h, w], got "
                f"{tuple(clean_hdr.shape)}.")

        rmin, rmax = self._clean_range(actual, clean_hdr)
        clean_hdr = self._crop(clean_hdr, is_train)

        generator = None
        if not is_train:
            generator = torch.Generator()
            generator.manual_seed(self.test_noise_seed + actual)

        model = self._noise_model_for(is_train)
        noisy, clean, meta = model.apply(
            clean_hdr, norm_min=rmin, norm_max=rmax,
            generator=generator, return_meta=True)

        if self.transform is not None:
            noisy, clean = self.transform(noisy, clean)

        sample: Dict[str, Any] = {
            "x": noisy,
            # 'xm' is a duplicate alias older training scripts expect; the
            # current collate drops it rather than pinning it twice.
            "xm": noisy,
            "y": clean,
        }
        if self.return_meta:
            sample["index"] = actual
            sample["source_id"] = self._entry_id(entry)
            sample["pattern"] = self.pattern
            sample["noise_meta"] = meta
        return sample

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(split='{self.split}', "
                f"sources={self.num_sources}, num_patch={self.num_patch}, "
                f"crop_size={self.crop_size}, pattern='{self.pattern}', "
                f"noise={self.noise_model!r})")
