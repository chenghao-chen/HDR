"""
HDR_Mobile_dataset.py — MobileHDR packed-Bayer dataset with synthetic noise
============================================================================

Tensors on disk: (4, H, W) float32 packed BGGR Bayer (channels B, G1, G2, R),
unnormalised HDR values (see convert_npz_to_pt.py).

Noise model (high noise — matches the proven recipe in HDR_dataset.py)
──────────────────────────────────────────────────────────────────────
Poisson-Gaussian in digital numbers (DN), signal in [0, pix_max]:

    var[DN²] = shot_gain * signal[DN] + read_var,
    shot_gain = 14, read_var ~ U(135, 160)

At 10 bit this gives σ ≈ 120 DN (12% of full scale) in highlights and
σ ≈ 12 DN in the blacks; on top of that the triangular low-light alpha
(biased toward 0.1) scales the signal down, so dark scenes get severe
relative noise. NOTE: the previous version normalised the signal to [0, 1]
before multiplying by shot_gain, which silently capped the noise variance
at ~14 DN² (~100x too weak).

Efficiency
──────────
* `crop_size` (train): the random crop is taken BEFORE noise synthesis, so
  noise is generated for crop_size² pixels instead of the full frame
  (~12x less dataloader CPU for 512² crops from ~2K x 1.5K frames).
* Tensors are loaded with mmap when available, so a crop only touches the
  pages it needs after the first access.
* Per-file min/max is cached so crops are normalised by the FULL image
  range — crops keep their absolute brightness (a dark crop stays dark).

Determinism
───────────
* File lists are sorted (glob order is filesystem-dependent).
* split="test" draws noise from a per-index torch.Generator, so every run
  benchmarks the exact same noisy inputs.
"""

import os
import glob
import random

import torch
from torch.utils.data import Dataset


def _rand(generator=None) -> float:
    """Uniform [0,1) scalar, optionally from a dedicated generator."""
    return torch.rand((), generator=generator).item()


def add_photon_noise(image: torch.Tensor, nbits: int = 10,
                     random_alpha: bool = True, do_expand: bool = False,
                     shot_gain: float = 14.0,               # ← increased
                     read_noise_range=(135.0, 160.0),       # ← increased
                     norm_min=None, norm_max=None,
                     generator: torch.Generator = None) -> tuple:
    pix_max = float(2 ** nbits - 1)

    rmin = image.min() if norm_min is None else norm_min
    rmax = image.max() if norm_max is None else norm_max
    rng = float(rmax) - float(rmin)
    image = ((image - rmin) / rng * pix_max) if rng > 1e-6 else (image * 0)

    if random_alpha:
        # Peaks at 0 (true extreme low-light), range [0, 1]
        alpha = abs(_rand(generator) - _rand(generator))
        alpha = max(alpha, 0.01)   # avoid pure black
    else:
        alpha = 1.0

    clean = image * alpha

    if do_expand and _rand(generator) < 0.3:
        offset = 2.0 ** (_rand(generator) * 6.0 + 5.0)
        clean = clean + offset
        # Now clipping at pix_max later will actually produce saturation

    clean = clean.clamp(min=0.0)

    read_var = read_noise_range[0] + \
        (read_noise_range[1] - read_noise_range[0]) * _rand(generator)

    noise_sigma = torch.sqrt(shot_gain * clean + read_var)
    noisy = clean + noise_sigma * torch.randn(clean.shape,
                                              generator=generator,
                                              dtype=clean.dtype)

    noisy = torch.round(noisy).clamp_(0.0, pix_max) / pix_max
    gt    = clean.clamp(0.0, pix_max) / pix_max
    return noisy, gt


class MobileHDRDataset(Dataset):
    def __init__(self, base_dir: str, split: str = "train", transform=None,
                 nbits: int = 10, random_alpha: bool = True,
                 num_patch: int = 16, crop_size: int = None,
                 do_expand: bool = False, shot_gain: float = 14.0,
                 read_noise_range=(135.0, 160.0), test_noise_seed: int = 2025):
        """
        crop_size: if set (train only), a random crop_size² crop is taken
            BEFORE noise synthesis and `transform` only needs to handle
            flips/transpose. If None, noise is added to the full frame and
            `transform` may crop (legacy behaviour).
        num_patch: virtual repeats per image per epoch (train only).
        """
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.nbits = nbits
        self.random_alpha = random_alpha
        self.num_patch = num_patch
        self.crop_size = crop_size
        self.do_expand = do_expand
        self.shot_gain = shot_gain
        self.read_noise_range = tuple(read_noise_range)
        self.test_noise_seed = test_noise_seed

        if self.split == "train":
            self.tensor_dir = os.path.join(base_dir, "train", "tensors")
            self.file_list = sorted(glob.glob(
                os.path.join(self.tensor_dir, "**", "*.pt"), recursive=True))
        elif self.split == "test":
            self.tensor_dir = os.path.join(base_dir, "test", "tensors", "with_gt")
            self.file_list = sorted(glob.glob(
                os.path.join(self.tensor_dir, "*.pt")))
        else:
            raise ValueError(f"Unknown split '{split}' (use train | test)")

        if len(self.file_list) == 0:
            raise RuntimeError(f"No .pt tensors found in {self.tensor_dir}.")

        # Per-file (min, max) of the FULL image, filled lazily per worker.
        self._range_cache = {}

    def __len__(self) -> int:
        # Virtual epoch: each image is visited num_patch times with fresh
        # crops/noise. Testing stays 1-to-1.
        if self.split == "train":
            return len(self.file_list) * self.num_patch
        return len(self.file_list)

    # Kept as no-ops: crops and noise are regenerated on every __getitem__.
    def regen_crops(self):
        pass

    def regen_noise(self):
        pass

    def _load_clean(self, path: str) -> torch.Tensor:
        try:
            # mmap: a crop only reads the pages it touches (torch >= 2.1)
            return torch.load(path, weights_only=True, mmap=True)
        except (TypeError, RuntimeError):
            return torch.load(path, weights_only=True)

    def _image_range(self, path: str, tensor: torch.Tensor):
        if path not in self._range_cache:
            self._range_cache[path] = (tensor.min().item(), tensor.max().item())
        return self._range_cache[path]

    def __getitem__(self, idx: int) -> dict:
        actual_idx = idx % len(self.file_list)
        tensor_path = self.file_list[actual_idx]

        clean_hdr = self._load_clean(tensor_path)
        assert clean_hdr.dim() == 3 and clean_hdr.shape[0] == 4, \
            f"Expected [4, H, W] Bayer tensor, got {tuple(clean_hdr.shape)} in {tensor_path}"

        rmin, rmax = self._image_range(tensor_path, clean_hdr)
        is_train = self.split == "train"

        # Crop BEFORE noise synthesis (train): any (top, left) is valid for
        # the packed representation since every packed pixel is a full BGGR
        # cell — the channel-position correspondence is preserved.
        if is_train and self.crop_size is not None:
            cs = self.crop_size
            _, h, w = clean_hdr.shape
            if h < cs or w < cs:
                raise RuntimeError(
                    f"Image {tensor_path} ({h}x{w}) smaller than crop_size={cs}")
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
            clean_hdr = clean_hdr[:, top:top + cs, left:left + cs]

        # Deterministic noise per test index -> reproducible benchmarks.
        gen = None
        if not is_train:
            gen = torch.Generator()
            gen.manual_seed(self.test_noise_seed + actual_idx)

        noisy_hdr, gt_hdr = add_photon_noise(
            clean_hdr,
            nbits=self.nbits,
            random_alpha=self.random_alpha if is_train else False,
            do_expand=self.do_expand if is_train else False,
            shot_gain=self.shot_gain,
            read_noise_range=self.read_noise_range,
            norm_min=rmin, norm_max=rmax,
            generator=gen,
        )

        if self.transform:
            noisy_hdr, gt_hdr = self.transform(noisy_hdr, gt_hdr)

        # 'xm' kept for backward compatibility with older training scripts.
        return {
            'x': noisy_hdr,
            'xm': noisy_hdr,
            'y': gt_hdr,
        }
