"""
hdr_data/sidd.py — SIDD (Smartphone Image Denoising Dataset) raw blocks
========================================================================

SIDD is the standard real-noise benchmark for raw denoising, and it differs
from every other dataset in this package in one structural way: it ships
*real* noisy/clean pairs. :class:`hdr_data.base.BaseBayerDataset` exists to
synthesise noise onto a clean tensor, so this loader deliberately does not
subclass it — there is no clean-only signal to add noise to, and routing
real pairs through the synthetic path would either double-corrupt them or
require disabling most of that class.

Task shape
──────────
SIDD scores RAW->RAW denoising: the model takes a noisy mosaic and returns a
denoised mosaic, with no demosaicing. That is what ``upsample=False`` and
``out_channels=4`` build in :func:`HDR_model_hybrid_Teacher.build_denoiser`.

CFA and why the metric does not care
────────────────────────────────────
The block files carry no CFA metadata, and SIDD spans five phones with
differing patterns. That matters for a demosaicing model, but not here:
packing is a reversible rearrangement of the same pixels, so PSNR computed
on the reconstructed mosaic is identical whichever phase we assume, as long
as pack and unpack agree. The pattern is still recorded on each sample so a
caller that cares (e.g. one exploiting colour structure) can use it.

Validation blocks
─────────────────
    ValidationNoisyBlocksRaw.mat   (40, 32, 256, 256) float32 in [0, 1]
    ValidationGtBlocksRaw.mat      same

40 scenes x 32 blocks = 1280 patches, black-level subtracted and normalised
by the dataset authors. Stored as MATLAB v5, so scipy.io.loadmat reads them;
v7.3 files would need h5py, and the loader falls back accordingly.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from .bayer import pack

__all__ = ["SIDDValidationRaw", "SIDDMediumRaw", "SIDD_CAMERA_PATTERNS",
           "build_cache"]


def _load_mat(path: str, key_hint: str):
    """Read a MATLAB array, whichever container version it uses."""
    try:
        from scipy.io import loadmat
        data = loadmat(path)
        keys = [k for k in data if not k.startswith("__")]
    except (NotImplementedError, ValueError):
        # v7.3 is HDF5; scipy raises rather than reading it.
        import h5py
        with h5py.File(path, "r") as fh:
            keys = [k for k in fh if not k.startswith("#")]
            key = key_hint if key_hint in keys else keys[0]
            # h5py returns the MATLAB array transposed.
            return torch.from_numpy(fh[key][()]).permute(
                *reversed(range(fh[key].ndim))).contiguous()
    if not keys:
        raise RuntimeError(f"{path}: no data arrays found")
    key = key_hint if key_hint in keys else keys[0]
    return torch.from_numpy(data[key])


class SIDDValidationRaw(Dataset):
    """
    SIDD validation raw blocks as packed-Bayer noisy/clean pairs.

    Parameters
    ──────────
    root
        Directory holding ValidationNoisyBlocksRaw.mat and
        ValidationGtBlocksRaw.mat.
    pattern
        CFA phase recorded on each sample. It does not affect the pixels or
        the metric (see the module docstring), only what downstream code is
        told the packing means.
    limit
        Keep only the first N blocks, for smoke runs.
    """

    expected_channels = 4

    def __init__(self, root: str, pattern: str = "RGGB",
                 limit: Optional[int] = None, return_meta: bool = False):
        noisy_path = os.path.join(root, "ValidationNoisyBlocksRaw.mat")
        gt_path = os.path.join(root, "ValidationGtBlocksRaw.mat")
        for p in (noisy_path, gt_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"{p} not found. Fetch the SIDD validation blocks into "
                    f"{root} (see hdr_data/sidd.py).")

        noisy = _load_mat(noisy_path, "ValidationNoisyBlocksRaw").float()
        clean = _load_mat(gt_path, "ValidationGtBlocksRaw").float()
        if noisy.shape != clean.shape:
            raise RuntimeError(
                f"noisy {tuple(noisy.shape)} and GT {tuple(clean.shape)} "
                "blocks disagree in shape")

        # (scenes, blocks, H, W) -> (scenes*blocks, H, W)
        self.noisy = noisy.reshape(-1, *noisy.shape[-2:])
        self.clean = clean.reshape(-1, *clean.shape[-2:])
        self.scenes_per_block = noisy.shape[1]
        if limit is not None:
            self.noisy = self.noisy[:limit]
            self.clean = self.clean[:limit]

        self.pattern = pattern
        self.return_meta = bool(return_meta)
        self.num_sources = self.noisy.shape[0]
        self.num_patch = 1

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # pack expects a mosaic with an explicit channel axis: [1, H, W].
        x = pack(self.noisy[idx].unsqueeze(0))
        y = pack(self.clean[idx].unsqueeze(0))
        sample: Dict[str, Any] = {"x": x, "xm": x, "y": y}
        if self.return_meta:
            sample["index"] = idx
            sample["source_id"] = f"scene{idx // self.scenes_per_block:02d}" \
                                  f"_block{idx % self.scenes_per_block:02d}"
            sample["pattern"] = self.pattern
            sample["noise_meta"] = {"real": True}
        return sample

    def __repr__(self) -> str:
        return (f"SIDDValidationRaw(blocks={len(self)}, "
                f"pattern='{self.pattern}', real_noise=True)")


# Camera code -> CFA pattern, from SIDD_Medium_Raw/bayer_patterns.csv. The
# scene-directory name carries the camera code in its third field, e.g.
# 0014_001_S6_03200_01250_3200_N -> S6 -> GRBG.
SIDD_CAMERA_PATTERNS = {
    "GP": "BGGR",   # Google Pixel
    "IP": "RGGB",   # iPhone 7
    "S6": "GRBG",   # Samsung Galaxy S6 Edge
    "N6": "BGGR",   # Motorola Nexus 6
    "G4": "BGGR",   # LG G4
}


class SIDDMediumRaw(Dataset):
    """
    SIDD-Medium raw training pairs, cropped to packed-Bayer patches.

    320 real noisy/clean pairs over 160 scene instances. Each .MAT is a
    MATLAB 7.3 (HDF5) file holding one full-resolution plane under key 'x',
    black-level subtracted and normalised to [0, 1] by the dataset authors.
    Files are opened lazily and sliced, never read whole: a single GT plane
    is ~60 MB, so materialising one per sample would dominate the step time.

    Crops are taken at even offsets. An odd offset shifts the Bayer phase,
    which would silently relabel every colour plane in the packed tensor and
    make the crop inconsistent with both its own pair and the recorded
    pattern.

    The CFA pattern is recorded per sample but planes are NOT rotated to a
    canonical phase. Normalising the phase would let the model rely on
    channel 0 always being red, but the official validation blocks ship no
    CFA metadata, so the same normalisation cannot be applied at evaluation
    time -- and a train/test mismatch here is far more costly than the
    consistency is worth. Packing is reversible either way, so the metric is
    unaffected.
    """

    expected_channels = 4

    def __init__(self, root: str, crop_size: int = 256, num_patch: int = 8,
                 transform=None, return_meta: bool = False,
                 scenes: Optional[list] = None,
                 cache_dir: Optional[str] = None):
        data_dir = os.path.join(root, "SIDD_Medium_Raw", "Data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"{data_dir} not found. Extract SIDD_Medium_Raw.zip into {root}.")

        if crop_size % 2:
            raise ValueError(f"crop_size must be even, got {crop_size}")

        self.pairs = []
        for scene in sorted(os.listdir(data_dir)):
            if scenes is not None and scene not in scenes:
                continue
            sdir = os.path.join(data_dir, scene)
            if not os.path.isdir(sdir):
                continue
            camera = scene.split("_")[2] if len(scene.split("_")) > 2 else ""
            pattern = SIDD_CAMERA_PATTERNS.get(camera, "RGGB")
            gts = sorted(f for f in os.listdir(sdir) if "_GT_RAW_" in f)
            for gt in gts:
                noisy = gt.replace("_GT_RAW_", "_NOISY_RAW_")
                if os.path.isfile(os.path.join(sdir, noisy)):
                    self.pairs.append((os.path.join(sdir, gt),
                                       os.path.join(sdir, noisy), pattern))
        if not self.pairs:
            raise RuntimeError(f"no GT/NOISY pairs found under {data_dir}")

        self.crop_size = crop_size
        self.num_patch = max(1, int(num_patch))
        self.transform = transform
        self.return_meta = bool(return_meta)
        self.num_sources = len(self.pairs)
        self._shapes: Dict[str, Any] = {}

        # The released planes are gzip-compressed HDF5 in 5-row chunks, so a
        # 256x256 crop decompresses ~50 chunks across two files on Lustre for
        # every sample -- enough to make training loader-bound at every model
        # size. With a cache built by build_cache(), crops are slices of
        # uncompressed packed memmaps, which the page cache shares between
        # the ranks on a node.
        self.cache_dir = cache_dir
        self._mmaps: Dict[str, Any] = {}
        if cache_dir is not None:
            missing = [g for g, n, _ in self.pairs
                       if not (os.path.isfile(_cache_path(cache_dir, g))
                               and os.path.isfile(_cache_path(cache_dir, n)))]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(self.pairs)} pairs missing from "
                    f"cache {cache_dir}; run scripts/cache_sidd_raw.py first "
                    f"(first missing: {missing[0]})")

    def __len__(self) -> int:
        return len(self.pairs) * self.num_patch

    def _plane_shape(self, path: str):
        """(H, W) of a plane, cached — h5py stores MATLAB arrays transposed."""
        if path not in self._shapes:
            import h5py
            with h5py.File(path, "r") as fh:
                w, h = fh["x"].shape
            self._shapes[path] = (h, w)
        return self._shapes[path]

    @staticmethod
    def _read_crop(path: str, top: int, left: int, size: int) -> torch.Tensor:
        import h5py
        with h5py.File(path, "r") as fh:
            # Stored transposed, so the file axes are (W, H).
            block = fh["x"][left:left + size, top:top + size]
        return torch.from_numpy(block).float().t().contiguous()

    def _mmap(self, path: str):
        import numpy as np
        if path not in self._mmaps:
            self._mmaps[path] = np.load(_cache_path(self.cache_dir, path),
                                        mmap_mode="r")
        return self._mmaps[path]

    def _cached_pair(self, gt_path: str, noisy_path: str):
        gt, nz = self._mmap(gt_path), self._mmap(noisy_path)
        _, hp, wp = gt.shape
        cs = min(self.crop_size // 2, hp, wp)
        # Packed coordinates: every offset is a whole 2x2 cell, so the CFA
        # phase is preserved by construction.
        top = int(torch.randint(0, hp - cs + 1, (1,)).item())
        left = int(torch.randint(0, wp - cs + 1, (1,)).item())
        y = torch.from_numpy(gt[:, top:top + cs, left:left + cs].copy())
        x = torch.from_numpy(nz[:, top:top + cs, left:left + cs].copy())
        return x, y

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gt_path, noisy_path, pattern = self.pairs[idx % len(self.pairs)]
        if self.cache_dir is not None:
            x, y = self._cached_pair(gt_path, noisy_path)
            if self.transform is not None:
                x, y = self.transform(x, y)
            sample: Dict[str, Any] = {"x": x, "xm": x, "y": y}
            if self.return_meta:
                sample["index"] = idx % len(self.pairs)
                sample["source_id"] = os.path.basename(os.path.dirname(gt_path))
                sample["pattern"] = pattern
                sample["noise_meta"] = {"real": True}
            return sample

        h, w = self._plane_shape(gt_path)
        cs = min(self.crop_size, h, w)
        cs -= cs % 2

        # Even offsets only: an odd offset rotates the CFA phase.
        top = int(torch.randint(0, max(1, (h - cs) // 2 + 1), (1,)).item()) * 2
        left = int(torch.randint(0, max(1, (w - cs) // 2 + 1), (1,)).item()) * 2
        top, left = min(top, h - cs), min(left, w - cs)

        clean = self._read_crop(gt_path, top, left, cs).unsqueeze(0)
        noisy = self._read_crop(noisy_path, top, left, cs).unsqueeze(0)

        x, y = pack(noisy), pack(clean)
        if self.transform is not None:
            x, y = self.transform(x, y)

        sample: Dict[str, Any] = {"x": x, "xm": x, "y": y}
        if self.return_meta:
            sample["index"] = idx % len(self.pairs)
            sample["source_id"] = os.path.basename(os.path.dirname(gt_path))
            sample["pattern"] = pattern
            sample["noise_meta"] = {"real": True}
        return sample

    def __repr__(self) -> str:
        return (f"SIDDMediumRaw(pairs={len(self.pairs)}, "
                f"crop={self.crop_size}, num_patch={self.num_patch})")


def _cache_path(cache_dir: str, plane_path: str) -> str:
    """<cache>/<scene>/<file stem>.npy for a released .MAT plane."""
    scene = os.path.basename(os.path.dirname(plane_path))
    stem = os.path.splitext(os.path.basename(plane_path))[0]
    return os.path.join(cache_dir, scene, stem + ".npy")


def _cache_one(args) -> str:
    import numpy as np
    import h5py
    src, dst = args
    if os.path.isfile(dst):
        return dst
    with h5py.File(src, "r") as fh:
        plane = torch.from_numpy(fh["x"][()]).float().t().contiguous()
    h, w = plane.shape
    plane = plane[: h - h % 2, : w - w % 2]
    packed = pack(plane.unsqueeze(0)).numpy().astype("float32")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp.npy"
    np.save(tmp, packed)
    # Rename into place last, so an interrupted build never leaves a
    # truncated file that a later run would trust.
    os.replace(tmp, dst)
    return dst


def build_cache(root: str, cache_dir: str, workers: int = 8) -> int:
    """
    Decompress every SIDD-Medium raw plane once into an uncompressed packed
    float32 .npy, so training reads crops as memmap slices. Values are kept
    exactly as released (float32, black-level subtracted, [0, 1]) -- no
    requantisation -- so results on the cache match results on the .MATs.
    Idempotent: existing files are skipped.
    """
    from multiprocessing import Pool
    ds = SIDDMediumRaw(root, crop_size=2, num_patch=1)
    jobs = []
    for gt, noisy, _ in ds.pairs:
        for src in (gt, noisy):
            jobs.append((src, _cache_path(cache_dir, src)))
    with Pool(max(1, workers)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cache_one, jobs), 1):
            if i % 40 == 0 or i == len(jobs):
                print(f"  cached {i}/{len(jobs)}", flush=True)
    return len(jobs)
