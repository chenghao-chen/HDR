"""
hdr_data/video_i2.py — the i2-2kfps high-speed video corpus
============================================================

``datasets/i2-2kfps_v1`` is 127 mp4 clips (96 train / 31 test) of
1024x512 high-frame-rate footage. Unlike MobileHDR it is not packed Bayer
and not even raw: it is decoded RGB. To feed the joint denoise+demosaic
model it must go through a *virtual sensor* —

    decode frame -> RGB float -> (optional) sRGB linearisation
                 -> CFA mosaic -> packed [4, h, w] -> noise synthesis

— which is what this loader does. The linearisation step matters: the
Poisson-Gaussian model is stated in linear photons, so applying it to
gamma-encoded values would put most of the noise in the wrong places.

Two source layouts are supported
────────────────────────────────
``.mp4`` clips
    The native layout. Frames are decoded on demand with OpenCV and the
    per-clip frame count is cached to a sidecar JSON so start-up does not
    re-probe 127 files every run.

a directory of extracted frames
    ``<root>/<split>/<clip>/frame_00000.png`` (or .npy). Slower to
    produce, but immune to the decoder problem below — and the only way to
    use this corpus on a host whose OpenCV cannot decode.

Decoder health
──────────────
OpenCV's bundled ffmpeg cannot always initialise its scaling context; on
the Polaris login nodes every ``read()`` then returns a frame-shaped
buffer of garbage while still reporting success, with only a
``swscaler ... Failed initializing scaling graph`` line on stderr to show
for it. Silently training on that would be far worse than crashing, so
:func:`probe_decoder` write-decodes a known pattern and verifies it comes
back, and the dataset refuses to build an mp4-backed index until that
check passes. Run it on the machine that will actually read the data — a
compute node, typically, not the login node.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .base import BaseBayerDataset
from .bayer import rgb_to_packed

__all__ = [
    "I2VideoDataset",
    "probe_decoder",
    "DecoderUnavailable",
    "srgb_to_linear",
    "linear_to_srgb",
    "count_frames",
]

#: Extensions accepted when reading a directory of pre-extracted frames.
FRAME_EXTENSIONS = (".png", ".npy", ".tif", ".tiff", ".jpg", ".jpeg")


class DecoderUnavailable(RuntimeError):
    """Raised when video decoding is impossible or demonstrably broken."""


# ─────────────────────────────────────────────────────────────────────────────
# Colour transfer
# ─────────────────────────────────────────────────────────────────────────────

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Invert the sRGB transfer function: display-encoded [0,1] -> linear.

    Uses the exact piecewise definition rather than a gamma-2.2
    approximation; the linear segment near black is precisely the region a
    low-light dataset lives in.
    """
    return torch.where(x <= 0.04045,
                       x / 12.92,
                       ((x.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Forward sRGB transfer function: linear -> display-encoded [0,1]."""
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308,
                       x * 12.92,
                       1.055 * x ** (1.0 / 2.4) - 0.055)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder plumbing
# ─────────────────────────────────────────────────────────────────────────────

def _import_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:                       # pragma: no cover
        raise DecoderUnavailable(
            "OpenCV (cv2) is required to read .mp4 clips. Install "
            "opencv-python, or point the dataset at a directory of "
            "pre-extracted frames with frames_root=."
        ) from exc
    return cv2


def probe_decoder(tmp_dir: Optional[str] = None,
                  raise_on_failure: bool = False) -> Dict[str, Any]:
    """
    Check that this host can actually decode video, not merely open it.

    Writes a six-frame clip of flat, known grey levels, reads it back and
    compares. A working decoder reproduces the levels to within codec
    tolerance; the broken-swscaler failure mode returns identical garbage
    for every frame, which this catches by checking both the levels and
    that the frames differ from one another.

    Returns a dict with ``ok`` plus diagnostics. With
    ``raise_on_failure=True`` a failure raises :class:`DecoderUnavailable`
    carrying the same detail.
    """
    import tempfile

    result: Dict[str, Any] = {"ok": False, "reason": None, "levels": None}
    try:
        cv2 = _import_cv2()
    except DecoderUnavailable as exc:
        result["reason"] = str(exc)
        if raise_on_failure:
            raise
        return result

    import numpy as np

    tmp_dir = tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"_hdr_decoder_probe_{os.getpid()}.mp4")
    levels = [20, 60, 100, 140, 180, 220]

    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 30.0, (64, 64))
        if not writer.isOpened():
            result["reason"] = "cv2.VideoWriter could not be opened (no codec)."
            if raise_on_failure:
                raise DecoderUnavailable(result["reason"])
            return result
        for lv in levels:
            writer.write(np.full((64, 64, 3), lv, np.uint8))
        writer.release()

        cap = cv2.VideoCapture(path)
        got: List[float] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            got.append(float(frame.mean()))
        cap.release()
        result["levels"] = got

        if len(got) < len(levels):
            result["reason"] = (
                f"decoded {len(got)} of {len(levels)} frames written")
        elif max(got) - min(got) < 1.0:
            result["reason"] = (
                "every decoded frame has the same mean level "
                f"({got[0]:.3f}) although six different levels were written "
                "— the decoder is returning an unconverted buffer. This is "
                "the OpenCV/ffmpeg 'Failed initializing scaling graph' "
                "failure; try a compute node, a different OpenCV build, or "
                "pre-extracted frames.")
        else:
            err = max(abs(g - lv) for g, lv in zip(got, levels))
            if err > 12.0:
                result["reason"] = (
                    f"decoded levels differ from written levels by up to "
                    f"{err:.1f} DN, more than codec tolerance")
            else:
                result["ok"] = True
                result["max_level_error"] = err
    except Exception as exc:                          # pragma: no cover
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    if raise_on_failure and not result["ok"]:
        raise DecoderUnavailable(result["reason"] or "unknown decoder failure")
    return result


def count_frames(path: str) -> int:
    """
    Number of frames in a clip, from container metadata.

    Metadata frame counts can be wrong for some containers, so the loader
    treats this as an upper bound and tolerates a short read.
    """
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise DecoderUnavailable(f"Could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


class _ClipReader:
    """
    A single open capture, reused across sequential reads.

    Random access to a compressed clip means seeking to the nearest
    keyframe and decoding forward, so requesting frames in order is
    dramatically cheaper than jumping around. This holds one clip open and
    only seeks when the requested frame is not the next one — which, with
    a sequential sampler, is almost never.
    """

    def __init__(self, path: str):
        self.cv2 = _import_cv2()
        self.path = str(path)
        self.cap = self.cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise DecoderUnavailable(f"Could not open video: {self.path}")
        self._next_index = 0

    def read(self, frame_index: int):
        """Return the BGR uint8 frame at `frame_index`, or raise."""
        if frame_index != self._next_index:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            self._next_index = frame_index
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise DecoderUnavailable(
                f"Failed to read frame {frame_index} of {self.path}")
        self._next_index = frame_index + 1
        return frame

    def close(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):                                # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class I2VideoDataset(BaseBayerDataset):
    """
    Frames from the i2-2kfps clips, mosaicked into packed CFA.

    Parameters
    ──────────
    base_dir
        Root holding ``train/`` and ``test/`` of .mp4 clips.
    frames_root
        Use pre-extracted frames instead of decoding: a directory whose
        immediate children are per-clip folders of image or .npy frames.
        When set, no video decoding happens and `base_dir` is ignored.
    frame_stride
        Keep every Nth frame. High-speed footage is heavily redundant
        between adjacent frames, so a stride well above 1 buys diversity
        per byte read. Default 8.
    max_frames_per_clip
        Cap on frames taken from any one clip, so a long clip cannot
        dominate the epoch.
    linearise
        Undo the sRGB transfer function after decoding (default True).
        Turn it off only if the source is already linear-light.
    scale
        Multiplier applied after linearisation. The i2 clips are
        photon-starved — most frames sit near zero with sparse bright
        events — so the default 1.0 keeps them dark, which is realistic;
        raise it to simulate a longer exposure.
    check_decoder
        Run :func:`probe_decoder` before building an mp4-backed index and
        refuse to continue if it fails. Leave this on.
    cache_index
        Write/read the per-clip frame counts to ``<base_dir>/
        .i2_frame_counts.json`` so start-up does not re-probe every clip.

    All other arguments are :class:`BaseBayerDataset`'s. Note the CFA
    pattern defaults to BGGR to match the rest of the project: the frames
    have no real CFA, so the choice is ours and consistency wins.
    """

    #: Sidecar file holding cached frame counts.
    INDEX_CACHE_NAME = ".i2_frame_counts.json"

    def __init__(self, base_dir: Optional[str] = None, *,
                 frames_root: Optional[str] = None,
                 frame_stride: int = 8,
                 max_frames_per_clip: Optional[int] = None,
                 linearise: bool = True, scale: float = 1.0,
                 check_decoder: bool = True, cache_index: bool = True,
                 **kwargs):
        if base_dir is None and frames_root is None:
            raise ValueError("Pass base_dir (mp4 clips) or frames_root.")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if max_frames_per_clip is not None and max_frames_per_clip < 1:
            raise ValueError(
                f"max_frames_per_clip must be >= 1, got {max_frames_per_clip}")

        self.base_dir = str(base_dir) if base_dir is not None else None
        self.frames_root = str(frames_root) if frames_root is not None else None
        self.frame_stride = int(frame_stride)
        self.max_frames_per_clip = max_frames_per_clip
        self.linearise = bool(linearise)
        self.scale = float(scale)
        self.check_decoder = bool(check_decoder)
        self.cache_index = bool(cache_index)
        self._reader: Optional[_ClipReader] = None
        kwargs.setdefault("pattern", "BGGR")
        super().__init__(**kwargs)

    # ── index ────────────────────────────────────────────────────────────
    @property
    def uses_video(self) -> bool:
        """True when frames come from mp4 clips rather than a frames dir."""
        return self.frames_root is None

    @property
    def split_dir(self) -> str:
        root = self.frames_root if self.frames_root else self.base_dir
        return os.path.join(root, self.split)

    def _clip_paths(self) -> List[str]:
        d = self.split_dir
        if not os.path.isdir(d):
            return []
        if self.uses_video:
            return sorted(
                os.path.join(d, f) for f in os.listdir(d)
                if f.lower().endswith(".mp4"))
        return sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if os.path.isdir(os.path.join(d, f)))

    def _frame_files(self, clip_dir: str) -> List[str]:
        return sorted(
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
            if f.lower().endswith(FRAME_EXTENSIONS))

    def _load_count_cache(self) -> Dict[str, int]:
        if not (self.cache_index and self.base_dir):
            return {}
        path = os.path.join(self.base_dir, self.INDEX_CACHE_NAME)
        try:
            with open(path) as fh:
                data = json.load(fh)
            return {str(k): int(v) for k, v in data.items()}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_count_cache(self, counts: Dict[str, int]) -> None:
        if not (self.cache_index and self.base_dir):
            return
        path = os.path.join(self.base_dir, self.INDEX_CACHE_NAME)
        try:
            merged = self._load_count_cache()
            merged.update(counts)
            with open(path, "w") as fh:
                json.dump(merged, fh, indent=1, sort_keys=True)
        except OSError:
            # A read-only dataset directory is not a reason to fail; the
            # cache is an optimisation, and re-probing merely costs time.
            pass

    def _build_index(self) -> Sequence[Tuple[str, int]]:
        clips = self._clip_paths()
        if not clips:
            return []

        if self.uses_video and self.check_decoder:
            probe = probe_decoder()
            if not probe["ok"]:
                raise DecoderUnavailable(
                    f"Video decoding is not usable on this host: "
                    f"{probe['reason']}\n"
                    f"Either run where OpenCV can decode, or extract frames "
                    f"once and pass frames_root=<dir>. Pass "
                    f"check_decoder=False only if you have verified decoding "
                    f"another way.")

        entries: List[Tuple[str, int]] = []
        cached = self._load_count_cache()
        fresh: Dict[str, int] = {}

        for clip in clips:
            key = os.path.basename(clip)
            if self.uses_video:
                n = cached.get(key)
                if n is None:
                    n = count_frames(clip)
                    fresh[key] = n
            else:
                n = len(self._frame_files(clip))
            if n <= 0:
                continue
            picked = list(range(0, n, self.frame_stride))
            if self.max_frames_per_clip is not None:
                picked = picked[:self.max_frames_per_clip]
            entries.extend((clip, i) for i in picked)

        if fresh:
            self._save_count_cache(fresh)
        return entries

    def _empty_index_hint(self) -> str:
        return (f"Looked in {self.split_dir} for "
                f"{'*.mp4 clips' if self.uses_video else 'per-clip frame directories'}.")

    def _entry_id(self, entry: Tuple[str, int]) -> str:
        clip, frame = entry
        return f"{os.path.splitext(os.path.basename(clip))[0]}_f{frame:05d}"

    # ── loading ──────────────────────────────────────────────────────────
    def _decode_video_frame(self, clip: str, frame_index: int):
        if self._reader is None or self._reader.path != clip:
            if self._reader is not None:
                self._reader.close()
            self._reader = _ClipReader(clip)
        return self._reader.read(frame_index)

    def _load_frame_rgb(self, entry: Tuple[str, int]) -> torch.Tensor:
        """Return the frame as [3, H, W] float32 RGB in [0, 1]."""
        clip, frame_index = entry
        if self.uses_video:
            bgr = self._decode_video_frame(clip, frame_index)
            arr = bgr[:, :, ::-1].copy()                     # BGR -> RGB
            rgb = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        else:
            files = self._frame_files(clip)
            path = files[frame_index]
            if path.lower().endswith(".npy"):
                import numpy as np
                arr = np.load(path)
                t = torch.from_numpy(np.ascontiguousarray(arr)).float()
                if t.dim() == 2:
                    t = t.unsqueeze(0).expand(3, -1, -1).clone()
                elif t.shape[-1] in (3, 4):
                    t = t[..., :3].permute(2, 0, 1).contiguous()
                if t.dtype == torch.float32 and float(t.max()) > 1.5:
                    t = t / 255.0
                rgb = t.float()
            else:
                from PIL import Image
                import numpy as np
                with Image.open(path) as im:
                    arr = np.asarray(im.convert("RGB"))
                rgb = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0

        if rgb.shape[0] != 3:
            raise RuntimeError(
                f"{self._entry_id(entry)}: expected 3 colour channels, got "
                f"{rgb.shape[0]}")
        return rgb

    def _load_clean(self, entry: Tuple[str, int]) -> torch.Tensor:
        rgb = self._load_frame_rgb(entry)

        if self.linearise:
            rgb = srgb_to_linear(rgb)
        if self.scale != 1.0:
            rgb = rgb * self.scale

        # The mosaic needs even dimensions, and the model's three
        # PixelUnshuffle(2) stages need the packed dims divisible by 8 —
        # so trim to a multiple of 16 in sensor pixels.
        h, w = rgb.shape[-2], rgb.shape[-1]
        h -= h % 16
        w -= w % 16
        if h < 16 or w < 16:
            raise RuntimeError(
                f"{self._entry_id(entry)}: frame {rgb.shape[-2]}x"
                f"{rgb.shape[-1]} is too small after trimming to a multiple "
                f"of 16.")
        rgb = rgb[..., :h, :w]
        return rgb_to_packed(rgb, self.pattern)

    def close(self) -> None:
        """Release the held capture. Called automatically at teardown."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self):                                # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
