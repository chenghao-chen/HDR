"""
hdr_data/bayer.py — CFA pattern algebra for packed Bayer tensors
================================================================

The rest of the project speaks one dialect: a *packed* tensor of shape
[..., 4, h, w] whose channel c holds the sensor pixel at intra-cell offset

    (row, col) = (c // 2, c % 2)

so that ``F.pixel_shuffle(packed, 2)`` reconstructs the [..., 1, 2h, 2w]
sensor mosaic and ``F.pixel_unshuffle(mosaic, 2)`` inverts it exactly.
Packed channel order for BGGR is therefore (B, G1, G2, R) — the order
HDR_Mobile_dataset.py writes and HDR_model_hybrid_Teacher.py consumes.

This module generalises that from "BGGR, hardcoded" to "any of the four
CFA phases", because the two new datasets do not arrive as BGGR:

  * i2-2kfps video frames are decoded as RGB and must be *mosaicked* into
    a chosen pattern (see ``rgb_to_packed``);
  * Kalantari 2017 supplies aligned RGB LDR/HDR images, same story.

Pattern naming follows the usual convention: the four letters are the
colours at cell positions (0,0), (0,1), (1,0), (1,1) — so "BGGR" means
B at (0,0), G at (0,1), G at (1,0), R at (1,1).

Flips, transposes and the CFA phase
───────────────────────────────────
A spatial flip of the *sensor mosaic* moves every pixel to a cell offset
with a different colour filter. Two different things are often called
"Bayer-aware augmentation" and this module implements both, explicitly:

  ``FlipMode.PIXEL``  A true flip of the sensor image. The packed
      channels must be permuted to stay consistent with pixel_shuffle,
      and the CFA phase of the result *changes* (hflip: BGGR -> GBRG).
      ``pattern_after()`` reports the new phase. This is what
      train_A100_MoE_two_phase.make_d4_transform does, and what
      tests/test_d4_augmentation.py pins:
          S(hflip(p)[[1, 0, 3, 2]]) == hflip(S(p))
      It is a genuine mosaic flip; the caller is responsible for knowing
      that the phase moved (see augment.py, which does).

  ``FlipMode.CELL``   A flip at 2x2-cell granularity: cells are
      reordered, intra-cell layout is untouched. The picture is mirrored
      to within half a cell and the CFA phase is *preserved*, so the
      result can be fed to a pattern-specific consumer (GBTF, a trained
      model) with no relabelling. This is the safer default for
      augmenting data whose pattern must stay fixed.

Everything here is pure torch, batched or unbatched, autograd-safe, and
device/dtype agnostic.
"""

from __future__ import annotations

import enum
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "CFA_PATTERNS",
    "FlipMode",
    "canonical_pattern",
    "pattern_colors",
    "packed_channel_names",
    "color_index",
    "pack",
    "unpack",
    "cfa_masks",
    "PERM_HFLIP",
    "PERM_VFLIP",
    "PERM_TRANSPOSE",
    "PERM_ROT180",
    "pattern_after",
    "hflip",
    "vflip",
    "rot180",
    "transpose",
    "shift_phase",
    "rgb_to_mosaic",
    "rgb_to_packed",
    "mosaic_to_sparse_rgb",
    "packed_to_sparse_rgb",
    "packed_to_half_rgb",
    "green_channels",
]


# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────

#: The four Bayer phases, as the colours at cell offsets (0,0) (0,1) (1,0) (1,1).
CFA_PATTERNS: Dict[str, Tuple[str, str, str, str]] = {
    "BGGR": ("B", "G", "G", "R"),
    "RGGB": ("R", "G", "G", "B"),
    "GRBG": ("G", "R", "B", "G"),
    "GBRG": ("G", "B", "R", "G"),
}

#: RGB plane index for each colour letter.
_COLOR_TO_INDEX = {"R": 0, "G": 1, "B": 2}


class FlipMode(enum.Enum):
    """How a spatial flip treats the 2x2 CFA cell. See the module docstring."""

    #: True flip of the sensor mosaic; the CFA phase changes.
    PIXEL = "pixel"
    #: Flip whole cells; the CFA phase is preserved.
    CELL = "cell"


def canonical_pattern(pattern: str) -> str:
    """
    Normalise and validate a pattern name.

    >>> canonical_pattern("bggr")
    'BGGR'
    """
    if not isinstance(pattern, str):
        raise TypeError(f"pattern must be a str, got {type(pattern).__name__}")
    key = pattern.strip().upper()
    if key not in CFA_PATTERNS:
        raise ValueError(
            f"Unknown CFA pattern '{pattern}'. "
            f"Expected one of {sorted(CFA_PATTERNS)}."
        )
    return key


def pattern_colors(pattern: str) -> Tuple[str, str, str, str]:
    """The four colour letters of `pattern`, in packed-channel order."""
    return CFA_PATTERNS[canonical_pattern(pattern)]


def packed_channel_names(pattern: str) -> Tuple[str, str, str, str]:
    """
    Human-readable names for the four packed channels, disambiguating the
    two greens by their order of appearance.

    >>> packed_channel_names("BGGR")
    ('B', 'G1', 'G2', 'R')
    >>> packed_channel_names("GRBG")
    ('G1', 'R', 'B', 'G2')
    """
    colors = pattern_colors(pattern)
    names, seen_green = [], 0
    for c in colors:
        if c == "G":
            seen_green += 1
            names.append(f"G{seen_green}")
        else:
            names.append(c)
    return tuple(names)


def color_index(pattern: str) -> Tuple[int, int, int, int]:
    """
    RGB plane index (R=0, G=1, B=2) for each packed channel.

    >>> color_index("BGGR")
    (2, 1, 1, 0)
    """
    return tuple(_COLOR_TO_INDEX[c] for c in pattern_colors(pattern))


def green_channels(pattern: str) -> Tuple[int, int]:
    """The two packed channel indices carrying green, in order."""
    idx = [i for i, c in enumerate(pattern_colors(pattern)) if c == "G"]
    return (idx[0], idx[1])


# ─────────────────────────────────────────────────────────────────────────────
# Pack / unpack
# ─────────────────────────────────────────────────────────────────────────────

def _require_packed(t: torch.Tensor) -> None:
    if t.dim() < 3 or t.shape[-3] != 4:
        raise ValueError(
            f"Expected a packed tensor [..., 4, h, w], got {tuple(t.shape)}."
        )


def _require_mosaic(t: torch.Tensor) -> None:
    if t.dim() < 3 or t.shape[-3] != 1:
        raise ValueError(
            f"Expected a mosaic tensor [..., 1, H, W], got {tuple(t.shape)}."
        )
    if t.shape[-1] % 2 or t.shape[-2] % 2:
        raise ValueError(
            f"Mosaic dimensions must be even, got {tuple(t.shape[-2:])}."
        )


def pack(mosaic: torch.Tensor) -> torch.Tensor:
    """
    [..., 1, 2h, 2w] sensor mosaic -> [..., 4, h, w] packed tensor.

    Unbatched [1, H, W] input is supported and returns [4, h, w].
    """
    _require_mosaic(mosaic)
    if mosaic.dim() == 3:
        return F.pixel_unshuffle(mosaic.unsqueeze(0), 2).squeeze(0)
    return F.pixel_unshuffle(mosaic, 2)


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """
    [..., 4, h, w] packed tensor -> [..., 1, 2h, 2w] sensor mosaic.

    Exact inverse of :func:`pack`.
    """
    _require_packed(packed)
    if packed.dim() == 3:
        return F.pixel_shuffle(packed.unsqueeze(0), 2).squeeze(0)
    return F.pixel_shuffle(packed, 2)


def cfa_masks(pattern: str, height: int, width: int, *,
              device=None, dtype=torch.bool) -> Dict[str, torch.Tensor]:
    """
    Per-colour sampling masks for a full-resolution mosaic of size
    (height, width), as {"R": [1, H, W], "G": ..., "B": ...}.

    Used by the pattern-generic demosaicing baselines, which cannot assume
    BGGR the way DifferentiableGBTF_BGGR does.
    """
    if height % 2 or width % 2:
        raise ValueError(f"Mosaic dims must be even, got ({height}, {width}).")
    colors = pattern_colors(pattern)
    masks = {
        c: torch.zeros((1, height, width), device=device, dtype=dtype)
        for c in ("R", "G", "B")
    }
    one = True if dtype == torch.bool else 1
    for c_idx, color in enumerate(colors):
        r, s = c_idx // 2, c_idx % 2
        masks[color][:, r::2, s::2] = one
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Spatial symmetries
# ─────────────────────────────────────────────────────────────────────────────
#
# These permutations depend only on the packed layout (channel c <-> cell
# offset (c//2, c%2)), never on which colours sit there — so one constant
# serves all four phases. Derivation for the horizontal flip, with
# S(p)[y, x] = p[2*(y%2) + (x%2), y//2, x//2] and even width W:
#
#   S(q)[y, x] = S(p)[y, W-1-x]
#              = p[2*(y%2) + ((W-1-x) % 2), y//2, (W-1-x)//2]
#              = p[2*r + (1-s), i, W/2-1-j]        (r = y%2, s = x%2)
#
# so q = flip(p, dims=(-1,))[[1, 0, 3, 2]]: the value that must land in
# channel 2r+(1-s) is the one from channel 2r+s, and the map is an
# involution, hence the gather list equals the scatter list.

#: q = flip(p, -1)[PERM_HFLIP] is the true horizontal flip of the mosaic.
PERM_HFLIP: Tuple[int, int, int, int] = (1, 0, 3, 2)
#: q = flip(p, -2)[PERM_VFLIP] is the true vertical flip of the mosaic.
PERM_VFLIP: Tuple[int, int, int, int] = (2, 3, 0, 1)
#: q = p.transpose(-2,-1)[PERM_TRANSPOSE] is the true mosaic transpose.
PERM_TRANSPOSE: Tuple[int, int, int, int] = (0, 2, 1, 3)
#: Composition of the horizontal and vertical flips.
PERM_ROT180: Tuple[int, int, int, int] = (3, 2, 1, 0)


def _permute_channels(packed: torch.Tensor,
                      perm: Sequence[int]) -> torch.Tensor:
    """Reindex the packed channel dim (dim -3) by `perm`."""
    index = torch.as_tensor(list(perm), device=packed.device, dtype=torch.long)
    return packed.index_select(-3, index)


def pattern_after(pattern: str, op: str) -> str:
    """
    The CFA phase of the result of a true (``FlipMode.PIXEL``) symmetry.

    op is one of "hflip", "vflip", "rot180", "transpose", "identity".

    >>> pattern_after("BGGR", "hflip")
    'GBRG'
    >>> pattern_after("BGGR", "transpose")
    'BGGR'
    >>> pattern_after("GRBG", "vflip")
    'BGGR'
    """
    colors = pattern_colors(pattern)
    perms = {
        "identity": (0, 1, 2, 3),
        "hflip": PERM_HFLIP,
        "vflip": PERM_VFLIP,
        "rot180": PERM_ROT180,
        "transpose": PERM_TRANSPOSE,
    }
    if op not in perms:
        raise ValueError(
            f"Unknown op '{op}'. Expected one of {sorted(perms)}."
        )
    # Channel c of the result holds what channel perms[op][c] held before,
    # so the colour landing at cell offset c is colors[perms[op][c]].
    out = "".join(colors[perms[op][c]] for c in range(4))
    return canonical_pattern(out)


def hflip(packed: torch.Tensor, pattern: str = "BGGR", *,
          mode: FlipMode = FlipMode.PIXEL) -> Tuple[torch.Tensor, str]:
    """
    Horizontally flip a packed tensor.

    Returns ``(flipped, new_pattern)``. Under ``FlipMode.PIXEL`` the result
    is a true mirror of the sensor image and ``new_pattern`` differs from
    `pattern`; under ``FlipMode.CELL`` cells are mirrored as units and
    ``new_pattern == pattern``.
    """
    _require_packed(packed)
    flipped = torch.flip(packed, dims=(-1,))
    if mode is FlipMode.CELL:
        return flipped, canonical_pattern(pattern)
    return _permute_channels(flipped, PERM_HFLIP), pattern_after(pattern, "hflip")


def vflip(packed: torch.Tensor, pattern: str = "BGGR", *,
          mode: FlipMode = FlipMode.PIXEL) -> Tuple[torch.Tensor, str]:
    """Vertically flip a packed tensor. See :func:`hflip`."""
    _require_packed(packed)
    flipped = torch.flip(packed, dims=(-2,))
    if mode is FlipMode.CELL:
        return flipped, canonical_pattern(pattern)
    return _permute_channels(flipped, PERM_VFLIP), pattern_after(pattern, "vflip")


def rot180(packed: torch.Tensor, pattern: str = "BGGR", *,
           mode: FlipMode = FlipMode.PIXEL) -> Tuple[torch.Tensor, str]:
    """Rotate a packed tensor by 180 degrees. See :func:`hflip`."""
    _require_packed(packed)
    flipped = torch.flip(packed, dims=(-2, -1))
    if mode is FlipMode.CELL:
        return flipped, canonical_pattern(pattern)
    return _permute_channels(flipped, PERM_ROT180), pattern_after(pattern, "rot180")


def transpose(packed: torch.Tensor, pattern: str = "BGGR", *,
              mode: FlipMode = FlipMode.PIXEL) -> Tuple[torch.Tensor, str]:
    """
    Transpose a packed tensor (swaps h and w).

    Note the transpose is the one symmetry that maps every CFA phase to
    itself under ``FlipMode.PIXEL`` when the two greens are
    interchangeable, which is why the training script may apply it freely
    to square patches.
    """
    _require_packed(packed)
    swapped = packed.transpose(-2, -1).contiguous()
    if mode is FlipMode.CELL:
        return swapped, canonical_pattern(pattern)
    return (_permute_channels(swapped, PERM_TRANSPOSE),
            pattern_after(pattern, "transpose"))


def shift_phase(mosaic: torch.Tensor, pattern: str,
                dy: int, dx: int) -> Tuple[torch.Tensor, str]:
    """
    Crop a mosaic by (dy, dx) pixels from the top-left, which advances the
    CFA phase and shrinks the image. Used to bring a flipped image back to
    a required phase at the cost of one row/column.

    dy, dx must be 0 or 1. Returns ``(cropped, new_pattern)``; the crop
    also trims the opposite edge so the result keeps even dimensions.
    """
    if dy not in (0, 1) or dx not in (0, 1):
        raise ValueError(f"dy, dx must each be 0 or 1, got ({dy}, {dx}).")
    _require_mosaic(mosaic)
    if dy == 0 and dx == 0:
        return mosaic, canonical_pattern(pattern)

    # Cropping one pixel off the top/left also costs one off the
    # bottom/right, so the result keeps the even dimensions pack() needs.
    h, w = mosaic.shape[-2], mosaic.shape[-1]
    out = mosaic[..., dy:h - dy, dx:w - dx]

    colors = pattern_colors(pattern)
    # The pixel now at cell offset (r, s) was at (r + dy, s + dx) mod 2.
    new = "".join(
        colors[2 * ((c // 2 + dy) % 2) + ((c % 2 + dx) % 2)] for c in range(4)
    )
    return out.contiguous(), canonical_pattern(new)


# ─────────────────────────────────────────────────────────────────────────────
# RGB <-> CFA
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_mosaic(rgb: torch.Tensor, pattern: str = "BGGR") -> torch.Tensor:
    """
    Sample a full-colour image down to a single-channel CFA mosaic.

    rgb: [..., 3, H, W] with H, W even. Returns [..., 1, H, W].

    This is the "virtual sensor" the video and multi-exposure loaders use
    to turn ordinary RGB frames into inputs the joint denoise+demosaic
    model can consume.
    """
    if rgb.dim() < 3 or rgb.shape[-3] != 3:
        raise ValueError(f"Expected [..., 3, H, W] RGB, got {tuple(rgb.shape)}.")
    h, w = rgb.shape[-2], rgb.shape[-1]
    if h % 2 or w % 2:
        raise ValueError(f"RGB dims must be even, got ({h}, {w}).")

    colors = pattern_colors(pattern)
    out = torch.zeros_like(rgb[..., :1, :, :])
    for c_idx, color in enumerate(colors):
        r, s = c_idx // 2, c_idx % 2
        plane = _COLOR_TO_INDEX[color]
        out[..., 0, r::2, s::2] = rgb[..., plane, r::2, s::2]
    return out


def rgb_to_packed(rgb: torch.Tensor, pattern: str = "BGGR") -> torch.Tensor:
    """
    [..., 3, 2h, 2w] RGB -> [..., 4, h, w] packed CFA, in one step.

    The inverse direction is demosaicing, which is the model's job (see
    hdr_baselines.demosaic for the classical references).
    """
    return pack(rgb_to_mosaic(rgb, pattern))


def mosaic_to_sparse_rgb(mosaic: torch.Tensor,
                         pattern: str = "BGGR") -> torch.Tensor:
    """
    [..., 1, H, W] mosaic -> [..., 3, H, W] with each pixel's measured
    colour in its own plane and zeros elsewhere.

    This is the standard starting point for every classical demosaicing
    algorithm: interpolation then fills the zeros.
    """
    _require_mosaic(mosaic)
    h, w = mosaic.shape[-2], mosaic.shape[-1]
    masks = cfa_masks(pattern, h, w, device=mosaic.device, dtype=mosaic.dtype)
    planes = [mosaic[..., 0, :, :] * masks[c][0] for c in ("R", "G", "B")]
    return torch.stack(planes, dim=-3)


def packed_to_sparse_rgb(packed: torch.Tensor,
                         pattern: str = "BGGR") -> torch.Tensor:
    """[..., 4, h, w] packed -> [..., 3, 2h, 2w] sparse RGB."""
    return mosaic_to_sparse_rgb(unpack(packed), pattern)


def packed_to_half_rgb(packed: torch.Tensor,
                       pattern: str = "BGGR") -> torch.Tensor:
    """
    Cheap half-resolution colour preview: [..., 4, h, w] -> [..., 3, h, w],
    one RGB pixel per CFA cell with the two greens averaged.

    Not a demosaicing algorithm — no interpolation happens — but it is the
    right thing for thumbnails, quick-look visualisations and for the
    "half-res" reference in the noise-model tests.
    """
    _require_packed(packed)
    colors = pattern_colors(pattern)
    planes = []
    for color in ("R", "G", "B"):
        idx = [i for i, c in enumerate(colors) if c == color]
        sel = packed.index_select(
            -3, torch.as_tensor(idx, device=packed.device, dtype=torch.long))
        planes.append(sel.mean(dim=-3))
    return torch.stack(planes, dim=-3)
