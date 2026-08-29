"""
hdr_data/augment.py — CFA-aware augmentation
=============================================

Augmenting packed Bayer is not augmenting an image. Every spatial symmetry
moves pixels onto cell positions with different colour filters, so a naive
flip silently colour-scrambles the data: shapes stay right, the loss still
goes down, and the model learns the wrong demosaicing prior.

There are exactly two consistent ways out, and this module implements both
rather than picking one silently (see hdr_data.bayer for the derivation):

``mode="cell"`` — **the default.** Flip and transpose whole 2x2 cells,
    leaving the intra-cell layout alone. All eight D4 symmetries are
    available and the CFA phase is *preserved*, so the augmented tensor is
    still BGGR (or whatever it started as) and can be fed to a
    pattern-specific consumer — GBTF, a trained model — unchanged. The
    picture is mirrored to within half a cell, which is not a defect worth
    caring about in an augmentation.

``mode="pixel"`` — A true mirror of the sensor mosaic, at the cost of the
    phase moving with it: hflip takes BGGR to GBRG. The transform reports
    the resulting pattern in ``last_pattern`` and, with
    ``restrict_to_pattern=True``, will only apply the symmetries that map
    the pattern to itself (for BGGR: identity and transpose).

``mode="legacy"`` — Bit-compatible with
    ``train_A100_MoE_two_phase.make_d4_transform``: pixel-mode flips with
    the phase change left unrecorded. Provided so existing runs stay
    reproducible. Note what this means in practice: an hflipped sample is
    GBRG data presented to the model as BGGR, and its GT is demosaiced by
    the BGGR-specific GBTF, so the R and B assignments swap on half the
    augmented samples. Use "cell" for new training.

All transforms take and return the ``(noisy, clean)`` pair together and
apply one shared random draw to both, because the pair must stay
pixel-aligned or the supervision target no longer matches the input.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from .bayer import (
    FlipMode,
    PERM_HFLIP,
    PERM_TRANSPOSE,
    PERM_VFLIP,
    canonical_pattern,
    pattern_after,
)

__all__ = [
    "Compose",
    "D4Transform",
    "RandomCropPacked",
    "CenterCropPacked",
    "make_d4_transform_compat",
    "identity_transform",
]

Pair = Tuple[torch.Tensor, torch.Tensor]
PairTransform = Callable[[torch.Tensor, torch.Tensor], Pair]


def _permute(t: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
    """Reindex the packed channel dim of an unbatched [4, h, w] tensor."""
    return t[list(perm)]


def identity_transform(noisy: torch.Tensor, clean: torch.Tensor) -> Pair:
    """A no-op transform, useful as a default argument."""
    return noisy, clean


class Compose:
    """
    Chain pair-transforms left to right.

        tf = Compose([RandomCropPacked(256), D4Transform("BGGR")])
        noisy, clean = tf(noisy, clean)
    """

    def __init__(self, transforms: Sequence[PairTransform]):
        self.transforms: List[PairTransform] = list(transforms)

    def __call__(self, noisy: torch.Tensor, clean: torch.Tensor) -> Pair:
        for t in self.transforms:
            noisy, clean = t(noisy, clean)
        return noisy, clean

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


class D4Transform:
    """
    Random dihedral (D4) augmentation for a packed CFA pair.

    Parameters
    ──────────
    pattern
        The CFA phase of the incoming tensors.
    allow_transpose
        Transpose swaps h and w, so it must be disabled for non-square
        inputs — Phase 2 full frames are not square, and a transposed
        sample would break ``collate_pad_to_max``'s aspect assumptions.
    mode
        ``"cell"`` (default), ``"pixel"`` or ``"legacy"``; see the module
        docstring.
    restrict_to_pattern
        Pixel mode only: skip any symmetry that would change the CFA
        phase, so the output pattern is guaranteed to equal the input's.
    p
        Probability of applying each of the (up to) three independent
        symmetries — hflip, vflip, transpose. The default 0.5 for each
        gives all eight D4 symmetries with equal probability.
    rng
        Optional ``random.Random`` for reproducible draws; defaults to the
        global ``random`` module, matching the training script.

    After a call, ``last_pattern`` holds the CFA phase of the output and
    ``last_ops`` the symmetries that were applied.
    """

    def __init__(self, pattern: str = "BGGR", *, allow_transpose: bool = True,
                 mode: str = "cell", restrict_to_pattern: bool = False,
                 p: float = 0.5, rng: Optional[random.Random] = None):
        if mode not in ("cell", "pixel", "legacy"):
            raise ValueError(
                f"mode must be 'cell', 'pixel' or 'legacy', got '{mode}'")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.pattern = canonical_pattern(pattern)
        self.allow_transpose = bool(allow_transpose)
        self.mode = mode
        self.restrict_to_pattern = bool(restrict_to_pattern)
        self.p = float(p)
        self._rng = rng
        self.last_pattern = self.pattern
        self.last_ops: Tuple[str, ...] = ()

    # -- internals --------------------------------------------------------
    def _draw(self) -> float:
        return (self._rng.random() if self._rng is not None else random.random())

    def _allows(self, op: str, pattern: str) -> bool:
        """Whether `op` may be applied given the phase-preservation policy."""
        if self.mode != "pixel" or not self.restrict_to_pattern:
            return True
        return pattern_after(pattern, op) == pattern

    def _apply_op(self, noisy: torch.Tensor, clean: torch.Tensor,
                  op: str, pattern: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if op == "hflip":
            noisy, clean = torch.flip(noisy, (-1,)), torch.flip(clean, (-1,))
            perm = PERM_HFLIP
        elif op == "vflip":
            noisy, clean = torch.flip(noisy, (-2,)), torch.flip(clean, (-2,))
            perm = PERM_VFLIP
        elif op == "transpose":
            noisy = noisy.transpose(-2, -1).contiguous()
            clean = clean.transpose(-2, -1).contiguous()
            perm = PERM_TRANSPOSE
        else:
            raise ValueError(f"Unknown op '{op}'")

        if self.mode == "cell":
            # Cells move as units: no channel permutation, phase unchanged.
            return noisy, clean, pattern
        return _permute(noisy, perm), _permute(clean, perm), pattern_after(pattern, op)

    # -- call -------------------------------------------------------------
    def __call__(self, noisy: torch.Tensor, clean: torch.Tensor) -> Pair:
        if noisy.shape != clean.shape:
            raise ValueError(
                f"noisy {tuple(noisy.shape)} and clean {tuple(clean.shape)} "
                f"must have the same shape.")
        if noisy.dim() != 3 or noisy.shape[0] != 4:
            raise ValueError(
                f"D4Transform expects unbatched [4, h, w], got "
                f"{tuple(noisy.shape)}.")

        pattern = self.pattern
        ops: List[str] = []

        for op in ("hflip", "vflip", "transpose"):
            # The legacy short-circuit `allow_transpose and random() > 0.5`
            # consumes no draw when transposing is disabled, so the check
            # must come before the draw to keep the RNG streams aligned.
            if op == "transpose" and not self.allow_transpose:
                continue

            # `> 1 - p` rather than `<= p`: with p=0.5 this is literally the
            # legacy `random.random() > 0.5`, so a shared seed reproduces a
            # legacy run draw for draw.
            if self._draw() <= 1.0 - self.p:
                continue

            if op == "transpose" and noisy.shape[-1] != noisy.shape[-2]:
                # Transposing a non-square patch swaps its dimensions and
                # produces a ragged batch. Legacy did this; every other mode
                # declines, having already spent the draw.
                if self.mode != "legacy":
                    continue
            if not self._allows(op, pattern):
                continue
            noisy, clean, pattern = self._apply_op(noisy, clean, op, pattern)
            ops.append(op)

        self.last_pattern = pattern
        self.last_ops = tuple(ops)
        return noisy, clean

    def __repr__(self) -> str:
        return (f"D4Transform(pattern='{self.pattern}', mode='{self.mode}', "
                f"allow_transpose={self.allow_transpose}, "
                f"restrict_to_pattern={self.restrict_to_pattern}, p={self.p})")


class RandomCropPacked:
    """
    Random ``size x size`` crop in *packed* coordinates.

    Any (top, left) is valid: one packed pixel is one whole CFA cell, so
    cropping on the packed grid can never change the phase. That is also
    why the dataset can crop before synthesising noise.
    """

    def __init__(self, size: int, rng: Optional[random.Random] = None):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = int(size)
        self._rng = rng

    def __call__(self, noisy: torch.Tensor, clean: torch.Tensor) -> Pair:
        _, h, w = noisy.shape
        if h < self.size or w < self.size:
            raise ValueError(
                f"Cannot take a {self.size}x{self.size} crop from a "
                f"{h}x{w} packed tensor.")
        rnd = self._rng if self._rng is not None else random
        top = rnd.randint(0, h - self.size)
        left = rnd.randint(0, w - self.size)
        sl = (slice(None), slice(top, top + self.size),
              slice(left, left + self.size))
        return noisy[sl], clean[sl]

    def __repr__(self) -> str:
        return f"RandomCropPacked(size={self.size})"


class CenterCropPacked:
    """Deterministic centre crop in packed coordinates — for evaluation."""

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = int(size)

    def __call__(self, noisy: torch.Tensor, clean: torch.Tensor) -> Pair:
        _, h, w = noisy.shape
        if h < self.size or w < self.size:
            raise ValueError(
                f"Cannot take a {self.size}x{self.size} crop from a "
                f"{h}x{w} packed tensor.")
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        sl = (slice(None), slice(top, top + self.size),
              slice(left, left + self.size))
        return noisy[sl], clean[sl]

    def __repr__(self) -> str:
        return f"CenterCropPacked(size={self.size})"


def make_d4_transform_compat(allow_transpose: bool) -> PairTransform:
    """
    A drop-in replacement for ``train_A100_MoE_two_phase.make_d4_transform``.

    Same draw order, same permutations, same results for the same RNG
    state — including the CFA phase change it does not track. Use
    ``D4Transform(mode="cell")`` for new work.
    """
    return D4Transform("BGGR", allow_transpose=allow_transpose, mode="legacy")
