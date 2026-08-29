"""
hdr_eval/inference.py — running a model over a whole frame
===========================================================

The model takes packed CFA at [B, 4, h, w] with h and w divisible by 8
(three PixelUnshuffle(2) stages) and returns RGB at sensor resolution,
[B, 3, 2h, 2w]. Getting a full frame through it means one of two things:

``full``   Pad up to a multiple of 8, run once, crop back. No seams, but
           memory grows with the frame and every distinct frame size
           triggers a fresh torch.compile.

``tiled``  Overlapping tiles, blended. Bounded memory, one compiled shape,
           and it works on frames that do not fit at all.

The tiled path here differs from the one in test_dual_MoE_two_phase.py in
two ways that matter:

* **The blend window tapers.** The original accumulates with a window of
  all ones, so overlapping tiles are averaged with equal weight right up
  to the tile edge, and any discontinuity between neighbouring tiles lands
  as a visible seam at the overlap boundary. Here the window ramps across
  the overlap region (a Tukey taper), so the transition is smooth. The
  taper is switched off on sides that touch the image boundary, where
  there is no neighbour to blend with and a taper would leave the border
  weighted by almost nothing.

* **Every pixel is covered.** The original steps
  ``range(0, Hp - patch + 1, stride)``, which silently drops the last
  partial tile when the padded size is not an exact multiple of the
  stride. The padding here is computed from the tile count, so the last
  tile always lands flush with the far edge.

Autocast is chosen by device: bfloat16 on any GPU (CUDA on Polaris, XPU on
Aurora), off on CPU where it is slower than fp32 and unnecessary, so the
same code runs on a login node.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "ModelOutput",
    "InferenceConfig",
    "autocast_context",
    "call_model",
    "infer_full",
    "infer_tiled",
    "infer",
    "blend_window",
]


@dataclass
class ModelOutput:
    """
    What a forward pass produced.

    ``experts`` and ``gates`` are None for models with no routing (the
    classical baselines), so consumers must check rather than assume.
    """

    blended: torch.Tensor                      # [B, 3, H, W]
    experts: Optional[torch.Tensor] = None     # [B, K, 3, H, W]
    gates: Optional[torch.Tensor] = None       # [B, K, H, W]

    @property
    def num_experts(self) -> int:
        """Number of expert branches, 0 when the model has none."""
        return 0 if self.experts is None else int(self.experts.shape[1])

    def detach_cpu(self) -> "ModelOutput":
        """A copy on the CPU, detached — for saving without pinning VRAM."""
        def move(t):
            return None if t is None else t.detach().float().cpu()
        return ModelOutput(move(self.blended), move(self.experts),
                           move(self.gates))


@dataclass
class InferenceConfig:
    """
    How to push a frame through a model.

    Attributes
    ──────────
    mode
        ``"full"`` or ``"tiled"``.
    tile
        Tile size in *packed* pixels. Must be divisible by 8.
    overlap
        Overlap between neighbouring tiles, in packed pixels. Also the
        width of the blend taper. A quarter of the tile is a good default;
        zero disables blending entirely and will show seams.
    amp_dtype
        Autocast dtype on CUDA. bfloat16 matches training. None disables
        autocast.
    multiple_of
        Input alignment the model requires; 8 for this architecture.
    """

    mode: str = "full"
    tile: int = 512
    overlap: int = 128
    amp_dtype: Optional[torch.dtype] = torch.bfloat16
    multiple_of: int = 8

    def __post_init__(self):
        if self.mode not in ("full", "tiled"):
            raise ValueError(f"mode must be 'full' or 'tiled', got '{self.mode}'")
        if self.tile <= 0:
            raise ValueError(f"tile must be positive, got {self.tile}")
        if self.tile % self.multiple_of:
            raise ValueError(
                f"tile ({self.tile}) must be divisible by "
                f"multiple_of ({self.multiple_of}).")
        if not 0 <= self.overlap < self.tile:
            raise ValueError(
                f"overlap must satisfy 0 <= overlap < tile, got "
                f"{self.overlap} with tile={self.tile}.")


# Accelerator backends that have a working autocast implementation. CPU
# autocast exists but is slower than plain fp32 for this model, so it is
# deliberately excluded rather than merely unsupported.
_AUTOCAST_DEVICES = ("cuda", "xpu")


def autocast_context(device: torch.device,
                     dtype: Optional[torch.dtype] = torch.bfloat16):
    """
    Autocast on a GPU, a no-op elsewhere.

    ``device_type`` has to match the backend: ``"cuda"`` on Polaris and
    ``"xpu"`` on Aurora. The test script hardcodes ``'cuda'``, which throws
    on both a CPU-only host and an Intel GPU; passing the device's own type
    makes one code path work on all three.
    """
    if dtype is None or device.type not in _AUTOCAST_DEVICES:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def call_model(model: Callable, x: torch.Tensor,
               snr_map: torch.Tensor) -> ModelOutput:
    """
    Invoke a model and normalise its return into a :class:`ModelOutput`.

    Accepts the project's ``(blended, experts, gates)`` triple, a bare
    tensor (the classical baselines), or a ModelOutput straight through.
    """
    out = model(x, snr_map)
    if isinstance(out, ModelOutput):
        return out
    if torch.is_tensor(out):
        return ModelOutput(out)
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            return ModelOutput(out[0], out[1], out[2])
        if len(out) == 1:
            return ModelOutput(out[0])
        raise ValueError(
            f"Model returned a {len(out)}-tuple; expected 1 or 3 elements "
            f"(blended[, experts, gates]).")
    raise TypeError(
        f"Model returned {type(out).__name__}; expected a tensor, a "
        f"(blended, experts, gates) tuple, or a ModelOutput.")


def _pad_to_multiple(x: torch.Tensor, multiple: int
                     ) -> Tuple[torch.Tensor, int, int]:
    """Reflect-pad the right/bottom edges up to a multiple. Returns padding."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        # Reflect padding needs the pad to be smaller than the dimension;
        # replicate is the fallback for very small inputs.
        mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, pad_h, pad_w


def blend_window(size: int, taper: int, *, top: bool = False,
                 bottom: bool = False, left: bool = False,
                 right: bool = False, device=None,
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    A ``[1, 1, size, size]`` Tukey-style blend window.

    The window ramps from 0 to 1 across `taper` pixels on each side, and
    is flat 1 in the middle. Sides flagged as touching the image boundary
    keep their full weight — there is nothing beyond the edge to blend
    with, and tapering there would leave the border pixels reconstructed
    almost entirely from the normalisation epsilon.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if taper < 0 or 2 * taper > size:
        raise ValueError(
            f"taper must satisfy 0 <= 2*taper <= size, got taper={taper}, "
            f"size={size}.")

    def axis(edge_lo: bool, edge_hi: bool) -> torch.Tensor:
        w = torch.ones(size, device=device, dtype=dtype)
        if taper == 0:
            return w
        # Raised-cosine ramp; the half-sample offset keeps it strictly
        # positive so no pixel is ever weighted at exactly zero.
        ramp = 0.5 * (1.0 - torch.cos(
            math.pi * (torch.arange(taper, device=device, dtype=dtype) + 0.5)
            / taper))
        if not edge_lo:
            w[:taper] = ramp
        if not edge_hi:
            w[size - taper:] = ramp.flip(0)
        return w

    wy = axis(top, bottom).view(size, 1)
    wx = axis(left, right).view(1, size)
    return (wy * wx).view(1, 1, size, size)


def _tile_starts(length: int, tile: int, stride: int) -> Tuple[list, int]:
    """
    Start offsets covering `length`, and the padded length they need.

    Returns starts that always include a final tile flush with the far
    edge, so no strip of the image is left uncovered.
    """
    if length <= tile:
        return [0], tile
    count = math.ceil((length - tile) / stride) + 1
    padded = (count - 1) * stride + tile
    return [i * stride for i in range(count)], padded


@torch.no_grad()
def infer_full(model: Callable, packed: torch.Tensor,
               snr_fn: Callable[[torch.Tensor], torch.Tensor],
               config: Optional[InferenceConfig] = None) -> ModelOutput:
    """
    One forward pass over the whole frame.

    ``packed`` is [B, 4, h, w] and the result is at sensor resolution,
    [B, 3, 2h, 2w], cropped back to exactly 2x the input dimensions.
    """
    config = config or InferenceConfig(mode="full")
    device = packed.device
    x, pad_h, pad_w = _pad_to_multiple(packed, config.multiple_of)
    snr = snr_fn(x)

    with autocast_context(device, config.amp_dtype):
        out = call_model(model, x, snr)

    h_out = (x.shape[-2] - pad_h) * 2
    w_out = (x.shape[-1] - pad_w) * 2
    return ModelOutput(
        out.blended[..., :h_out, :w_out].float(),
        None if out.experts is None else out.experts[..., :h_out, :w_out].float(),
        None if out.gates is None else out.gates[..., :h_out, :w_out].float(),
    )


@torch.no_grad()
def infer_tiled(model: Callable, packed: torch.Tensor,
                snr_fn: Callable[[torch.Tensor], torch.Tensor],
                config: Optional[InferenceConfig] = None) -> ModelOutput:
    """
    Overlapping-tile inference with a tapered blend.

    The SNR map is computed once on the padded full frame and cropped per
    tile, not recomputed per tile: it is normalised by a per-image
    maximum, so computing it tile-wise would give neighbouring tiles
    different normalisers and make the routing signal discontinuous
    exactly at the seams.
    """
    config = config or InferenceConfig(mode="tiled")
    device = packed.device
    tile = config.tile
    stride = tile - config.overlap

    b, _, h, w = packed.shape
    if b != 1:
        raise ValueError(
            f"Tiled inference handles one image at a time, got batch {b}.")

    ys, padded_h = _tile_starts(h, tile, stride)
    xs, padded_w = _tile_starts(w, tile, stride)
    pad_h, pad_w = padded_h - h, padded_w - w
    if pad_h or pad_w:
        mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
        x_pad = F.pad(packed, (0, pad_w, 0, pad_h), mode=mode)
    else:
        x_pad = packed

    snr_full = snr_fn(x_pad)

    # Probe one tile to learn the expert/gate shapes without assuming them.
    probe_tile = x_pad[:, :, ys[0]:ys[0] + tile, xs[0]:xs[0] + tile]
    probe_snr = snr_full[:, :, ys[0]:ys[0] + tile, xs[0]:xs[0] + tile]
    with autocast_context(device, config.amp_dtype):
        probe = call_model(model, probe_tile, probe_snr)
    num_experts = probe.num_experts
    channels = probe.blended.shape[1]

    out_h, out_w = padded_h * 2, padded_w * 2
    acc = torch.zeros((1, channels, out_h, out_w), device=device,
                      dtype=torch.float32)
    weight = torch.zeros((1, 1, out_h, out_w), device=device,
                         dtype=torch.float32)
    acc_experts = (torch.zeros((1, num_experts, channels, out_h, out_w),
                               device=device, dtype=torch.float32)
                   if probe.experts is not None else None)
    acc_gates = (torch.zeros((1, num_experts, out_h, out_w), device=device,
                             dtype=torch.float32)
                 if probe.gates is not None else None)

    out_tile = tile * 2
    # The window lives in sensor pixels, so the overlap doubles with it.
    taper = min(config.overlap * 2, out_tile // 2)

    for yi in ys:
        for xi in xs:
            patch = x_pad[:, :, yi:yi + tile, xi:xi + tile]
            snr_crop = snr_full[:, :, yi:yi + tile, xi:xi + tile]
            with autocast_context(device, config.amp_dtype):
                out = call_model(model, patch, snr_crop)

            win = blend_window(
                out_tile, taper,
                top=(yi == ys[0]), bottom=(yi == ys[-1]),
                left=(xi == xs[0]), right=(xi == xs[-1]),
                device=device, dtype=torch.float32)

            ys_out, xs_out = yi * 2, xi * 2
            sl = (slice(None), slice(None),
                  slice(ys_out, ys_out + out_tile),
                  slice(xs_out, xs_out + out_tile))
            acc[sl] += out.blended.float() * win
            weight[:, :, ys_out:ys_out + out_tile,
                   xs_out:xs_out + out_tile] += win
            if acc_experts is not None and out.experts is not None:
                acc_experts[:, :, :, ys_out:ys_out + out_tile,
                            xs_out:xs_out + out_tile] += (
                    out.experts.float() * win.unsqueeze(1))
            if acc_gates is not None and out.gates is not None:
                acc_gates[:, :, ys_out:ys_out + out_tile,
                          xs_out:xs_out + out_tile] += out.gates.float() * win

    denom = weight.clamp(min=1e-8)
    h_out, w_out = h * 2, w * 2
    blended = (acc / denom)[..., :h_out, :w_out]
    experts = (None if acc_experts is None
               else (acc_experts / denom.unsqueeze(1))[..., :h_out, :w_out])
    gates = (None if acc_gates is None
             else (acc_gates / denom)[..., :h_out, :w_out])
    return ModelOutput(blended, experts, gates)


def infer(model: Callable, packed: torch.Tensor,
          snr_fn: Callable[[torch.Tensor], torch.Tensor],
          config: Optional[InferenceConfig] = None) -> ModelOutput:
    """Dispatch to :func:`infer_full` or :func:`infer_tiled` by config."""
    config = config or InferenceConfig()
    if config.mode == "tiled":
        return infer_tiled(model, packed, snr_fn, config)
    return infer_full(model, packed, snr_fn, config)
