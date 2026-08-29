"""
hdr_eval/regions.py — where the error actually is
==================================================

A single PSNR over a whole HDR frame hides the thing you most want to
know. A denoiser can gain 2 dB overall while getting *worse* in the deep
shadows, because shadows are a small fraction of the squared error budget
and the highlights are most of it. Likewise a joint demosaicer's failures
concentrate on edges, which are a few percent of the pixels.

So this module builds masks — by luminance band, by local SNR, by
saturation, by edge strength — and applies any metric within them. The
runner then reports "PSNR-mu overall, and in the darkest decile, and on
edges", which is what a routing model with per-pixel experts needs in
order to be judged at all.

Masks are boolean ``[B, 1, H, W]`` tensors, always derived from the
*ground truth* so that two models are compared on exactly the same
pixels.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "luminance",
    "luminance_bands",
    "snr_bands",
    "saturation_mask",
    "edge_mask",
    "quantile_bands",
    "masked_mse",
    "masked_psnr",
    "masked_mae",
    "stratify",
    "DEFAULT_LUMINANCE_EDGES",
    "DEFAULT_SNR_EDGES",
]

#: Luminance band edges in [0, 1] linear: deep shadow, shadow, mid, highlight.
DEFAULT_LUMINANCE_EDGES: Tuple[float, ...] = (0.0, 0.02, 0.1, 0.5, 1.01)
#: Normalised-SNR band edges matching the router's own 0.5 threshold.
DEFAULT_SNR_EDGES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.01)

#: Rec.709 luma weights — the model outputs linear RGB, so these are the
#: right coefficients for a luminance proxy.
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] linear RGB -> [B, 1, H, W] Rec.709 luminance."""
    if rgb.shape[-3] != 3:
        raise ValueError(
            f"Expected 3 colour channels, got {tuple(rgb.shape)}")
    w = torch.tensor(_LUMA_WEIGHTS, dtype=rgb.dtype, device=rgb.device)
    return (rgb * w.view(-1, 1, 1)).sum(dim=-3, keepdim=True)


def _band_masks(values: torch.Tensor, edges: Sequence[float],
                prefix: str) -> Dict[str, torch.Tensor]:
    """Half-open [lo, hi) masks over `values`, named by their bounds."""
    if len(edges) < 2:
        raise ValueError(f"Need at least two edges, got {list(edges)}")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"Edges must be strictly increasing: {list(edges)}")
    out: Dict[str, torch.Tensor] = {}
    for lo, hi in zip(edges, edges[1:]):
        name = f"{prefix}[{lo:g},{hi:g})"
        out[name] = (values >= lo) & (values < hi)
    return out


def luminance_bands(gt_rgb: torch.Tensor,
                    edges: Sequence[float] = DEFAULT_LUMINANCE_EDGES,
                    ) -> Dict[str, torch.Tensor]:
    """
    Masks splitting the frame by ground-truth luminance.

    The default edges put the deep shadows (below 2% of full scale) in
    their own band, because that is where a low-light denoiser lives and
    where a whole-frame PSNR is least sensitive.
    """
    return _band_masks(luminance(gt_rgb), edges, "lum")


def snr_bands(snr_map: torch.Tensor,
              edges: Sequence[float] = DEFAULT_SNR_EDGES,
              ) -> Dict[str, torch.Tensor]:
    """
    Masks splitting the frame by the router's own normalised SNR map.

    Reported per band, these say whether the MoE gate is routing the
    pixels it should: expert usage and per-band PSNR should move together.
    """
    if snr_map.shape[-3] != 1:
        raise ValueError(
            f"Expected a single-channel SNR map, got {tuple(snr_map.shape)}")
    return _band_masks(snr_map, edges, "snr")


def quantile_bands(gt_rgb: torch.Tensor, num_bands: int = 4,
                   ) -> Dict[str, torch.Tensor]:
    """
    Equal-population luminance bands, computed per image.

    Fixed edges can leave a band empty on a frame with an unusual
    histogram; quantile bands never do, at the cost of not being
    comparable across images.
    """
    if num_bands < 2:
        raise ValueError(f"num_bands must be >= 2, got {num_bands}")
    lum = luminance(gt_rgb)
    b = lum.shape[0]
    qs = torch.linspace(0, 1, num_bands + 1, device=lum.device,
                        dtype=lum.dtype)[1:-1]
    out: Dict[str, torch.Tensor] = {}
    flat = lum.reshape(b, -1)
    cuts = torch.quantile(flat.float(), qs.float(), dim=1)      # [n-1, B]
    if cuts.dim() == 1:
        cuts = cuts.unsqueeze(0)
    lo = torch.full((b,), float("-inf"), device=lum.device)
    for i in range(num_bands):
        hi = (cuts[i] if i < num_bands - 1
              else torch.full((b,), float("inf"), device=lum.device))
        mask = (lum >= lo.view(-1, 1, 1, 1)) & (lum < hi.view(-1, 1, 1, 1))
        out[f"q{i}"] = mask
        lo = hi
    return out


def saturation_mask(gt_rgb: torch.Tensor, threshold: float = 0.99,
                    ) -> torch.Tensor:
    """
    Pixels where any channel is at or above `threshold` — i.e. clipped.

    Worth reporting separately: a model cannot recover detail that the
    sensor never captured, so clipped pixels flatter or punish a score
    depending only on how many of them a scene has.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    return (gt_rgb >= threshold).any(dim=-3, keepdim=True)


def edge_mask(gt_rgb: torch.Tensor, quantile: float = 0.9) -> torch.Tensor:
    """
    The strongest-gradient pixels, by per-image quantile of Sobel energy.

    Demosaicing artefacts — zippering, false colour — live on edges, and
    a flat-region-dominated average will not show them.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    lum = luminance(gt_rgb)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      dtype=lum.dtype, device=lum.device).view(1, 1, 3, 3)
    ky = kx.transpose(-2, -1)
    pad = F.pad(lum, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(pad, kx)
    gy = F.conv2d(pad, ky)
    energy = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    b = energy.shape[0]
    thresh = torch.quantile(energy.reshape(b, -1).float(), quantile, dim=1)
    return energy >= thresh.view(-1, 1, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Masked metrics
# ─────────────────────────────────────────────────────────────────────────────

def _expand_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B, 1, H, W] mask across `like`'s channels."""
    if mask.dim() != like.dim():
        raise ValueError(
            f"mask {tuple(mask.shape)} and tensor {tuple(like.shape)} must "
            f"have the same rank.")
    return mask.expand_as(like)


def masked_mse(pred: torch.Tensor, gt: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Per-image MSE over the masked pixels. [B] tensor; NaN where a mask is
    empty for that image, which is the honest answer — a metric over zero
    pixels is undefined, and silently reporting 0 would look like success.
    """
    err = (pred - gt) ** 2
    if mask is None:
        return err.flatten(1).mean(dim=1)
    m = _expand_mask(mask, err).to(err.dtype)
    total = m.flatten(1).sum(dim=1)
    summed = (err * m).flatten(1).sum(dim=1)
    out = summed / total.clamp(min=1e-12)
    return torch.where(total > 0, out, torch.full_like(out, float("nan")))


def masked_mae(pred: torch.Tensor, gt: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-image mean absolute error over the masked pixels. [B]."""
    err = (pred - gt).abs()
    if mask is None:
        return err.flatten(1).mean(dim=1)
    m = _expand_mask(mask, err).to(err.dtype)
    total = m.flatten(1).sum(dim=1)
    summed = (err * m).flatten(1).sum(dim=1)
    out = summed / total.clamp(min=1e-12)
    return torch.where(total > 0, out, torch.full_like(out, float("nan")))


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                data_range: float = 1.0, max_db: float = 100.0) -> torch.Tensor:
    """
    Per-image PSNR over the masked pixels. [B].

    A perfect match is reported as `max_db` rather than infinity so that
    averaging a column of results stays finite.
    """
    mse = masked_mse(pred, gt, mask)
    psnr = 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-20))
    return torch.clamp(psnr, max=max_db)


def stratify(pred: torch.Tensor, gt: torch.Tensor,
             masks: Dict[str, torch.Tensor],
             fn: Callable[..., torch.Tensor] = masked_psnr,
             include_coverage: bool = True) -> Dict[str, float]:
    """
    Apply `fn` inside every mask and return a flat ``{name: value}`` dict.

    With ``include_coverage`` each band also reports the fraction of
    pixels it covers, as ``<name>.coverage`` — without it a spectacular
    score on 12 pixels is indistinguishable from one on half the frame.
    """
    out: Dict[str, float] = {}
    total_px = float(gt[:, :1].numel())
    for name, mask in masks.items():
        value = fn(pred, gt, mask)
        out[name] = float(value.nanmean())
        if include_coverage:
            out[f"{name}.coverage"] = float(mask.sum()) / max(total_px, 1.0)
    return out
