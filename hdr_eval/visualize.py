"""
hdr_eval/visualize.py — figures, without a plotting stack
==========================================================

Everything here writes an image file using only torch and PIL, because
matplotlib is not installed in this project's environment and adding a
dependency for four figure types is a poor trade.

What it produces:

``save_comparison``  A labelled strip — noisy | predicted | reference —
    tone-mapped for display. The single most useful artefact of a
    benchmark run, and the one that catches the failures no metric names.
``save_error_heatmap``  Absolute error, colour-mapped. Where a model is
    wrong matters more than by how much on average: a uniform haze and a
    few blown edges can score the same PSNR.
``save_gate_map``  The MoE router's per-pixel expert weights, as a colour
    composite for up to three experts or a strip beyond that. If the gate
    has collapsed onto one expert this shows it instantly.
``save_crop_comparison``  Matched zoomed crops across models, which is how
    demosaicing artefacts are actually judged.

All of them tone-map first (mu-law, the project's mu=5000), because
linear HDR written straight to an 8-bit file is a black rectangle.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .tonemap import MU_DEFAULT, mu_law

__all__ = [
    "to_display",
    "save_image",
    "save_comparison",
    "save_error_heatmap",
    "save_gate_map",
    "save_crop_comparison",
    "colorize",
    "TURBO_ANCHORS",
]

#: Anchor colours of a turbo-like map: dark blue -> cyan -> green -> yellow
#: -> red. Perceptually monotone in lightness, unlike jet, and legible in
#: greyscale print.
TURBO_ANCHORS: Tuple[Tuple[float, float, float], ...] = (
    (0.19, 0.07, 0.23),
    (0.15, 0.44, 0.78),
    (0.15, 0.76, 0.60),
    (0.79, 0.85, 0.19),
    (0.86, 0.31, 0.13),
    (0.48, 0.01, 0.01),
)


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    """Accept [C, H, W] or [B, C, H, W]; return the first image batched."""
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x[:1]
    raise ValueError(f"Expected [C, H, W] or [B, C, H, W], got {tuple(x.shape)}")


def to_display(x: torch.Tensor, mu: float = MU_DEFAULT,
               scale: float = 1.0) -> torch.Tensor:
    """
    Linear HDR -> an 8-bit-ready [0, 1] image, tone-mapped and downscaled.

    `scale` below 1 shrinks the image: a full-resolution comparison strip
    of three sensor-resolution frames is ~100 MB of PNG, which nobody
    opens twice.
    """
    img = _as_batch(x).detach().float().clamp(0, 1)
    img = mu_law(img, mu).clamp(0, 1)
    if scale != 1.0:
        if not 0 < scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {scale}")
        img = F.interpolate(img, scale_factor=scale, mode="bilinear",
                            align_corners=False, antialias=True)
    return img


def save_image(path: str, image: torch.Tensor, quality: int = 92) -> str:
    """
    Write a [C, H, W] or [1, C, H, W] tensor in [0, 1] as PNG or JPEG.

    Returns the path, so callers can log it.
    """
    from PIL import Image

    img = _as_batch(image)[0].detach().float().clamp(0, 1).cpu()
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    if img.shape[0] not in (3, 4):
        raise ValueError(
            f"Cannot save an image with {img.shape[0]} channels.")
    arr = (img.permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pil = Image.fromarray(arr)
    if path.lower().endswith((".jpg", ".jpeg")):
        pil.save(path, format="JPEG", quality=quality)
    else:
        pil.save(path)
    return path


def _label_strip(width: int, labels: Sequence[str], panel_width: int,
                 height: int = 18) -> torch.Tensor:
    """A [3, height, width] caption bar naming each panel."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(labels):
        draw.text((i * panel_width + 6, 4), str(label), fill=(235, 235, 235))
    arr = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
    return arr.view(height, width, 3).permute(2, 0, 1).float() / 255.0


def save_comparison(path: str, panels: Mapping[str, torch.Tensor],
                    mu: float = MU_DEFAULT, scale: float = 0.5,
                    separator: int = 4, label: bool = True,
                    quality: int = 92) -> str:
    """
    Write a labelled left-to-right strip of tone-mapped panels.

        save_comparison("out.jpg", {"noisy": x, "ours": p, "gt": g})

    Panels must share spatial dimensions; that is the point of the figure.
    """
    if not panels:
        raise ValueError("Need at least one panel.")
    images = [to_display(v, mu, scale) for v in panels.values()]
    shapes = {tuple(im.shape[-2:]) for im in images}
    if len(shapes) != 1:
        raise ValueError(
            f"Comparison panels must have matching sizes, got {sorted(shapes)}")

    h, w = images[0].shape[-2], images[0].shape[-1]
    gap = torch.ones(1, 3, h, separator) if separator > 0 else None
    parts = []
    for i, im in enumerate(images):
        if i and gap is not None:
            parts.append(gap)
        parts.append(im)
    strip = torch.cat(parts, dim=-1)[0]

    if label:
        panel_w = w + (separator if separator > 0 else 0)
        caption = _label_strip(strip.shape[-1], list(panels.keys()), panel_w)
        strip = torch.cat([caption, strip], dim=-2)
    return save_image(path, strip, quality=quality)


def colorize(values: torch.Tensor, vmin: Optional[float] = None,
             vmax: Optional[float] = None) -> torch.Tensor:
    """
    Map a [1, 1, H, W] or [H, W] scalar field to RGB via the turbo anchors.

    Values are normalised to [vmin, vmax] (defaults: the field's own min
    and max) and linearly interpolated between the anchor colours.
    """
    v = values.detach().float()
    while v.dim() > 2:
        v = v[0]
    lo = float(v.min()) if vmin is None else float(vmin)
    hi = float(v.max()) if vmax is None else float(vmax)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    t = ((v - lo) / (hi - lo)).clamp(0, 1)

    anchors = torch.tensor(TURBO_ANCHORS, dtype=torch.float32, device=v.device)
    n = anchors.shape[0] - 1
    pos = t * n
    idx = pos.floor().clamp(0, n - 1).long()
    frac = (pos - idx).unsqueeze(-1)
    c0 = anchors[idx]
    c1 = anchors[idx + 1]
    rgb = c0 * (1 - frac) + c1 * frac                 # [H, W, 3]
    return rgb.permute(2, 0, 1)


def save_error_heatmap(path: str, pred: torch.Tensor, gt: torch.Tensor,
                       mu: float = MU_DEFAULT, scale: float = 0.5,
                       vmax: Optional[float] = None) -> str:
    """
    Colour-mapped absolute error, measured through the tone curve.

    Measuring the error *after* tone mapping is deliberate: a linear-light
    error map is entirely highlights, which is not where the eye or the
    loss is looking.
    """
    p = to_display(pred, mu, scale)
    g = to_display(gt, mu, scale)
    if p.shape != g.shape:
        raise ValueError(
            f"pred {tuple(p.shape)} and gt {tuple(g.shape)} must match.")
    err = (p - g).abs().mean(dim=1, keepdim=True)
    return save_image(path, colorize(err, 0.0, vmax))


def save_gate_map(path: str, gates: torch.Tensor, scale: float = 0.5,
                  quality: int = 92) -> str:
    """
    Visualise per-pixel routing weights, [B, K, H, W].

    Up to three experts are composited into one RGB image (expert k drives
    channel k), which makes the spatial structure of the routing obvious
    at a glance. Beyond three, each expert gets its own greyscale panel.
    """
    if gates.dim() != 4:
        raise ValueError(f"Expected [B, K, H, W] gates, got {tuple(gates.shape)}")
    g = gates[:1].detach().float().clamp(0, 1)
    if scale != 1.0:
        g = F.interpolate(g, scale_factor=scale, mode="bilinear",
                          align_corners=False)
    k = g.shape[1]
    if k <= 3:
        rgb = torch.zeros(1, 3, g.shape[-2], g.shape[-1])
        rgb[:, :k] = g
        return save_image(path, rgb, quality=quality)

    panels = {f"expert{i}": g[:, i:i + 1].expand(-1, 3, -1, -1) for i in range(k)}
    # Gates are already in [0, 1]; a near-zero mu keeps to_display's
    # curve effectively linear so the weights are read literally.
    return save_comparison(path, panels, mu=1e-6, scale=1.0, quality=quality)


def save_crop_comparison(path: str, panels: Mapping[str, torch.Tensor],
                         box: Tuple[int, int, int, int], zoom: int = 4,
                         mu: float = MU_DEFAULT, quality: int = 92) -> str:
    """
    Matched zoomed crops, nearest-neighbour upscaled.

    ``box`` is ``(top, left, height, width)`` in the images' own pixels.
    Nearest-neighbour is the right interpolation here: the artefacts being
    inspected are per-pixel, and a smooth upscale would hide them.
    """
    top, left, height, width = box
    if height <= 0 or width <= 0:
        raise ValueError(f"box height and width must be positive, got {box}")
    if zoom < 1:
        raise ValueError(f"zoom must be >= 1, got {zoom}")

    crops: Dict[str, torch.Tensor] = {}
    for name, img in panels.items():
        t = _as_batch(img)
        if top + height > t.shape[-2] or left + width > t.shape[-1]:
            raise ValueError(
                f"Crop {box} does not fit in panel '{name}' of size "
                f"{tuple(t.shape[-2:])}.")
        crop = t[..., top:top + height, left:left + width]
        crops[name] = F.interpolate(crop, scale_factor=zoom, mode="nearest")
    return save_comparison(path, crops, mu=mu, scale=1.0, quality=quality)
