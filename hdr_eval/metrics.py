"""
hdr_eval/metrics.py — image quality metrics
============================================

Everything here takes ``[B, C, H, W]`` float tensors in [0, 1] and returns
either a per-image ``[B]`` tensor (the ``*_per_image`` functions) or a
float (the convenience wrappers). Per-image is the primitive because an
average of averages over differently sized images is not the average, and
because the runner writes one CSV row per image.

What is here and why
────────────────────
``psnr``        The headline number, in linear light or through a tone curve.
``ssim``        Structural similarity, bit-identical to the implementation
                in test_dual_MoE_two_phase.py so historical numbers stay
                comparable.
``ms_ssim``     Multi-scale SSIM — much better correlated with perceived
                quality on large images, where single-scale SSIM is
                dominated by fine texture.
``LPIPSMetric`` Learned perceptual distance, the same VGG net the training
                loss uses. Lazily constructed, tiled so a full-resolution
                frame does not have to fit in VRAM in one piece.
``delta_e_2000``CIEDE2000 colour difference. This is the metric that
                actually catches demosaicing failure: false colour on
                edges costs almost nothing in PSNR and is glaring in dE00.
``MetricSuite`` A configured bundle: given (pred, gt) it returns the flat
                dict of every enabled metric, in linear and tone-mapped
                form, which is what the runner writes out.

None of these functions move data between devices; feed them tensors that
already live where the work should happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .tonemap import MU_DEFAULT, mu_law

__all__ = [
    "mse_per_image",
    "mae_per_image",
    "psnr_per_image",
    "psnr",
    "ssim_map",
    "ssim_per_image",
    "ssim",
    "ms_ssim_per_image",
    "ms_ssim",
    "LPIPSMetric",
    "rgb_to_lab",
    "delta_e_76",
    "delta_e_2000",
    "MetricConfig",
    "MetricSuite",
]

#: Perfect-match PSNR is reported as this rather than infinity, so that a
#: column of results can be averaged.
PSNR_CEILING_DB = 100.0

#: The scale weights from Wang et al.'s MS-SSIM paper.
MS_SSIM_WEIGHTS: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _check_pair(pred: torch.Tensor, gt: torch.Tensor) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} and gt {tuple(gt.shape)} must have "
            f"the same shape.")
    if pred.dim() != 4:
        raise ValueError(
            f"Expected [B, C, H, W] tensors, got {tuple(pred.shape)}.")


# ─────────────────────────────────────────────────────────────────────────────
# Pixel metrics
# ─────────────────────────────────────────────────────────────────────────────

def mse_per_image(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean squared error per image. [B]."""
    _check_pair(pred, gt)
    return ((pred - gt) ** 2).flatten(1).mean(dim=1)


def mae_per_image(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean absolute error per image. [B]."""
    _check_pair(pred, gt)
    return (pred - gt).abs().flatten(1).mean(dim=1)


def psnr_per_image(pred: torch.Tensor, gt: torch.Tensor,
                   data_range: float = 1.0,
                   ceiling_db: float = PSNR_CEILING_DB) -> torch.Tensor:
    """Peak signal-to-noise ratio per image, in dB. [B]."""
    mse = mse_per_image(pred, gt)
    out = 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-20))
    return torch.clamp(out, max=ceiling_db)


def psnr(pred: torch.Tensor, gt: torch.Tensor,
         data_range: float = 1.0) -> float:
    """Batch-mean PSNR in dB. Matches the test script's ``psnr``."""
    return float(psnr_per_image(pred, gt, data_range).mean())


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float, channels: int,
                     dtype: torch.dtype, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, window_size, window_size)


def ssim_map(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11,
             data_range: float = 1.0, sigma: float = 1.5,
             padding: bool = True) -> torch.Tensor:
    """
    The per-pixel SSIM map, [B, C, H, W].

    ``padding=True`` reproduces the project's existing implementation
    (zero padding, so the border is measured against a dark surround);
    ``padding=False`` is the textbook 'valid' version, which is slightly
    higher because it drops those border pixels.
    """
    _check_pair(pred, gt)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels,
                              pred.dtype, pred.device)
    pad = window_size // 2 if padding else 0

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(gt, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1 = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu1_sq
    s2 = F.conv2d(gt * gt, kernel, padding=pad, groups=channels) - mu2_sq
    s12 = F.conv2d(pred * gt, kernel, padding=pad, groups=channels) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * s12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)
    return num / den


def ssim_per_image(pred: torch.Tensor, gt: torch.Tensor,
                   window_size: int = 11, data_range: float = 1.0,
                   sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM per image. [B]."""
    return ssim_map(pred, gt, window_size, data_range, sigma).flatten(1).mean(dim=1)


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11,
         data_range: float = 1.0) -> float:
    """Batch-mean SSIM. Matches the test script's ``ssim`` exactly."""
    return float(ssim_map(pred, gt, window_size, data_range).mean())


def _ssim_stats(pred: torch.Tensor, gt: torch.Tensor, kernel: torch.Tensor,
                data_range: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(ssim, contrast-structure) maps for one MS-SSIM scale, 'valid' mode."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    channels = pred.shape[1]

    mu1 = F.conv2d(pred, kernel, groups=channels)
    mu2 = F.conv2d(gt, kernel, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1 = F.conv2d(pred * pred, kernel, groups=channels) - mu1_sq
    s2 = F.conv2d(gt * gt, kernel, groups=channels) - mu2_sq
    s12 = F.conv2d(pred * gt, kernel, groups=channels) - mu1_mu2

    cs = (2 * s12 + c2) / (s1 + s2 + c2)
    luminance = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    return luminance * cs, cs


def ms_ssim_per_image(pred: torch.Tensor, gt: torch.Tensor,
                      window_size: int = 11, data_range: float = 1.0,
                      sigma: float = 1.5,
                      weights: Sequence[float] = MS_SSIM_WEIGHTS,
                      ) -> torch.Tensor:
    """
    Multi-scale SSIM per image. [B].

    Each scale halves the resolution, so the input must survive
    ``len(weights) - 1`` halvings and still be larger than the window:
    with the default 5 scales and an 11-tap window that is 161 pixels a
    side. Smaller inputs raise rather than silently dropping scales,
    because a 3-scale MS-SSIM is not comparable to a 5-scale one.
    """
    _check_pair(pred, gt)
    levels = len(weights)
    if levels < 1:
        raise ValueError("weights must not be empty")

    smallest = min(pred.shape[-2], pred.shape[-1]) / (2 ** (levels - 1))
    if smallest <= window_size:
        need = (window_size + 1) * (2 ** (levels - 1))
        raise ValueError(
            f"MS-SSIM with {levels} scales and window {window_size} needs "
            f"inputs of at least {need}x{need}; got "
            f"{pred.shape[-2]}x{pred.shape[-1]}. Use fewer scales "
            f"(weights=...) or ssim_per_image for small crops.")

    kernel = _gaussian_kernel(window_size, sigma, pred.shape[1],
                              pred.dtype, pred.device)
    w = torch.tensor(list(weights), dtype=pred.dtype, device=pred.device)

    cs_per_scale: List[torch.Tensor] = []
    ssim_last: Optional[torch.Tensor] = None
    for i in range(levels):
        ssim_i, cs_i = _ssim_stats(pred, gt, kernel, data_range)
        cs_per_scale.append(cs_i.flatten(1).mean(dim=1).clamp(min=1e-8))
        ssim_last = ssim_i.flatten(1).mean(dim=1).clamp(min=1e-8)
        if i < levels - 1:
            pred = F.avg_pool2d(pred, kernel_size=2)
            gt = F.avg_pool2d(gt, kernel_size=2)

    stacked = torch.stack(cs_per_scale[:-1] + [ssim_last], dim=0)  # [L, B]
    return torch.prod(stacked ** w.view(-1, 1), dim=0)


def ms_ssim(pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> float:
    """Batch-mean MS-SSIM."""
    return float(ms_ssim_per_image(pred, gt, **kwargs).mean())


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────

class LPIPSMetric:
    """
    Learned perceptual distance, matching the training loss's VGG net.

    The net is built on first use and cached, because constructing it
    downloads weights and takes seconds — unacceptable per image, fine
    once per run.

    Full-resolution HDR frames are large enough that the VGG activations
    do not fit comfortably in memory, so by default the image is scored in
    ``tile`` sized pieces and the tile scores averaged. That is not
    identical to a whole-image LPIPS (the receptive field is truncated at
    tile borders) but it is stable, and consistent across models, which is
    what a comparison needs. Set ``tile=None`` to score whole images.

    Inputs are [0, 1]; the [-1, 1] conversion LPIPS expects happens here.
    """

    def __init__(self, net: str = "vgg", device=None,
                 tile: Optional[int] = 512, tone_curve: bool = True,
                 mu: float = MU_DEFAULT):
        self.net_name = net
        self.device = torch.device(device) if device is not None else None
        self.tile = tile
        self.tone_curve = bool(tone_curve)
        self.mu = float(mu)
        self._net = None

    @property
    def available(self) -> bool:
        """Whether the lpips package can be imported."""
        try:
            import lpips  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_net(self):
        if self._net is None:
            try:
                import lpips
            except ImportError as exc:               # pragma: no cover
                raise RuntimeError(
                    "The 'lpips' package is required for the LPIPS metric. "
                    "Install it, or disable the metric with "
                    "MetricConfig(lpips=False)."
                ) from exc
            net = lpips.LPIPS(net=self.net_name)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            if self.device is not None:
                net = net.to(self.device)
            self._net = net
        return self._net

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """[0,1] -> the [-1,1] the net wants, through the tone curve."""
        x = x.clamp(0.0, 1.0).float()
        if self.tone_curve:
            x = mu_law(x, self.mu)
        return x * 2.0 - 1.0

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Per-image perceptual distance (lower is better). [B]."""
        _check_pair(pred, gt)
        net = self._get_net()
        device = self.device or pred.device
        p = self._prepare(pred).to(device)
        g = self._prepare(gt).to(device)

        if self.tile is None:
            return net(p, g).flatten().cpu()

        b, _, h, w = p.shape
        step = int(self.tile)
        totals = torch.zeros(b, dtype=torch.float32)
        count = 0
        for top in range(0, max(h - step + 1, 1), step):
            for left in range(0, max(w - step + 1, 1), step):
                pt = p[:, :, top:top + step, left:left + step]
                gtile = g[:, :, top:top + step, left:left + step]
                if min(pt.shape[-2:]) < 32:
                    # Smaller than the net's first pooling stack; scoring it
                    # would be dominated by padding.
                    continue
                totals += net(pt, gtile).flatten().float().cpu()
                count += 1
        if count == 0:
            return net(p, g).flatten().float().cpu()
        return totals / count


# ─────────────────────────────────────────────────────────────────────────────
# Colour
# ─────────────────────────────────────────────────────────────────────────────

#: Linear sRGB (D65) -> CIE XYZ.
_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
#: D65 white point.
_WHITE_D65 = (0.95047, 1.00000, 1.08883)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Linear sRGB in [0, 1] -> CIE L*a*b*. [B, 3, H, W] in, same shape out.

    Input is taken to be *linear* light, which is what the model emits —
    passing display-encoded values here would shift every hue.
    """
    if rgb.shape[-3] != 3:
        raise ValueError(f"Expected 3 channels, got {tuple(rgb.shape)}")
    m = torch.tensor(_RGB_TO_XYZ, dtype=rgb.dtype, device=rgb.device)
    x = rgb.clamp(min=0.0)
    xyz = torch.einsum("ij,bjhw->bihw", m, x)

    white = torch.tensor(_WHITE_D65, dtype=rgb.dtype,
                         device=rgb.device).view(1, 3, 1, 1)
    t = xyz / white

    delta = 6.0 / 29.0
    f = torch.where(t > delta ** 3,
                    t.clamp(min=1e-12) ** (1.0 / 3.0),
                    t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([lightness, a, b], dim=1)


def delta_e_76(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean CIE76 colour difference per image. [B]."""
    _check_pair(pred, gt)
    d = rgb_to_lab(pred) - rgb_to_lab(gt)
    return torch.sqrt((d ** 2).sum(dim=1) + 1e-12).flatten(1).mean(dim=1)


def delta_e_2000(pred: torch.Tensor, gt: torch.Tensor,
                 kl: float = 1.0, kc: float = 1.0, kh: float = 1.0,
                 ) -> torch.Tensor:
    """
    Mean CIEDE2000 colour difference per image. [B]. Lower is better.

    CIEDE2000 is worth its complexity here: it downweights differences in
    saturated colours and near-neutral hue shifts roughly the way vision
    does, so it flags the false-colour fringes a demosaicer produces on
    edges without also flagging harmless luminance error.
    """
    _check_pair(pred, gt)
    lab1 = rgb_to_lab(gt)
    lab2 = rgb_to_lab(pred)
    l1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    l2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    eps = 1e-12
    c1 = torch.sqrt(a1 ** 2 + b1 ** 2 + eps)
    c2 = torch.sqrt(a2 ** 2 + b2 ** 2 + eps)
    c_bar = 0.5 * (c1 + c2)

    c7 = c_bar ** 7
    g = 0.5 * (1.0 - torch.sqrt(c7 / (c7 + 25.0 ** 7 + eps)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2

    c1p = torch.sqrt(a1p ** 2 + b1 ** 2 + eps)
    c2p = torch.sqrt(a2p ** 2 + b2 ** 2 + eps)

    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p

    dhp = h2p - h1p
    dhp = torch.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = torch.where(dhp < -180.0, dhp + 360.0, dhp)
    # Hue difference is undefined when either colour is neutral; zero is
    # the conventional choice and keeps the gradient finite.
    chroma_zero = (c1p * c2p) <= eps
    dhp = torch.where(chroma_zero, torch.zeros_like(dhp), dhp)
    dhp_term = 2.0 * torch.sqrt(c1p * c2p + eps) * torch.sin(
        torch.deg2rad(dhp) / 2.0)

    lp_bar = 0.5 * (l1 + l2)
    cp_bar = 0.5 * (c1p + c2p)

    h_sum = h1p + h2p
    h_diff = (h1p - h2p).abs()
    hp_bar = torch.where(
        h_diff <= 180.0, 0.5 * h_sum,
        torch.where(h_sum < 360.0, 0.5 * (h_sum + 360.0),
                    0.5 * (h_sum - 360.0)))
    hp_bar = torch.where(chroma_zero, h_sum, hp_bar)

    t = (1.0
         - 0.17 * torch.cos(torch.deg2rad(hp_bar - 30.0))
         + 0.24 * torch.cos(torch.deg2rad(2.0 * hp_bar))
         + 0.32 * torch.cos(torch.deg2rad(3.0 * hp_bar + 6.0))
         - 0.20 * torch.cos(torch.deg2rad(4.0 * hp_bar - 63.0)))

    d_theta = 30.0 * torch.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    cp7 = cp_bar ** 7
    rc = 2.0 * torch.sqrt(cp7 / (cp7 + 25.0 ** 7 + eps))
    lp_term = (lp_bar - 50.0) ** 2
    sl = 1.0 + (0.015 * lp_term) / torch.sqrt(20.0 + lp_term + eps)
    sc = 1.0 + 0.045 * cp_bar
    sh = 1.0 + 0.015 * cp_bar * t
    rt = -torch.sin(torch.deg2rad(2.0 * d_theta)) * rc

    term_l = dlp / (kl * sl)
    term_c = dcp / (kc * sc)
    term_h = dhp_term / (kh * sh)
    de = torch.sqrt(term_l ** 2 + term_c ** 2 + term_h ** 2
                    + rt * term_c * term_h + eps)
    return de.flatten(1).mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# The configured bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricConfig:
    """
    Which metrics to compute, and how.

    ``mu`` must match the value used in training (5000) for PSNR-mu to be
    comparable with the W&B curves; the runner records it in the output so
    a mismatch is at least visible after the fact.
    """

    mu: float = MU_DEFAULT
    linear: bool = True          # also report metrics in linear light
    tonemapped: bool = True      # report metrics through the mu-law curve
    ssim: bool = True
    ms_ssim: bool = False        # needs >= 161px inputs; off for crops
    lpips: bool = False          # needs the lpips package and a GPU to be quick
    delta_e: bool = True
    mae: bool = True
    lpips_net: str = "vgg"
    lpips_tile: Optional[int] = 512
    data_range: float = 1.0

    def enabled_names(self) -> List[str]:
        """The metric column names this config will produce, in order."""
        names: List[str] = []
        for domain, on in (("linear", self.linear), ("mu", self.tonemapped)):
            if not on:
                continue
            names.append(f"psnr_{domain}")
            if self.mae:
                names.append(f"mae_{domain}")
            if self.ssim:
                names.append(f"ssim_{domain}")
            if self.ms_ssim:
                names.append(f"msssim_{domain}")
        if self.lpips:
            names.append("lpips")
        if self.delta_e:
            names.append("delta_e2000")
        return names


class MetricSuite:
    """
    Compute a configured set of metrics for one (pred, gt) pair.

        suite = MetricSuite(MetricConfig(ms_ssim=True))
        row = suite(pred, gt)      # {"psnr_mu": 34.2, "ssim_mu": 0.94, ...}

    Values are per-batch means as Python floats, so the result drops
    straight into a CSV row. Use the module-level ``*_per_image``
    functions when the individual images are needed.
    """

    def __init__(self, config: Optional[MetricConfig] = None, device=None):
        self.config = config or MetricConfig()
        self._lpips: Optional[LPIPSMetric] = None
        self.device = device
        if self.config.lpips:
            self._lpips = LPIPSMetric(
                net=self.config.lpips_net, device=device,
                tile=self.config.lpips_tile, mu=self.config.mu)

    @property
    def names(self) -> List[str]:
        """Column names this suite produces."""
        return self.config.enabled_names()

    def _domain_metrics(self, pred: torch.Tensor, gt: torch.Tensor,
                        suffix: str) -> Dict[str, float]:
        cfg = self.config
        out: Dict[str, float] = {
            f"psnr_{suffix}": float(
                psnr_per_image(pred, gt, cfg.data_range).mean()),
        }
        if cfg.mae:
            out[f"mae_{suffix}"] = float(mae_per_image(pred, gt).mean())
        if cfg.ssim:
            out[f"ssim_{suffix}"] = float(
                ssim_per_image(pred, gt, data_range=cfg.data_range).mean())
        if cfg.ms_ssim:
            out[f"msssim_{suffix}"] = float(
                ms_ssim_per_image(pred, gt, data_range=cfg.data_range).mean())
        return out

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        _check_pair(pred, gt)
        cfg = self.config
        pred = pred.clamp(0.0, 1.0).float()
        gt = gt.clamp(0.0, 1.0).float()

        row: Dict[str, float] = {}
        if cfg.linear:
            row.update(self._domain_metrics(pred, gt, "linear"))
        if cfg.tonemapped:
            row.update(self._domain_metrics(
                mu_law(pred, cfg.mu), mu_law(gt, cfg.mu), "mu"))
        if self._lpips is not None:
            row["lpips"] = float(self._lpips(pred, gt).mean())
        if cfg.delta_e:
            row["delta_e2000"] = float(delta_e_2000(pred, gt).mean())
        return row
