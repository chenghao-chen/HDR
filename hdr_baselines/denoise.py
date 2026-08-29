"""
hdr_baselines/denoise.py — classical denoisers
===============================================

Six references, all pure torch, all operating on ``[B, C, H, W]`` in
[0, 1]. They are used two ways: on packed CFA (C=4, each channel a
separate sub-lattice — the correct domain for denoising before
demosaicing) and on demosaiced RGB (C=3).

    ``GaussianDenoise``   Isotropic blur. The floor: it removes noise and
        detail in equal measure, which is exactly the trade every other
        method is trying to escape.
    ``MedianDenoise``     3x3 median. Cheap, keeps edges, destroys texture,
        and is the only one here that handles hot pixels properly.
    ``BilateralDenoise``  Range-weighted averaging: the classic
        edge-preserving smoother.
    ``GuidedFilterDenoise`` He et al.'s guided filter, self-guided. Same
        edge-preserving behaviour as bilateral at O(1) per pixel.
    ``NLMDenoise``        Non-local means. Averages over similar patches
        rather than nearby ones, so repeated texture survives.
    ``WaveletDenoise``    Haar wavelet shrinkage with a BayesShrink
        threshold — a per-subband, per-image adaptive threshold estimated
        from the data rather than a tuned constant.

Signal-dependent noise
──────────────────────
All six assume noise of roughly constant strength, which sensor noise is
not: shot noise grows as the square root of the signal. Wrapping any of
them in :class:`VarianceStabilised` applies the generalised Anscombe
transform first, so the denoiser sees a signal whose noise really is
uniform, and inverts afterwards. On a Poisson-Gaussian image this is
usually worth more than the choice of denoiser.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Denoiser",
    "estimate_noise_sigma",
    "GaussianDenoise",
    "MedianDenoise",
    "BilateralDenoise",
    "GuidedFilterDenoise",
    "NLMDenoise",
    "WaveletDenoise",
    "VarianceStabilised",
    "box_filter",
    "gaussian_kernel1d",
    "haar_dwt",
    "haar_idwt",
    "generalised_anscombe",
    "inverse_generalised_anscombe",
]


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, replicate-padded. [B, C, H, W]."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return x
    k = 2 * radius + 1
    padded = F.pad(x, (radius,) * 4, mode="replicate")
    return F.avg_pool2d(padded, kernel_size=k, stride=1)


def gaussian_kernel1d(sigma: float, radius: Optional[int] = None, *,
                      device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalised 1-D Gaussian. Radius defaults to ceil(3*sigma)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _separable_blur(x: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
    """Apply a 1-D kernel along both axes, per channel, replicate-padded."""
    c = x.shape[1]
    r = (kernel1d.numel() - 1) // 2
    kx = kernel1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = kernel1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    return F.conv2d(F.pad(out, (0, 0, r, r), mode="replicate"), ky, groups=c)


def haar_dwt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor]:
    """
    One level of orthonormal 2-D Haar. [B, C, 2h, 2w] -> four [B, C, h, w].

    Returns ``(ll, lh, hl, hh)``. Orthonormal, so the coefficient variance
    equals the signal variance — which is what lets the noise level be
    estimated in the HH band and used as-is elsewhere.
    """
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError(
            f"Haar needs even dimensions, got {tuple(x.shape[-2:])}")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    return ((a + b + c + d) * 0.5, (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5, (a - b - c + d) * 0.5)


def haar_idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor,
              hh: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`haar_dwt`. Four [B, C, h, w] -> [B, C, 2h, 2w]."""
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    out = torch.zeros(*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2,
                      device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def estimate_noise_sigma(x: torch.Tensor) -> torch.Tensor:
    """
    Per-image noise sigma from the finest Haar HH band. [B, 1, 1, 1].

    The robust MAD estimator, ``median|c| / 0.6745``: the HH band of a
    natural image is nearly all noise, and the median is insensitive to
    the few coefficients that are real edges. This is what lets the
    classical baselines set their own parameters per image instead of
    being handed a constant tuned on one noise level — which would make
    every comparison at a different level unfair to them.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
    pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    _, _, _, hh = haar_dwt(x)
    b = hh.shape[0]
    return (hh.reshape(b, -1).abs().median(dim=1).values / 0.6745).view(b, 1, 1, 1)


def generalised_anscombe(x: torch.Tensor, gain: float, read_var: float,
                         ) -> torch.Tensor:
    """
    Generalised Anscombe transform: Poisson-Gaussian -> ~unit-variance.

    ``2/gain * sqrt(gain*x + 3*gain^2/8 + read_var)``. After this the
    noise standard deviation is approximately 1 everywhere, so a
    constant-sigma denoiser becomes appropriate.
    """
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")
    inner = gain * x + 3.0 * gain ** 2 / 8.0 + read_var
    return (2.0 / gain) * torch.sqrt(inner.clamp(min=0.0))


def inverse_generalised_anscombe(y: torch.Tensor, gain: float,
                                 read_var: float) -> torch.Tensor:
    """
    Algebraic inverse of :func:`generalised_anscombe`.

    The algebraic inverse is biased low for very few photons (the exact
    unbiased inverse needs a numeric table); at the signal levels this
    project works at the difference is far below the noise floor.
    """
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")
    inner = (y * gain / 2.0) ** 2
    return (inner - 3.0 * gain ** 2 / 8.0 - read_var) / gain


# ─────────────────────────────────────────────────────────────────────────────
# Denoisers
# ─────────────────────────────────────────────────────────────────────────────

class Denoiser(nn.Module):
    """
    Base class: ``forward(x) -> x_denoised``, same shape, same device.

    Denoisers are plain modules rather than :class:`BaselineModel`s
    because they are only half of a pipeline — see hdr_baselines.pipelines
    for the composition that makes a scorable model.
    """

    name: str = "denoiser"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def for_noise_sigma(cls, noise_sigma: float) -> "Denoiser":
        """
        An instance parameterised for a given noise level.

        Subclasses override; the default ignores the hint, which is right
        for the self-tuning methods.
        """
        del noise_sigma
        return cls()

    def retuned(self, noise_sigma: float) -> "Denoiser":
        """
        A copy of *this* denoiser tuned for `noise_sigma`.

        The instance-level entry point, so wrappers that cannot be rebuilt
        from a class alone (:class:`VarianceStabilised` holds an inner
        denoiser) can retune correctly.
        """
        return type(self).for_noise_sigma(noise_sigma)

    def extra_repr(self) -> str:
        return f"name='{self.name}'"


class GaussianDenoise(Denoiser):
    """Isotropic Gaussian blur — the trade-off-free floor."""

    name = "gaussian"

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)

    @classmethod
    def for_noise_sigma(cls, noise_sigma: float) -> "GaussianDenoise":
        """Blur radius scaled to the noise level."""
        return cls(sigma=max(0.5, min(3.0, 0.5 + 8.0 * float(noise_sigma))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = gaussian_kernel1d(self.sigma, device=x.device, dtype=x.dtype)
        return _separable_blur(x, k)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}"


class MedianDenoise(Denoiser):
    """
    Sliding-window median.

    The only method here that removes hot pixels outright rather than
    smearing them, which is why it is worth keeping in the lineup even
    though it flattens texture.
    """

    name = "median"

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd and >= 3, got {kernel_size}")
        self.kernel_size = int(kernel_size)

    @classmethod
    def for_noise_sigma(cls, noise_sigma: float) -> "MedianDenoise":
        """A wider median for heavier noise, capped at 5x5."""
        return cls(kernel_size=5 if float(noise_sigma) > 0.1 else 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        r = k // 2
        b, c, h, w = x.shape
        patches = F.unfold(F.pad(x, (r,) * 4, mode="replicate"),
                           kernel_size=k)                    # [B, C*k*k, H*W]
        patches = patches.view(b, c, k * k, h * w)
        return patches.median(dim=2).values.view(b, c, h, w)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}"


class BilateralDenoise(Denoiser):
    """
    Bilateral filter: spatial Gaussian times range Gaussian.

    Memory is O(window^2) times the image, so the window is deliberately
    small by default; for full-resolution frames prefer
    :class:`GuidedFilterDenoise`, which is O(1) per pixel and behaves
    similarly.
    """

    name = "bilateral"

    def __init__(self, sigma_spatial: float = 1.5, sigma_range: float = 0.1,
                 window: int = 5):
        super().__init__()
        if sigma_spatial <= 0 or sigma_range <= 0:
            raise ValueError("sigma_spatial and sigma_range must be positive")
        if window < 3 or window % 2 == 0:
            raise ValueError(f"window must be odd and >= 3, got {window}")
        self.sigma_spatial = float(sigma_spatial)
        self.sigma_range = float(sigma_range)
        self.window = int(window)

    @classmethod
    def for_noise_sigma(cls, noise_sigma: float) -> "BilateralDenoise":
        """Range sigma a few noise sigmas wide, so noise averages but edges do not."""
        return cls(sigma_spatial=1.5,
                   sigma_range=max(1e-3, 2.5 * float(noise_sigma)), window=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.window
        r = k // 2
        b, c, h, w = x.shape

        coords = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        dy, dx = torch.meshgrid(coords, coords, indexing="ij")
        spatial = torch.exp(-(dx ** 2 + dy ** 2) /
                            (2 * self.sigma_spatial ** 2)).reshape(1, 1, k * k, 1)

        patches = F.unfold(F.pad(x, (r,) * 4, mode="replicate"),
                           kernel_size=k).view(b, c, k * k, h * w)
        centre = x.view(b, c, 1, h * w)
        rangew = torch.exp(-((patches - centre) ** 2) /
                           (2 * self.sigma_range ** 2))
        weights = spatial * rangew
        out = (patches * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1e-8)
        return out.view(b, c, h, w)

    def extra_repr(self) -> str:
        return (f"sigma_spatial={self.sigma_spatial}, "
                f"sigma_range={self.sigma_range}, window={self.window}")


class GuidedFilterDenoise(Denoiser):
    """
    Self-guided filter (He, Sun & Tang).

    Fits a local linear model of the image on itself in every window; in
    flat regions the fit degenerates to the local mean (smoothing), near
    an edge the linear term dominates (preservation). ``eps`` is the
    variance below which a region counts as flat, so it plays the role
    bilateral's ``sigma_range`` does — and it is in *variance* units,
    i.e. roughly the square of the noise sigma you want removed.
    """

    name = "guided"

    def __init__(self, radius: int = 2, eps: float = 0.005):
        super().__init__()
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.radius = int(radius)
        self.eps = float(eps)

    @classmethod
    def for_noise_sigma(cls, noise_sigma: float) -> "GuidedFilterDenoise":
        """
        eps is a variance, so it must scale as sigma^2 — the default
        0.005 corresponds to sigma ~= 0.07, this project's high-noise
        preset. Leaving eps far below the noise variance makes the filter
        a near-identity, which is the single easiest way to accidentally
        report a broken baseline.
        """
        return cls(radius=2, eps=max(1e-6, (1.5 * float(noise_sigma)) ** 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.radius
        mean = box_filter(x, r)
        mean_sq = box_filter(x * x, r)
        var = (mean_sq - mean * mean).clamp(min=0.0)
        a = var / (var + self.eps)
        b = mean - a * mean
        return box_filter(a, r) * x + box_filter(b, r)

    def extra_repr(self) -> str:
        return f"radius={self.radius}, eps={self.eps:g}"


class NLMDenoise(Denoiser):
    """
    Non-local means over a bounded search window.

    Implemented the efficient way — for each candidate displacement,
    the patch distance for *every* pixel is one box filter of the shifted
    squared difference — so cost is O(search^2) convolutions rather than
    O(H*W*search^2*patch^2) inner loops.
    """

    name = "nlm"

    def __init__(self, search_radius: int = 3, patch_radius: int = 1,
                 h: float = 0.08):
        super().__init__()
        if search_radius < 1 or patch_radius < 0:
            raise ValueError("search_radius >= 1 and patch_radius >= 0 required")
        if h <= 0:
            raise ValueError(f"h must be positive, got {h}")
        self.search_radius = int(search_radius)
        self.patch_radius = int(patch_radius)
        self.h = float(h)

    @classmethod
    def for_noise_sigma(cls, noise_sigma: float) -> "NLMDenoise":
        """Filter strength proportional to the noise level."""
        return cls(search_radius=3, patch_radius=1,
                   h=max(1e-3, 1.2 * float(noise_sigma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, p = self.search_radius, self.patch_radius
        pad = F.pad(x, (s,) * 4, mode="replicate")
        _, _, h, w = x.shape

        acc = torch.zeros_like(x)
        wsum = torch.zeros_like(x)
        h2 = self.h ** 2
        for dy in range(-s, s + 1):
            for dx in range(-s, s + 1):
                shifted = pad[:, :, s + dy:s + dy + h, s + dx:s + dx + w]
                dist = box_filter((shifted - x) ** 2, p)
                weight = torch.exp(-dist / h2)
                acc = acc + weight * shifted
                wsum = wsum + weight
        return acc / wsum.clamp(min=1e-8)

    def extra_repr(self) -> str:
        return (f"search_radius={self.search_radius}, "
                f"patch_radius={self.patch_radius}, h={self.h}")


class WaveletDenoise(Denoiser):
    """
    Haar wavelet shrinkage with a BayesShrink threshold.

    The noise level is estimated per image from the finest HH band by the
    robust MAD estimator (``sigma = median|c| / 0.6745``), and each
    subband gets its own threshold ``sigma_n^2 / sigma_x`` where
    ``sigma_x`` is the estimated signal standard deviation in that band.
    Nothing is tuned by hand, which is what makes it a fair automatic
    baseline across noise levels.
    """

    name = "wavelet"

    def __init__(self, levels: int = 3, soft: bool = True):
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = int(levels)
        self.soft = bool(soft)

    @classmethod
    def for_noise_sigma(cls, noise_sigma: float) -> "WaveletDenoise":
        """BayesShrink estimates its own threshold, so the hint is unused."""
        del noise_sigma
        return cls()

    @staticmethod
    def _mad_sigma(coeffs: torch.Tensor) -> torch.Tensor:
        """Robust per-image noise sigma from a detail band. [B, 1, 1, 1]."""
        b = coeffs.shape[0]
        flat = coeffs.reshape(b, -1).abs()
        return (flat.median(dim=1).values / 0.6745).view(b, 1, 1, 1)

    def _shrink(self, band: torch.Tensor, sigma_n: torch.Tensor
                ) -> torch.Tensor:
        b = band.shape[0]
        var_y = band.reshape(b, -1).pow(2).mean(dim=1).view(b, 1, 1, 1)
        var_x = (var_y - sigma_n ** 2).clamp(min=1e-12)
        thresh = (sigma_n ** 2) / var_x.sqrt()
        if self.soft:
            return torch.sign(band) * (band.abs() - thresh).clamp(min=0.0)
        return torch.where(band.abs() > thresh, band, torch.zeros_like(band))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each level halves the resolution, so pad up to a multiple of
        # 2**levels and crop the reconstruction back.
        h, w = x.shape[-2], x.shape[-1]
        m = 2 ** self.levels
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

        current = x
        bands = []
        for _ in range(self.levels):
            ll, lh, hl, hh = haar_dwt(current)
            bands.append((lh, hl, hh))
            current = ll

        sigma_n = self._mad_sigma(bands[0][2])          # finest HH
        for level in range(self.levels - 1, -1, -1):
            lh, hl, hh = bands[level]
            current = haar_idwt(current,
                                self._shrink(lh, sigma_n),
                                self._shrink(hl, sigma_n),
                                self._shrink(hh, sigma_n))
        return current[..., :h, :w]

    def extra_repr(self) -> str:
        return f"levels={self.levels}, soft={self.soft}"


class VarianceStabilised(Denoiser):
    """
    Run any denoiser in the generalised Anscombe domain.

    Sensor noise is signal-dependent; every denoiser above assumes it is
    not. Transforming first makes the assumption true, so the same
    denoiser removes deep-shadow noise without over-smoothing the
    highlights.

    ``gain`` and ``read_var`` are in normalised [0, 1] units, i.e. the DN
    values divided by full scale — matching what the datasets emit. For
    the project's high-noise preset at 10 bits that is
    ``gain = 14/1023 ~= 0.0137`` and ``read_var = 150/1023^2 ~= 1.4e-4``.
    """

    name = "vst"

    def __init__(self, inner: Denoiser, gain: float = 14.0 / 1023.0,
                 read_var: float = 150.0 / (1023.0 ** 2),
                 auto_inner: bool = True):
        super().__init__()
        if not isinstance(inner, Denoiser):
            raise TypeError(
                f"inner must be a Denoiser, got {type(inner).__name__}")
        self.inner = inner
        self.gain = float(gain)
        self.read_var = float(read_var)
        self.auto_inner = bool(auto_inner)
        self.name = f"vst_{inner.name}"

    def retuned(self, noise_sigma: float) -> "VarianceStabilised":
        """
        Return self: the inner denoiser must be tuned to the noise level
        *in the stabilised domain*, which is not the level measured
        outside it. :meth:`forward` does that itself, per image.
        """
        del noise_sigma
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = generalised_anscombe(x, self.gain, self.read_var)
        # After the transform the noise sigma is ~1 in transform units;
        # rescale to [0, 1]-ish so the inner denoiser's tuned constants
        # (sigma_range, eps, h) still mean what they were tuned to mean.
        scale = y.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        z_in = y / scale

        inner = self.inner
        if self.auto_inner:
            sigma = float(estimate_noise_sigma(z_in).mean())
            if sigma > 1e-6:
                inner = self.inner.retuned(sigma).to(x.device)

        z = inner(z_in) * scale
        return inverse_generalised_anscombe(z, self.gain, self.read_var)

    def extra_repr(self) -> str:
        return (f"inner={self.inner.name}, gain={self.gain:g}, "
                f"read_var={self.read_var:g}, auto_inner={self.auto_inner}")
