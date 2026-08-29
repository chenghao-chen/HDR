"""
hdr_data/noise.py — sensor noise synthesis, calibration and analysis
=====================================================================

The training pipeline's realism ceiling is its noise model. The original
``HDR_Mobile_dataset.add_photon_noise`` implements the core of it: a
Poisson-Gaussian model in digital numbers (DN),

    var[DN^2] = shot_gain * signal[DN] + read_var

with ``shot_gain = 14`` and ``read_var ~ U(135, 160)`` at 10 bits, plus a
triangular low-light exposure factor and an optional highlight offset that
pushes part of the frame into saturation.

This module keeps that model *bit-exact* (see :meth:`NoiseModel.legacy` and
the ``legacy_add_photon_noise`` equivalence test) and adds the parts a real
CMOS sensor has that the original leaves out, each independently switchable
and off by default:

  * per-channel gain — the four CFA channels have different quantum
    efficiencies, so green is systematically less noisy than blue;
  * PRNU — multiplicative fixed-pattern gain, the dominant non-uniformity
    once the signal is bright;
  * row/column FPN — additive banding from per-row and per-column readout
    offsets, which denoisers trained on i.i.d. noise handle badly;
  * hot pixels — a sparse set of stuck-high sensels;
  * explicit quantisation and saturation control.

It also provides the inverse direction — :func:`calibrate_from_pairs`
recovers ``(shot_gain, read_var)`` from clean/noisy image pairs by
least-squares on binned variance-vs-signal — so a preset can be checked
against, or fitted to, real captures rather than assumed.

Conventions
───────────
* ``apply()`` takes an *unnormalised* HDR tensor plus the (min, max) range
  of the image it was cropped from, and returns ``(noisy, clean)`` both
  normalised to [0, 1] — exactly the contract MobileHDRDataset relies on.
* Every stochastic draw goes through an optional ``torch.Generator`` so a
  test split can be made deterministic per index.
* Nothing here is Bayer-specific except ``channel_gain``, which is applied
  along dim -3 when the input is packed [..., 4, h, w].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "NoiseParams",
    "NoiseModel",
    "NOISE_PRESETS",
    "get_preset",
    "list_presets",
    "poisson_gaussian",
    "sigma_from_params",
    "calibrate_from_pairs",
    "snr_db",
]


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NoiseParams:
    """
    A complete sensor noise description.

    The first four fields reproduce the original model; everything after
    ``do_expand`` is an addition that defaults to "disabled", so
    ``NoiseParams()`` is the legacy behaviour.

    Attributes
    ──────────
    nbits
        Bit depth of the virtual ADC. Signals are scaled to
        ``[0, 2**nbits - 1]`` DN before noise is applied.
    shot_gain
        Photon (shot) noise slope in DN^2 per DN. Physically this is the
        sensor's conversion gain: 14 at 10 bits gives sigma ~= 120 DN in
        the highlights.
    read_noise_range
        ``(lo, hi)`` for the per-image uniform draw of read variance in
        DN^2. Set both to the same value for a fixed read noise.
    random_alpha
        Sample a low-light exposure factor per image. The legacy sampler
        is triangular, ``|u1 - u2|``, which peaks at 0 and so biases hard
        toward very dark frames.
    alpha_mode
        ``"triangular"`` (legacy), ``"uniform"``, ``"log_uniform"`` or
        ``"fixed"``. Only consulted when ``random_alpha`` is True.
    alpha_range
        ``(lo, hi)`` bounds for the non-legacy alpha samplers.
    alpha_floor
        Alpha is clamped up to this to avoid a pure-black frame.
    do_expand / expand_prob / expand_log2_range
        With probability ``expand_prob``, add a constant offset of
        ``2 ** U(*expand_log2_range)`` DN so the frame clips at full scale
        and the model sees genuine saturation.
    channel_gain
        Optional per-packed-channel multiplicative gain applied to the
        clean signal before noise, modelling per-CFA-channel quantum
        efficiency. Length must match the channel dim (4 for packed Bayer).
    prnu_std
        Standard deviation of the multiplicative per-pixel gain (PRNU),
        e.g. 0.01 for 1%. Drawn fresh per call.
    row_fpn_std / col_fpn_std
        Standard deviation, in DN, of additive per-row / per-column
        offsets — readout banding. Drawn fresh per call.
    hot_pixel_rate
        Fraction of pixels forced to ``hot_pixel_level * pix_max``.
    quantize
        Round to integer DN before normalising (the legacy model does).
    clip_high
        Clamp at full scale, i.e. model sensor saturation (legacy: True).
    """

    nbits: int = 10
    shot_gain: float = 14.0
    read_noise_range: Tuple[float, float] = (135.0, 160.0)

    random_alpha: bool = True
    alpha_mode: str = "triangular"
    alpha_range: Tuple[float, float] = (0.0, 1.0)
    alpha_floor: float = 0.01

    do_expand: bool = False
    expand_prob: float = 0.3
    expand_log2_range: Tuple[float, float] = (5.0, 11.0)

    channel_gain: Optional[Tuple[float, ...]] = None
    prnu_std: float = 0.0
    row_fpn_std: float = 0.0
    col_fpn_std: float = 0.0
    hot_pixel_rate: float = 0.0
    hot_pixel_level: float = 1.0

    quantize: bool = True
    clip_high: bool = True

    def __post_init__(self):
        if self.nbits < 1 or self.nbits > 32:
            raise ValueError(f"nbits must be in [1, 32], got {self.nbits}")
        if self.shot_gain < 0:
            raise ValueError(f"shot_gain must be >= 0, got {self.shot_gain}")
        lo, hi = self.read_noise_range
        if lo < 0 or hi < lo:
            raise ValueError(
                f"read_noise_range must be (lo, hi) with 0 <= lo <= hi, "
                f"got {self.read_noise_range}")
        if self.alpha_mode not in ("triangular", "uniform", "log_uniform", "fixed"):
            raise ValueError(f"Unknown alpha_mode '{self.alpha_mode}'")
        a_lo, a_hi = self.alpha_range
        if not (0.0 <= a_lo <= a_hi <= 1.0):
            raise ValueError(f"alpha_range must satisfy 0<=lo<=hi<=1, got {self.alpha_range}")
        if self.alpha_mode == "log_uniform" and a_lo <= 0.0:
            raise ValueError("log_uniform alpha_range needs lo > 0")
        if not 0.0 <= self.expand_prob <= 1.0:
            raise ValueError(f"expand_prob must be in [0, 1], got {self.expand_prob}")
        if not 0.0 <= self.hot_pixel_rate <= 1.0:
            raise ValueError(f"hot_pixel_rate must be in [0, 1], got {self.hot_pixel_rate}")
        for name in ("prnu_std", "row_fpn_std", "col_fpn_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    @property
    def pix_max(self) -> float:
        """Full-scale value in DN."""
        return float(2 ** self.nbits - 1)

    @property
    def is_legacy(self) -> bool:
        """True when no post-legacy feature is enabled."""
        return (self.channel_gain is None
                and self.prnu_std == 0.0
                and self.row_fpn_std == 0.0
                and self.col_fpn_std == 0.0
                and self.hot_pixel_rate == 0.0
                and self.alpha_mode == "triangular"
                and self.quantize
                and self.clip_high)

    def replace(self, **kwargs) -> "NoiseParams":
        """A copy with fields overridden (dataclasses.replace)."""
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict view, for CSV/JSON run metadata."""
        return {
            "nbits": self.nbits,
            "shot_gain": self.shot_gain,
            "read_noise_range": list(self.read_noise_range),
            "random_alpha": self.random_alpha,
            "alpha_mode": self.alpha_mode,
            "alpha_range": list(self.alpha_range),
            "alpha_floor": self.alpha_floor,
            "do_expand": self.do_expand,
            "expand_prob": self.expand_prob,
            "expand_log2_range": list(self.expand_log2_range),
            "channel_gain": (None if self.channel_gain is None
                             else list(self.channel_gain)),
            "prnu_std": self.prnu_std,
            "row_fpn_std": self.row_fpn_std,
            "col_fpn_std": self.col_fpn_std,
            "hot_pixel_rate": self.hot_pixel_rate,
            "hot_pixel_level": self.hot_pixel_level,
            "quantize": self.quantize,
            "clip_high": self.clip_high,
        }


#: Named parameter sets. "mobile_hdr" is the model the released checkpoints
#: were trained under; the graded set below is for stratified evaluation.
NOISE_PRESETS: Dict[str, NoiseParams] = {
    # Exactly HDR_Mobile_dataset.add_photon_noise's defaults.
    "mobile_hdr": NoiseParams(),
    "legacy": NoiseParams(),

    # Graded severity at a fixed exposure, for "how does PSNR fall off with
    # noise level" curves. random_alpha is off so the only variable is noise.
    "clean": NoiseParams(shot_gain=0.0, read_noise_range=(0.0, 0.0),
                         random_alpha=False),
    "low": NoiseParams(shot_gain=1.0, read_noise_range=(4.0, 6.0),
                       random_alpha=False),
    "medium": NoiseParams(shot_gain=4.0, read_noise_range=(30.0, 40.0),
                          random_alpha=False),
    "high": NoiseParams(shot_gain=14.0, read_noise_range=(135.0, 160.0),
                        random_alpha=False),
    "extreme": NoiseParams(shot_gain=32.0, read_noise_range=(400.0, 480.0),
                           random_alpha=False),

    # High noise plus the structured artefacts a real readout adds.
    "realistic": NoiseParams(
        shot_gain=14.0, read_noise_range=(135.0, 160.0),
        channel_gain=(0.92, 1.0, 1.0, 0.88),   # B, G1, G2, R quantum efficiency
        prnu_std=0.01, row_fpn_std=2.0, col_fpn_std=1.0,
        hot_pixel_rate=1e-5,
    ),
    # Low light + banding: the case the MoE router is supposed to specialise on.
    "realistic_lowlight": NoiseParams(
        shot_gain=14.0, read_noise_range=(135.0, 160.0),
        random_alpha=True, alpha_mode="log_uniform", alpha_range=(0.01, 0.3),
        channel_gain=(0.92, 1.0, 1.0, 0.88),
        prnu_std=0.01, row_fpn_std=3.0, col_fpn_std=1.5,
        hot_pixel_rate=1e-5,
    ),
}


def list_presets() -> Tuple[str, ...]:
    """Names accepted by :func:`get_preset`, sorted."""
    return tuple(sorted(NOISE_PRESETS))


def get_preset(name: str) -> NoiseParams:
    """Look up a preset by name, with a helpful error on a typo."""
    if name not in NOISE_PRESETS:
        raise KeyError(
            f"Unknown noise preset '{name}'. Available: {list(list_presets())}")
    return NOISE_PRESETS[name]


# ─────────────────────────────────────────────────────────────────────────────
# Sampling primitives
# ─────────────────────────────────────────────────────────────────────────────

def _rand(generator: Optional[torch.Generator] = None) -> float:
    """One uniform [0,1) scalar — the legacy draw, kept for bit-exactness."""
    return torch.rand((), generator=generator).item()


def _randn_like(x: torch.Tensor,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Standard normal shaped like x, honouring a CPU generator."""
    return torch.randn(x.shape, generator=generator, dtype=x.dtype,
                       device=x.device)


def sigma_from_params(signal_dn: torch.Tensor, params: NoiseParams,
                      read_var: Optional[float] = None) -> torch.Tensor:
    """
    Analytic per-pixel noise sigma in DN for a clean signal in DN.

    ``read_var`` defaults to the midpoint of ``params.read_noise_range``,
    which is the right choice when the caller wants an expectation rather
    than the realisation of one particular draw.

    This is what the evaluation harness uses to stratify results by
    predicted SNR without having to re-run the sampler.
    """
    if read_var is None:
        lo, hi = params.read_noise_range
        read_var = 0.5 * (lo + hi)
    var = params.shot_gain * signal_dn.clamp(min=0.0) + float(read_var)
    return torch.sqrt(var.clamp(min=0.0))


def snr_db(signal_dn: torch.Tensor, params: NoiseParams,
           read_var: Optional[float] = None, eps: float = 1e-8) -> torch.Tensor:
    """Per-pixel SNR in dB implied by the noise model: 20*log10(S / sigma)."""
    sigma = sigma_from_params(signal_dn, params, read_var)
    return 20.0 * torch.log10(signal_dn.clamp(min=0.0) / (sigma + eps) + eps)


def poisson_gaussian(clean_dn: torch.Tensor, shot_gain: float, read_var: float,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Gaussian approximation to Poisson-Gaussian noise, in DN.

    The Gaussian approximation to the shot term is what the original model
    uses; it is accurate above a few tens of photons and much cheaper than
    sampling a Poisson variate per pixel. Below that it under-disperses
    slightly, which matters only for the very darkest presets.
    """
    sigma = torch.sqrt(shot_gain * clean_dn.clamp(min=0.0) + read_var)
    return clean_dn + sigma * _randn_like(clean_dn, generator)


# ─────────────────────────────────────────────────────────────────────────────
# The model
# ─────────────────────────────────────────────────────────────────────────────

class NoiseModel:
    """
    Callable sensor simulator.

        model = NoiseModel(get_preset("realistic"))
        noisy, clean = model.apply(hdr_tensor, norm_min=lo, norm_max=hi)

    Both outputs are normalised to [0, 1]; ``clean`` is the noise-free
    signal *after* exposure scaling and saturation, i.e. the correct
    supervision target for the noisy input beside it.

    The draw order is fixed and documented so that a fixed generator seed
    reproduces a run exactly:

        1. alpha   (2 uniforms for "triangular", 1 otherwise)
        2. expand  (1 uniform for the coin, +1 if it lands)
        3. read_var (1 uniform)
        4. channel gain / PRNU / FPN / hot pixels (only if enabled)
        5. the pixel noise field (1 normal per pixel)

    Steps 1-3 and 5 match the legacy function exactly, so a legacy-preset
    model with the same seed produces bit-identical output.
    """

    def __init__(self, params: NoiseParams = NoiseParams()):
        if not isinstance(params, NoiseParams):
            raise TypeError(
                f"params must be a NoiseParams, got {type(params).__name__}")
        self.params = params

    # -- construction ----------------------------------------------------
    @classmethod
    def from_preset(cls, name: str) -> "NoiseModel":
        """Build from a name in :data:`NOISE_PRESETS`."""
        return cls(get_preset(name))

    @classmethod
    def legacy(cls, shot_gain: float = 14.0,
               read_noise_range: Sequence[float] = (135.0, 160.0),
               nbits: int = 10, random_alpha: bool = True,
               do_expand: bool = False) -> "NoiseModel":
        """The original ``add_photon_noise`` configuration."""
        return cls(NoiseParams(
            nbits=nbits, shot_gain=shot_gain,
            read_noise_range=tuple(read_noise_range),
            random_alpha=random_alpha, do_expand=do_expand,
        ))

    def __repr__(self) -> str:
        p = self.params
        return (f"NoiseModel(shot_gain={p.shot_gain}, "
                f"read_noise_range={p.read_noise_range}, nbits={p.nbits}, "
                f"alpha={p.alpha_mode if p.random_alpha else 'off'})")

    # -- individual stages ------------------------------------------------
    def sample_alpha(self, generator: Optional[torch.Generator] = None) -> float:
        """
        Draw the exposure factor. The legacy triangular sampler consumes
        two uniforms, which is why the draw count is part of the contract.
        """
        p = self.params
        if not p.random_alpha:
            return 1.0
        lo, hi = p.alpha_range
        if p.alpha_mode == "triangular":
            # |u1 - u2| on [0,1]: density peaks at 0 -> mostly very dark.
            a = abs(_rand(generator) - _rand(generator))
            a = lo + (hi - lo) * a
        elif p.alpha_mode == "uniform":
            a = lo + (hi - lo) * _rand(generator)
        elif p.alpha_mode == "log_uniform":
            a = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * _rand(generator))
        else:                                    # "fixed"
            a = hi
        return max(a, p.alpha_floor)

    def sample_read_var(self, generator: Optional[torch.Generator] = None) -> float:
        """Draw the per-image read variance in DN^2."""
        lo, hi = self.params.read_noise_range
        return lo + (hi - lo) * _rand(generator)

    def _apply_expand(self, clean: torch.Tensor,
                      generator: Optional[torch.Generator]) -> torch.Tensor:
        """Optionally lift the whole frame so highlights clip at full scale."""
        p = self.params
        if not p.do_expand:
            return clean
        if _rand(generator) >= p.expand_prob:
            return clean
        lo, hi = p.expand_log2_range
        offset = 2.0 ** (lo + (hi - lo) * _rand(generator))
        return clean + offset

    def _apply_channel_gain(self, clean: torch.Tensor) -> torch.Tensor:
        """Per-CFA-channel quantum efficiency, applied along dim -3."""
        gains = self.params.channel_gain
        if gains is None:
            return clean
        n = clean.shape[-3] if clean.dim() >= 3 else 1
        if len(gains) != n:
            raise ValueError(
                f"channel_gain has {len(gains)} entries but the tensor has "
                f"{n} channels ({tuple(clean.shape)}).")
        g = torch.as_tensor(gains, dtype=clean.dtype, device=clean.device)
        return clean * g.view(*([1] * (clean.dim() - 3)), n, 1, 1)

    def _apply_prnu(self, clean: torch.Tensor,
                    generator: Optional[torch.Generator]) -> torch.Tensor:
        """Multiplicative per-pixel gain non-uniformity."""
        std = self.params.prnu_std
        if std <= 0:
            return clean
        gain = 1.0 + std * _randn_like(clean, generator)
        return clean * gain.clamp(min=0.0)

    def _apply_fpn(self, noisy: torch.Tensor,
                   generator: Optional[torch.Generator]) -> torch.Tensor:
        """
        Additive row and column offsets in DN — readout banding.

        The offsets are constant along their axis and shared across
        channels, which is what makes banding visually structured (and
        what an i.i.d.-noise-trained denoiser smears rather than removes).
        """
        p = self.params
        if p.row_fpn_std <= 0 and p.col_fpn_std <= 0:
            return noisy
        h, w = noisy.shape[-2], noisy.shape[-1]
        out = noisy
        if p.row_fpn_std > 0:
            rows = torch.randn((h, 1), generator=generator, dtype=noisy.dtype,
                               device=noisy.device) * p.row_fpn_std
            out = out + rows
        if p.col_fpn_std > 0:
            cols = torch.randn((1, w), generator=generator, dtype=noisy.dtype,
                               device=noisy.device) * p.col_fpn_std
            out = out + cols
        return out

    def _apply_hot_pixels(self, noisy: torch.Tensor,
                          generator: Optional[torch.Generator]) -> torch.Tensor:
        """Force a sparse random set of sensels to (near) full scale."""
        p = self.params
        if p.hot_pixel_rate <= 0:
            return noisy
        u = torch.rand(noisy.shape, generator=generator, dtype=noisy.dtype,
                       device=noisy.device)
        level = p.hot_pixel_level * p.pix_max
        return torch.where(u < p.hot_pixel_rate,
                           torch.full_like(noisy, level), noisy)

    # -- the whole pipeline ----------------------------------------------
    def apply(self, image: torch.Tensor, *,
              norm_min: Optional[float] = None,
              norm_max: Optional[float] = None,
              generator: Optional[torch.Generator] = None,
              return_meta: bool = False):
        """
        Simulate a capture.

        Parameters
        ──────────
        image
            Unnormalised HDR tensor, any shape ending in [..., H, W].
        norm_min, norm_max
            Range used to scale `image` into DN. Pass the *full frame's*
            min/max when `image` is a crop, so crops keep their absolute
            brightness (a dark crop must stay dark). Defaults to the
            tensor's own min/max.
        generator
            CPU generator for reproducible draws.
        return_meta
            Also return the realised ``{"alpha", "read_var", ...}``.

        Returns
        ──────
        ``(noisy, clean)`` in [0, 1], or ``(noisy, clean, meta)``.
        """
        p = self.params
        pix_max = p.pix_max

        rmin = image.min() if norm_min is None else norm_min
        rmax = image.max() if norm_max is None else norm_max
        rng = float(rmax) - float(rmin)
        scaled = ((image - rmin) / rng * pix_max) if rng > 1e-6 else (image * 0)

        alpha = self.sample_alpha(generator)
        clean = scaled * alpha
        clean = self._apply_expand(clean, generator)
        clean = clean.clamp(min=0.0)

        read_var = self.sample_read_var(generator)

        # Signal-dependent stages happen before the noise draw so that the
        # shot term sees the gain-modulated signal, as on a real sensor.
        clean = self._apply_channel_gain(clean)
        clean = self._apply_prnu(clean, generator)

        noisy = poisson_gaussian(clean, p.shot_gain, read_var, generator)
        noisy = self._apply_fpn(noisy, generator)
        noisy = self._apply_hot_pixels(noisy, generator)

        if p.quantize:
            noisy = torch.round(noisy)
        hi = pix_max if p.clip_high else float("inf")
        noisy = noisy.clamp(0.0, hi) / pix_max
        gt = clean.clamp(0.0, hi) / pix_max

        if not return_meta:
            return noisy, gt
        meta = {
            "alpha": alpha,
            "read_var": read_var,
            "pix_max": pix_max,
            "norm_min": float(rmin),
            "norm_max": float(rmax),
        }
        return noisy, gt, meta

    __call__ = apply


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_from_pairs(clean: torch.Tensor, noisy: torch.Tensor, *,
                         nbits: int = 10, num_bins: int = 32,
                         min_count: int = 64, max_clip_frac: float = 0.01,
                         normalised: bool = True) -> Dict[str, float]:
    """
    Recover ``(shot_gain, read_var)`` from paired clean/noisy images.

    The Poisson-Gaussian model says the residual variance is affine in the
    signal, ``var = shot_gain * signal + read_var`` (all in DN), so binning
    pixels by clean level and least-squares fitting the per-bin variance
    recovers both constants.

    Two classes of bin are excluded, and both matter: bins holding fewer
    than ``min_count`` pixels (their variance estimate is too noisy to fit
    through), and bins where more than ``max_clip_frac`` of the pixels sit
    against 0 or full scale. Clipping truncates the residual distribution
    and so *depresses* the measured variance exactly where the signal is
    darkest and brightest; fitting through those bins pulls the slope down
    and the intercept up. Skipping them is what makes the fit recover the
    parameters it was given rather than a flatter compromise line.

    Parameters
    ──────────
    clean, noisy
        Same-shaped tensors. With ``normalised=True`` (the default) they
        are taken to be in [0, 1] and are scaled up by ``2**nbits - 1``
        first, which is the form :meth:`NoiseModel.apply` returns.
    max_clip_frac
        Reject a bin when this fraction of its pixels are clipped. Set to
        1.0 to disable the check.

    Returns
    ──────
    ``{"shot_gain", "read_var", "r2", "num_bins_used", "num_pixels",
    "num_bins_clipped"}``.
    ``r2`` is the coefficient of determination of the affine fit — a value
    well below ~0.9 means the data does not follow this model (structured
    noise, misalignment, or a clipped signal dominating the fit).
    """
    if clean.shape != noisy.shape:
        raise ValueError(
            f"clean {tuple(clean.shape)} and noisy {tuple(noisy.shape)} "
            f"must have the same shape.")
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")

    scale = float(2 ** nbits - 1) if normalised else 1.0
    c = clean.detach().reshape(-1).double() * scale
    n = noisy.detach().reshape(-1).double() * scale
    resid = n - c

    lo, hi = float(c.min()), float(c.max())
    if hi - lo < 1e-9:
        raise ValueError(
            "Clean signal is constant; cannot separate shot noise from read "
            "noise without a range of signal levels.")

    edges = torch.linspace(lo, hi, num_bins + 1, dtype=torch.float64)
    idx = torch.bucketize(c, edges[1:-1].contiguous(), right=False)

    pix_max = float(2 ** nbits - 1)
    clipped = (n <= 0.0) | (n >= pix_max)

    xs, ys, counts = [], [], []
    num_clipped_bins = 0
    for b in range(num_bins):
        sel = idx == b
        cnt = int(sel.sum())
        if cnt < min_count:
            continue
        if float(clipped[sel].double().mean()) > max_clip_frac:
            num_clipped_bins += 1
            continue
        xs.append(float(c[sel].mean()))
        ys.append(float(resid[sel].var(unbiased=True)))
        counts.append(cnt)

    if len(xs) < 2:
        raise ValueError(
            f"Only {len(xs)} usable bins (min_count={min_count}, "
            f"max_clip_frac={max_clip_frac}, {num_clipped_bins} bins rejected "
            f"for clipping); need at least 2. Use a larger image, fewer bins, "
            f"or a signal that spans more of the unclipped range.")

    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    w = torch.tensor(counts, dtype=torch.float64)
    w = w / w.sum()

    # Weighted least squares for y = a*x + b, weighting by bin population.
    xm = float((w * x).sum())
    ym = float((w * y).sum())
    var_x = float((w * (x - xm) ** 2).sum())
    if var_x < 1e-12:
        raise ValueError("Bin centres are degenerate; cannot fit.")
    cov = float((w * (x - xm) * (y - ym)).sum())
    a = cov / var_x
    b = ym - a * xm

    pred = a * x + b
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    return {
        "shot_gain": a,
        "read_var": b,
        "r2": r2,
        "num_bins_used": len(xs),
        "num_bins_clipped": num_clipped_bins,
        "num_pixels": int(c.numel()),
    }
