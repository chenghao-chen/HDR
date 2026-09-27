"""
Adapter making USVD a drop-in alternative to the MoE and FiLM denoisers.

The project's models share one signature -- ``(x, snr_map) -> (blended,
expert_outs, gates)`` -- so the trainers, the evaluation harness and the
checkpoint loader can treat any of them interchangeably. USVD's own
signature is ``(y, sigma_map=None) -> (x, sigma_map)``, so this wraps it.

Two adaptations are worth stating explicitly, because they are the places
where the published spec and this project's task do not line up:

* **Resolution.** USVD is flat: it denoises at the resolution it is given
  and the output has the same shape as the input. That matches RAW->RAW
  exactly (4 packed channels in, 4 out). The HDR task needs sensor-
  resolution RGB, so with ``upsample=True`` a single 1x1 projection plus
  PixelShuffle(2) is appended. That is the smallest addition that reaches
  the target shape without reintroducing an encoder-decoder, which the
  design explicitly rejects.

* **Noise conditioning.** The other models are handed the project's
  ``snr_map``. USVD estimates its own per-pixel sigma, which is central to
  the design rather than incidental, so ``snr_map`` is accepted for
  signature compatibility and ignored by default. Pass
  ``use_external_snr=True`` to convert it to a sigma instead and compare the
  two conditioning sources.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model.usvd import USVD

__all__ = ["USVDDenoiser"]

_CLAMP_EPS = 1.0 / (2 ** 20 - 1)


class USVDDenoiser(nn.Module):
    def __init__(self, out_channels: int = 3, in_channels: int = 4,
                 upsample: bool = True, num_iters: int = 8,
                 share_weights: bool = False, patch_size: int = 8,
                 top_k: int = 16, d_model: int = 256, hidden_mult: int = 2,
                 alpha_init: float = 0.1, chunk_size: int = 1024,
                 checkpoint: bool = False, use_external_snr: bool = False,
                 **unused):
        super().__init__()
        # Accepts and discards the trunk hyperparameters (dim, num_blocks,
        # heads, se_reduction, ...) that the shared training scripts pass to
        # every mode. USVD has no trunk, so they have no meaning here; taking
        # them keeps one model_kwargs dict usable across all modes.
        super().__setattr__("_ignored_kwargs", sorted(unused))

        self.num_experts = 1          # contract compatibility
        self.upsample = bool(upsample)
        self.use_external_snr = bool(use_external_snr)

        self.core = USVD(in_channels=in_channels, num_iters=num_iters,
                         share_weights=share_weights, patch_size=patch_size,
                         top_k=top_k, d_model=d_model, hidden_mult=hidden_mult,
                         alpha_init=alpha_init, chunk_size=chunk_size,
                         checkpoint=checkpoint)

        fan = 4 if self.upsample else 1
        self.proj_out = nn.Conv2d(in_channels, out_channels * fan, 1)
        self.up = nn.PixelShuffle(2) if self.upsample else nn.Identity()
        # Small positive output rather than zero: the project has twice hit
        # dead gradients from an exactly-zero final layer feeding a clamp
        # whose backward is zero there. See ExpertHead in
        # HDR_model_hybrid_Teacher.py.
        nn.init.normal_(self.proj_out.weight, std=1e-3)
        nn.init.constant_(self.proj_out.bias, 0.005)

    def forward(self, x, snr_map=None):
        sigma_in = None
        if self.use_external_snr and snr_map is not None:
            # snr_map is a signal-to-noise estimate, so sigma goes as its
            # reciprocal; without the floor a noise-free region divides by 0.
            sigma_in = 1.0 / snr_map.clamp_min(1e-3)

        denoised, sigma_map = self.core(x, sigma_in)
        out = self.up(self.proj_out(denoised)).clamp(min=_CLAMP_EPS)

        expert_outs = out.unsqueeze(1)
        gates = torch.ones_like(out[:, :1])
        if not self.training:
            out = out.clamp(max=1.0)
            expert_outs = expert_outs.clamp(max=1.0)
        self.last_sigma_map = sigma_map
        return out, expert_outs, gates
