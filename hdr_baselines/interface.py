"""
hdr_baselines/interface.py — one signature for every comparison model
======================================================================

A benchmark is only as good as the things it compares against. To make
"the MoE teacher versus a classical pipeline versus a small learned net" a
single loop rather than three special cases, everything in this package
presents the project's own model signature:

    blended, expert_outs, gates = model(x, snr_map)

      x            [B, 4, h, w]  packed CFA in [0, 1]
      snr_map      [B, 1, h, w]  normalised local SNR (may be ignored)
      blended      [B, 3, 2h, 2w] sensor-resolution RGB
      expert_outs  [B, K, 3, 2h, 2w]
      gates        [B, K, 2h, 2w] summing to 1 over K

A model with no routing reports K = 1, its own output as the single
expert, and an all-ones gate — so per-expert and gate-usage columns stay
meaningful (and trivially correct) instead of being blank.

Subclasses implement :meth:`BaselineModel.predict_rgb` and get the rest.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["BaselineModel", "SINGLE_EXPERT_DOC"]

SINGLE_EXPERT_DOC = (
    "Reports K=1: its own output as the single expert and an all-ones gate."
)


class BaselineModel(nn.Module):
    """
    Base class for every comparison model.

    Attributes
    ──────────
    name
        Registry name, used in reports and output paths.
    pattern
        CFA phase the model assumes. Classical demosaicers are
        pattern-generic; GBTF is BGGR-only and says so.
    trainable
        False for the classical pipelines, True for the learned
        references — the runner uses this to decide whether an untrained
        model's numbers are meaningful or just a random-init smoke test.
    """

    name: str = "baseline"
    pattern: str = "BGGR"
    trainable: bool = False

    def __init__(self, name: Optional[str] = None, pattern: str = "BGGR"):
        super().__init__()
        if name is not None:
            self.name = name
        self.pattern = pattern
        self.num_experts = 1

    # ── the one thing subclasses implement ───────────────────────────────
    def predict_rgb(self, x: torch.Tensor,
                    snr_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Packed CFA [B, 4, h, w] -> sensor-resolution RGB [B, 3, 2h, 2w].
        """
        raise NotImplementedError

    # ── the shared wrapper ───────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                snr_map: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4 or x.shape[1] != 4:
            raise ValueError(
                f"{self.name}: expected packed [B, 4, h, w], got "
                f"{tuple(x.shape)}.")
        rgb = self.predict_rgb(x, snr_map)
        expected = (x.shape[0], 3, x.shape[2] * 2, x.shape[3] * 2)
        if tuple(rgb.shape) != expected:
            raise RuntimeError(
                f"{self.name}: predict_rgb returned {tuple(rgb.shape)}, "
                f"expected {expected}.")
        experts = rgb.unsqueeze(1)
        gates = torch.ones((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]),
                           device=rgb.device, dtype=rgb.dtype)
        return rgb, experts, gates

    def extra_repr(self) -> str:
        return f"name='{self.name}', pattern='{self.pattern}', trainable={self.trainable}"
