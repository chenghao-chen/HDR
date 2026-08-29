"""
hdr_eval/checkpoints.py — rebuilding a trained model from a .pth
=================================================================

``train_A100_MoE_two_phase.py`` stores the architecture alongside the
weights — ``mode``, ``model_kwargs`` and ``num_experts`` — precisely so a
benchmark does not have to be kept in sync by hand. This module reads
them.

One difference from ``test_dual_MoE_two_phase.load_model_from_checkpoint``
is deliberate and worth stating: that function currently hardcodes

    num_experts = 2 #ckpt.get("num_experts", fallback_num_experts)

so a 3-expert checkpoint is rebuilt as a 2-expert model and the load
fails, or (worse, for a `dual` checkpoint where the count happens to
match) succeeds while describing the model wrongly in the report. Here
the stored value is used, and the fallback applies only when the
checkpoint predates the metadata.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

__all__ = ["CheckpointInfo", "load_checkpoint", "load_model_from_checkpoint"]

#: Architecture used for checkpoints written before model_kwargs was stored.
LEGACY_MODEL_KWARGS: Dict[str, Any] = {
    "dim": 32,
    "num_blocks": [4, 4, 4, 4],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    # Checkpoints from before SE blocks existed need se_reduction=None;
    # there is no way to detect that from the file, so a load failure
    # mentioning unexpected 'se.' keys means this needs to be None.
    "se_reduction": 8,
}


class CheckpointInfo:
    """What a checkpoint says about itself."""

    def __init__(self, path: str, payload: Dict[str, Any]):
        self.path = path
        self.mode = payload.get("mode", "dual")
        self.model_kwargs = payload.get("model_kwargs") or {}
        self.num_experts = payload.get("num_experts")
        self.epoch = payload.get("epoch")
        self.phase = payload.get("phase")
        self.best_psnr_mu = payload.get("best_psnr_mu")
        self.has_metadata = "mode" in payload and "model_kwargs" in payload

    def __repr__(self) -> str:
        return (f"CheckpointInfo(mode='{self.mode}', "
                f"num_experts={self.num_experts}, epoch={self.epoch}, "
                f"phase={self.phase}, best_psnr_mu={self.best_psnr_mu})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "num_experts": self.num_experts,
            "epoch": self.epoch,
            "phase": self.phase,
            "best_psnr_mu": self.best_psnr_mu,
            "model_kwargs": dict(self.model_kwargs),
            "has_metadata": self.has_metadata,
        }


def load_checkpoint(path: str, device=None) -> Tuple[Dict[str, Any],
                                                     CheckpointInfo]:
    """Read a checkpoint file and its metadata, without building a model."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint at {path}")
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} does not contain a checkpoint dict "
            f"(got {type(payload).__name__}).")
    return payload, CheckpointInfo(path, payload)


def load_model_from_checkpoint(path: str, device=None,
                               fallback_kwargs: Optional[Dict[str, Any]] = None,
                               fallback_num_experts: int = 2,
                               fallback_mode: str = "dual",
                               strict: bool = True):
    """
    Rebuild the trained architecture and load its weights.

    Returns ``(model, info)``. The model is in eval mode, which matters:
    the teacher and the MoE only apply their upper clamp outside training.

    The three ``fallback_*`` arguments apply only to checkpoints written
    before the training script stored its metadata. ``fallback_mode``
    defaults to "dual" to match the old test script's assumption, but a
    legacy single- or moe-mode checkpoint needs it set — there is nothing
    in the file to infer it from, and a wrong guess surfaces as a
    state_dict mismatch rather than silently wrong weights.
    """
    payload, info = load_checkpoint(path, device)
    from HDR_model_hybrid_Teacher import build_denoiser

    if not info.has_metadata:
        info.mode = fallback_mode

    kwargs = dict(info.model_kwargs or fallback_kwargs or LEGACY_MODEL_KWARGS)
    num_experts = (info.num_experts if info.num_experts is not None
                   else fallback_num_experts)

    model = build_denoiser(info.mode, num_experts=num_experts, **kwargs)
    if device is not None:
        model = model.to(device)

    state = payload.get("model_state_dict", payload)
    # torch.compile prefixes every key when the model was compiled at save.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    model.eval()
    # Marks the model as carrying real weights, so the runner does not
    # label its results as an untrained smoke test.
    model.loaded_checkpoint = True
    info.num_experts = getattr(model, "num_experts", num_experts)
    return model, info
