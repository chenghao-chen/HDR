"""
Shared factories for the HDR test suite.

Everything here is CPU-only and tiny — the point is to exercise shapes,
invariants and control flow, not to train anything.
"""

import os

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Tensor factories
# ─────────────────────────────────────────────────────────────────────────────

def packed_bayer(batch=1, h=32, w=32, seed=0, low=0.0, high=1.0):
    """
    Random packed BGGR tensor [B, 4, h, w] in [low, high].
    h, w are PACKED dimensions; the sensor image is 2h x 2w.
    Model encoders need h, w divisible by 8.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((batch, 4, h, w), generator=g)
    return x * (high - low) + low


def packed_bayer_3d(h=32, w=32, seed=0):
    """Unbatched [4, h, w] packed BGGR — the shape the Dataset returns."""
    return packed_bayer(1, h, w, seed)[0]


def mosaic_from_packed(packed):
    """
    [B, 4, h, w] packed BGGR -> [B, 1, 2h, 2w] mosaic.
    PixelShuffle puts channel c at intra-cell offset (c//2, c%2):
    B->(0,0)  G1->(0,1)  G2->(1,0)  R->(1,1) — the BGGR layout.
    """
    return F.pixel_shuffle(packed, 2)


def packed_from_mosaic(mosaic):
    """Inverse of mosaic_from_packed: [B, 1, 2h, 2w] -> [B, 4, h, w]."""
    return F.pixel_unshuffle(mosaic, 2)


def constant_mosaic(batch=1, h=32, w=32, value=0.5):
    """Packed tensor whose sensor mosaic is a uniform `value` everywhere."""
    return torch.full((batch, 4, h, w), float(value))


def snr_map_for(x):
    """Convenience: the canonical SNR map the models expect for input x."""
    from HDR_model_hybrid_Teacher import estimate_local_snr_map
    return estimate_local_snr_map(x, window_size=5)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic on-disk dataset
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(root, n_train=3, n_test=2, h=64, w=64, seed=0,
                 train_subdirs=("static", "dynamic")):
    """
    Writes a MobileHDRDataset-compatible tree under `root`:

        root/train/tensors/<subdir>/*.pt      (4, h, w) float32
        root/test/tensors/with_gt/*.pt        (4, h, w) float32

    Values span [0, 1] with a deterministic gradient + noise so per-file
    min/max ranges are non-degenerate. Returns the root path as a str.
    """
    root = str(root)
    g = torch.Generator().manual_seed(seed)

    def _write(path, idx):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Deterministic content with a real dynamic range: a spatial ramp
        # plus a small per-file offset, so min/max normalisation is meaningful.
        # The offset is kept under 0.1 (idx % 10) — a larger one would push the
        # whole tensor past 1.0 and clamp it to a constant, which collapses
        # add_photon_noise into its zero-range guard and makes every
        # value-level assertion against the fixture vacuous.
        ramp = torch.linspace(0, 1, h).view(1, h, 1).expand(4, h, w).clone()
        jitter = torch.rand((4, h, w), generator=g) * 0.1
        t = (ramp + jitter + (idx % 10) * 0.01).clamp(0, 1).contiguous().float()
        torch.save(t, path)

    for i in range(n_train):
        sub = train_subdirs[i % len(train_subdirs)]
        _write(os.path.join(root, "train", "tensors", sub, f"train_{i:03d}.pt"), i)

    for i in range(n_test):
        _write(os.path.join(root, "test", "tensors", "with_gt", f"test_{i:03d}.pt"), 100 + i)

    return root


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint factory
# ─────────────────────────────────────────────────────────────────────────────

def make_checkpoint(path, model, mode="moe", num_experts=2, model_kwargs=None,
                    epoch=1, phase=1, loss=0.123, best_psnr_mu=25.0):
    """
    Writes a checkpoint in exactly the format train_A100_MoE_two_phase.py saves,
    so test_dual_MoE_two_phase.load_model_from_checkpoint can consume it.
    """
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "phase": phase,
        "mode": mode,
        "num_experts": num_experts,
        "model_kwargs": model_kwargs or {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "loss": loss,
        "best_psnr_mu": best_psnr_mu,
    }
    torch.save(ckpt, str(path))
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Assertions
# ─────────────────────────────────────────────────────────────────────────────

def assert_finite(t, name="tensor"):
    assert torch.isfinite(t).all(), f"{name} contains NaN or Inf"


def assert_in_range(t, lo=0.0, hi=1.0, name="tensor", atol=1e-5):
    assert float(t.min()) >= lo - atol, f"{name} min {float(t.min())} < {lo}"
    assert float(t.max()) <= hi + atol, f"{name} max {float(t.max())} > {hi}"


def assert_shape(t, expected, name="tensor"):
    assert tuple(t.shape) == tuple(expected), \
        f"{name} shape {tuple(t.shape)} != expected {tuple(expected)}"
