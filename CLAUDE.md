# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install torchinfo  # optional, for FLOPs estimation in test script

# Train (local)
python train_A100_MoE_two_phase.py

# Test / evaluate (local)
python test_dual_MoE_two_phase.py

# Submit to SLURM cluster
sbatch submit_train.sh
sbatch submit_test.sh

# Inspect a checkpoint's metadata
python test.py
```

Training is controlled by top-level flags at the bottom of the `if __name__ == "__main__":` block in `train_A100_MoE_two_phase.py` — edit `PHASE`, `MODE`, `NUM_EXPERTS`, `PHASE1_CHECKPOINT`, and `USE_COMPILE` there directly between runs. Similarly, `test_dual_MoE_two_phase.py` has a configuration block at the top of `__main__` (dataset dir, checkpoint path, inference mode, etc.).

## Architecture

This is a **joint HDR denoising + demosaicing** project. The model takes noisy packed BGGR Bayer sensor data and outputs clean sensor-resolution RGB in a single forward pass — no external demosaicing step needed for the model output.

### Tensor formats

- **Model input**: `(B, 4, H, W)` packed BGGR Bayer in `[0, 1]`, where `H, W` are the *packed* dimensions (= sensor / 2). H and W must each be divisible by 8 (three PixelUnshuffle(2) stages in the encoder).
- **Model output**: `(B, 3, 2H, 2W)` clean RGB at full sensor resolution.
- **SNR map**: `(B, 1, H, W)` computed from the noisy input via `estimate_local_snr_map()` in `HDR_model_hybrid_Teacher.py`. Both train and test import this same function — never reimplement it separately.

### Files

| File | Purpose |
|------|---------|
| `HDR_model_hybrid_Teacher.py` | All model definitions + `build_denoiser()` factory |
| `blocks_Restormer.py` | Restormer attention blocks (MDTA, GDFN, RestormerBlock) used as the bottleneck |
| `DifferentiableGBTF_BGGR.py` | Differentiable GBTF demosaicing; used to generate clean RGB GT from clean Bayer during training and for noisy/GT visualisation at test time |
| `HDR_Mobile_dataset.py` | `MobileHDRDataset` — loads `.pt` Bayer tensors, synthesises Poisson-Gaussian noise, applies crops and augmentation |
| `train_A100_MoE_two_phase.py` | Two-phase training loop with W&B logging, rollback, and checkpoint management |
| `test_dual_MoE_two_phase.py` | Full evaluation: PSNR-linear, PSNR-µ, SSIM, per-expert stats, CSV output, JPEG visuals |
| `test.py` | One-off script to print epoch/loss/PSNR metadata from a checkpoint |

### Models (all in `HDR_model_hybrid_Teacher.py`)

Three modes share the unified forward signature `blended, expert_outs, gates = model(x, snr_map)`:

- **`MoEDenoiser`** (`mode="moe"`) — Primary model. One shared CNN-Transformer trunk feeds `K` lightweight `ExpertHead` decoders. A tiny `NoiseGate` CNN (5-ch input: 4 BGGR + 1 SNR) produces per-pixel softmax routing weights, bilinearly upsampled 2× to sensor resolution before blending expert RGB outputs.
- **`DualSNRDenoiser`** (`mode="dual"`) — Legacy: two full `TransUNet_Teacher_HDR` teachers blended by the SNR map directly.
- **`SingleDenoiser`** (`mode="single"`) — Ablation baseline: one teacher, uniform gate.

`build_denoiser(mode, num_experts, **kwargs)` is the shared factory used by both train and test scripts.

### `TransUNet_Teacher_HDR` (shared backbone / teacher)

CNN U-Net with a Restormer transformer bottleneck:
- **Encoder**: PixelUnshuffle embedding → 2 CNN levels with `ResidualConvBlock` + `HeavyExposhare`
- **Bottleneck**: `RestormerBlock` stack (channel-wise transposed attention — O(C²) not O(HW²))
- **Decoder**: PixelShuffle upsampling with skip connections → two PixelShuffle stages to reach sensor resolution and project to RGB

### Two-phase training

- **Phase 1** (50 epochs): 512×512 packed patches, batch 8, lr=1e-4 with 10-epoch linear warmup → cosine to 1e-6. Full D4 augmentation (H-flip, V-flip, transpose) with paired BGGR channel permutations.
- **Phase 2** (30 epochs): full-resolution images, batch 1, lr=5e-6, cosine to 1e-7. Transpose augmentation disabled (non-square frames). Phase 2 initialises from the best Phase 1 checkpoint.

Loss = L1-µ (µ-law tonemapped, µ=5000) + γ·LPIPS + aux·per-expert gate-weighted L1-µ + balance·load-balance (MoE only). An epoch-level rollback reverts weights if loss spikes > `rollback_mult × last_epoch_loss`.

### Dataset

`.pt` files of shape `(4, H, W)` float32 unnormalised HDR packed BGGR Bayer tensors:
- Train: `{dataset_dir}/train/tensors/**/*.pt`
- Test: `{dataset_dir}/test/tensors/with_gt/*.pt`

Dataset dir on the cluster: `/scratch/gilbreth/chen4848/datasets/Mobile-HDR`

Noise model: Poisson-Gaussian in digital numbers — `var = 14 * signal_DN + U(135, 160)` at 10-bit. The random exposure alpha is biased toward low-light. For train, the random crop is taken **before** noise synthesis (~12× cheaper). For test, noise is seeded per index for reproducibility.

### Checkpoint format

Checkpoints store `mode`, `model_kwargs`, `num_experts`, `epoch`, `phase`, `loss`, and `best_psnr_mu` so the test script can rebuild the exact architecture without manual sync. The test script's `load_model_from_checkpoint()` reads these automatically; `MODEL_KWARGS` and `FALLBACK_NUM_EXPERTS` in the test script are only for legacy checkpoints that predate this metadata.

### Metrics

All metrics are computed in the RGB domain (model output vs. GBTF-demosaiced GT):
- **PSNR-µ** (primary, µ=5000 — Kalantari SIGGRAPH 2017 standard) — must match between train W&B curves and test results
- **PSNR-linear**
- **SSIM**

W&B project: `hdr-dual-moe`
