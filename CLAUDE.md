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

This is an **HDR Bayer denoising** project. The model takes noisy packed BGGR Bayer sensor data and outputs clean denoised packed Bayer at the same resolution. GBTF demosaicing is applied at test time only to obtain RGB for visualisation and evaluation.

### Tensor formats

- **Model input**: `(B, 4, H, W)` noisy packed BGGR Bayer in `[0, 1]`, where `H, W` are the *packed* dimensions (= sensor / 2). Channel order: B=0, G1=1, G2=2, R=3. H and W must each be divisible by 8 (three PixelUnshuffle(2) stages in the encoder).
- **Model output**: `(B, 4, H, W)` denoised packed BGGR Bayer at the same resolution as input. No demosaicing — GT is clean Bayer, loss is Bayer vs Bayer.
- **SNR map**: `(B, 1, H, W)` computed from the noisy input via `estimate_local_snr_map()` in `HDR_model_hybrid_Teacher.py`. Both train and test import this same function — never reimplement it separately.
- **RGB for viewing**: at test time, `F.pixel_shuffle(y, 2)` converts packed Bayer `(B,4,H,W)` → mosaic `(B,1,2H,2W)`, then `DifferentiableGBTF_BGGR` converts to `(B,3,2H,2W)` RGB.

### Files

| File | Purpose |
|------|---------|
| `HDR_model_hybrid_Teacher.py` | All model definitions + `build_denoiser()` factory |
| `blocks_Restormer.py` | Restormer attention blocks (MDTA, GDFN, RestormerBlock) used as the bottleneck |
| `DifferentiableGBTF_BGGR.py` | GBTF demosaicing — used only at test time to convert denoised Bayer to RGB for metrics and visualisation |
| `HDR_Mobile_dataset.py` | `MobileHDRDataset` — loads `.pt` Bayer tensors, synthesises Poisson-Gaussian noise, applies crops and augmentation |
| `train_A100_MoE_two_phase.py` | Two-phase training loop with W&B logging, rollback, and checkpoint management |
| `test_dual_MoE_two_phase.py` | Full evaluation: PSNR-linear, PSNR-µ, SSIM, per-expert stats, CSV output, JPEG visuals |
| `test.py` | One-off script to print epoch/loss/PSNR metadata from a checkpoint |

### Models (all in `HDR_model_hybrid_Teacher.py`)

Three modes share the unified forward signature `blended, expert_outs, gates = model(x, snr_map)`:

- **`MoEDenoiser`** (`mode="moe"`) — Primary model. One shared CNN-Transformer trunk feeds `K` lightweight `ExpertHead` decoders. A tiny `NoiseGate` CNN (5-ch input: 4 BGGR + 1 SNR) produces per-pixel softmax routing weights over K experts. Gate and expert outputs are both at packed Bayer resolution H×W (no upsampling — clean separation from the old JDD approach).
- **`DualSNRDenoiser`** (`mode="dual"`) — Legacy: two full `TransUNet_Teacher_HDR` teachers blended by the SNR map directly.
- **`SingleDenoiser`** (`mode="single"`) — Ablation baseline: one teacher, uniform gate.

`build_denoiser(mode, num_experts, **kwargs)` is the shared factory. Default `out_channels=4` (packed Bayer). Both train and test read `model_kwargs` from the checkpoint — no manual sync needed.

### `TransUNet_Teacher_HDR` (shared backbone / teacher)

CNN U-Net with a Restormer transformer bottleneck:
- **Encoder**: PixelUnshuffle(2) embedding (4-ch packed → 16-ch at H/2) → 2 CNN levels with `ResidualConvBlock` + `HeavyExposhare` + PixelUnshuffle downsampling
- **Bottleneck**: `RestormerBlock` stack (channel-wise transposed attention — O(C²) not O(HW²)) at H/8
- **Decoder**: PixelShuffle upsampling with skip connections, back to H/2 → H
- **Output head**: 3×3 conv projecting `dim//2` channels → 4 packed Bayer channels at H×W

### `ExpertHead` (lightweight MoE decoder)

Takes trunk features at H/2, applies ResidualConvBlocks, one PixelShuffle(2) to reach H, then a zero-initialised 1×1 conv to `out_channels=4` Bayer. Zero init means training starts from near-zero predictions, giving stable early gradients.

### Two-phase training

- **Phase 1** (50 epochs, smoke-test schedule): 512×512 packed patches, batch 8, lr=1e-4 with 3-epoch linear warmup → cosine to 3e-5. Full D4 augmentation (H-flip, V-flip, transpose) with paired BGGR channel permutations. LR stays in active range through all 50 epochs.
- **Phase 2** (30 epochs): full-resolution images, batch 1, lr=5e-6, cosine to 1e-7. Transpose augmentation disabled (non-square frames). Phase 2 initialises from the best Phase 1 checkpoint.

Loss = L1-µ (µ-law tonemapped, µ=5000, Bayer vs Bayer GT) + γ·LPIPS (pseudo-RGB from Bayer: R=ch3, G=avg(ch1,ch2), B=ch0, no GBTF needed) + aux·per-expert gate-weighted L1-µ + balance·load-balance (MoE only). An epoch-level rollback reverts weights if loss spikes > `rollback_mult × last_epoch_loss`.

### Dataset

`.pt` files of shape `(4, H, W)` float32 unnormalised HDR packed BGGR Bayer tensors:
- Train: `{dataset_dir}/train/tensors/**/*.pt`
- Test: `{dataset_dir}/test/tensors/with_gt/*.pt`

Dataset dir on the cluster: `/scratch/gilbreth/chen4848/datasets/Mobile-HDR`

Noise model: Poisson-Gaussian in digital numbers — `var = 14 * signal_DN + U(135, 160)` at 10-bit. The random exposure alpha is biased toward low-light. For train, the random crop is taken **before** noise synthesis (~12× cheaper). For test, noise is seeded per index for reproducibility.

### Checkpoint format

Checkpoints store `mode`, `model_kwargs` (including `out_channels`), `num_experts`, `epoch`, `phase`, `loss`, and `best_psnr_mu` so the test script can rebuild the exact architecture without manual sync. The test script's `load_model_from_checkpoint()` reads these automatically; `MODEL_KWARGS` and `FALLBACK_NUM_EXPERTS` in the test script are only for legacy checkpoints that predate this metadata.

### Metrics

**Training**: PSNR-µ and PSNR-linear reported in Bayer domain (direct comparison to Bayer GT — what the loss optimises).

**Test**: metrics computed in RGB domain after GBTF demosaicing (denoised Bayer → GBTF → RGB, compared to clean Bayer GT → GBTF → RGB):
- **PSNR-µ** (primary, µ=5000 — Kalantari SIGGRAPH 2017 standard)
- **PSNR-linear**
- **SSIM**

Note: training PSNR-µ (Bayer domain) and test PSNR-µ (RGB domain after GBTF) are in different domains and are NOT directly comparable numerically. This is expected and correct — training optimises the Bayer signal, test measures end-to-end display quality.

W&B project: `hdr-dual-moe`
