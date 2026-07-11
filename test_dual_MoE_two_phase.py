"""
test_dual_MoE_two_phase.py — Benchmark for the MoE / DualSNR HDR Bayer denoisers
==================================================================================
The model outputs denoised packed BGGR Bayer (B, 4, H, W).
GBTF demosaicing is applied here to convert to RGB for metrics and visualisation.
This keeps the training loss clean (pure Bayer vs Bayer GT) while evaluating
final quality in the display-ready RGB domain.

Features:
  - Auto-rebuilds the exact architecture from the checkpoint (mode,
    model_kwargs and num_experts are stored by train_A100_MoE_two_phase.py;
    falls back to MODEL_KWARGS below for legacy checkpoints)
  - Full-image or overlapping-patch inference (INFERENCE flag)
  - RGB-domain metrics computed after GBTF demosaicing:
    • PSNR-linear, PSNR-µ (µ=5000, Kalantari 2017), SSIM
  - Per-expert PSNR and per-pixel gate usage statistics
  - Deterministic test noise (handled inside MobileHDRDataset) → reproducible
  - FLOPs estimation (via torchinfo if available, else skipped gracefully)
  - Per-image wall-clock timing
  - Saves side-by-side RGB comparison: Noisy | Denoised | GT
  - CSV results log for easy analysis
"""

import os
import csv
import sys
import time
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from HDR_model_hybrid_Teacher import build_denoiser, estimate_local_snr_map
from HDR_Mobile_dataset import MobileHDRDataset
from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR


# ─────────────────────────────────────────────────────────────────────────────
# Tee: mirror all print() output to both stdout and a log file simultaneously
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    """
    Replaces sys.stdout so every print() writes to both the terminal and
    a log file.  Call Tee.close() (or use as a context manager) when done.
    """
    def __init__(self, path: str):
        self._terminal = sys.__stdout__
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._log = open(path, "w", buffering=1)

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._terminal
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_from_checkpoint(checkpoint_path, fallback_kwargs, device,
                               fallback_num_experts=2):
    """
    Rebuilds the architecture from metadata stored in the checkpoint
    ('mode', 'model_kwargs', 'num_experts'); falls back to the arguments
    given here for legacy checkpoints that predate the metadata.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode         = ckpt.get("mode", "dual")
    model_kwargs = ckpt.get("model_kwargs", fallback_kwargs)
    num_experts  = ckpt.get("num_experts", fallback_num_experts)
    print(f"  Checkpoint mode: '{mode}'  num_experts={num_experts}")
    print(f"  Model kwargs:    {model_kwargs}")

    model = build_denoiser(mode, num_experts=num_experts, **model_kwargs).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mode


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def hdr_tonemap(x, mu=5000):
    """µ-law tonemapping: log1p(µ·x) / log1p(µ). mu=5000 matches training."""
    return torch.log1p(mu * x) / math.log1p(mu)


def psnr(pred, gt, data_range=1.0):
    """Per-image PSNR, returns mean over batch."""
    with torch.no_grad():
        mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
        p = torch.where(mse == 0,
                        torch.tensor(100.0, device=pred.device),
                        10.0 * torch.log10(data_range ** 2 / mse))
        return p.mean().item()


def ssim(pred, gt, window_size=11, data_range=1.0):
    """Structural Similarity — averaged over batch and channels. Pure PyTorch."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    C  = pred.shape[1]

    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(C, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(gt,   kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1  = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    s2  = F.conv2d(gt   * gt,   kernel, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(pred * gt,   kernel, padding=pad, groups=C) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return (num / den).mean().item()


def packed_bayer_to_mosaic(packed):
    """
    [B, 4, h, w] packed BGGR (B=0, G1=1, G2=2, R=3) -> [B, 1, 2h, 2w] mosaic.
    PixelShuffle places channel c at cell offset (c//2, c%2):
    B→(0,0), G1→(0,1), G2→(1,0), R→(1,1) — exactly the BGGR layout.
    """
    return F.pixel_shuffle(packed, 2)


def bayer_to_rgb(packed, gbtf_module):
    """
    Converts packed BGGR [B, 4, H, W] → RGB [B, 3, 2H, 2W] via GBTF.
    Input must be in [0, 1]. Output is clamped to [0, 1].
    """
    mosaic = packed_bayer_to_mosaic(packed.clamp(0, 1).float())
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        rgb = gbtf_module(mosaic)
    return rgb.float().clamp(0, 1)


def save_jpg(tensor, path, quality=92):
    """Save a [C, H, W] float tensor as JPEG."""
    to_pil_image(tensor.clamp(0, 1).cpu()).save(path, format="JPEG", quality=quality)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def infer_patches(model, noisy_bayer, num_experts, patch_size=256, overlap=32,
                  device='cuda'):
    """
    Runs the model on a single [1, 4, H, W] Bayer image using overlapping patches.
    The model outputs denoised packed Bayer at the same H×W resolution.
    Accumulation and blending are done at packed Bayer resolution.

    Returns: (blended [1,4,H,W],
              expert_outs [1,K,4,H,W],
              gates [1,K,H,W])
    """
    _, _, H, W = noisy_bayer.shape
    stride = patch_size - overlap

    pad_h = (math.ceil((H - overlap) / stride) * stride + overlap) - H
    pad_w = (math.ceil((W - overlap) / stride) * stride + overlap) - W
    x_pad = F.pad(noisy_bayer, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = x_pad.shape

    # Accumulate at packed Bayer resolution (same as model I/O)
    pred_sum   = torch.zeros((1, 4, Hp, Wp), device=device)
    expert_sum = torch.zeros((1, num_experts, 4, Hp, Wp), device=device)
    gate_sum   = torch.zeros((1, num_experts, Hp, Wp), device=device)
    weight_sum = torch.zeros((1, 1, Hp, Wp), device=device)
    hann_1d = torch.hann_window(patch_size, periodic=False, device=device)
    win     = (hann_1d.unsqueeze(0) * hann_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1,1,P,P]

    snr_full = estimate_local_snr_map(x_pad, window_size=5)

    for yi in range(0, Hp - patch_size + 1, stride):
        for xi in range(0, Wp - patch_size + 1, stride):
            patch    = x_pad[:, :, yi:yi+patch_size, xi:xi+patch_size]
            snr_crop = snr_full[:, :, yi:yi+patch_size, xi:xi+patch_size]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred, experts, gates = model(patch, snr_crop)

            pred_sum  [:, :,    yi:yi+patch_size, xi:xi+patch_size] += pred.float() * win
            expert_sum[:, :, :, yi:yi+patch_size, xi:xi+patch_size] += experts.float() * win.unsqueeze(1)
            gate_sum  [:, :,    yi:yi+patch_size, xi:xi+patch_size] += gates.float() * win[:, 0]
            weight_sum[:, :,    yi:yi+patch_size, xi:xi+patch_size] += win

    denom = weight_sum + 1e-8
    return (
        pred_sum  [...,    :H, :W] / denom[...,            :H, :W],
        expert_sum[..., :, :H, :W] / denom.unsqueeze(1)[..., :H, :W],
        gate_sum  [...,    :H, :W] / denom[...,            :H, :W],
    )


def infer_full(model, noisy_bayer, device='cuda'):
    """
    Full-resolution inference on a single [1, 4, H, W] packed Bayer image.
    Reflect-pads H and W to the next multiple of 8 if needed (encoder requires
    H, W divisible by 8 due to three PixelUnshuffle(2) stages), then crops
    the output back to the original Bayer dimensions.

    Returns: (blended [1,4,H,W], expert_outs [1,K,4,H,W], gates [1,K,H,W])
    """
    _, _, H, W = noisy_bayer.shape

    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    x = (F.pad(noisy_bayer, (0, pad_w, 0, pad_h), mode='reflect')
         if pad_h or pad_w else noisy_bayer)

    snr_map = estimate_local_snr_map(x, window_size=5)

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred, experts, gates = model(x, snr_map)

    # Crop back to original Bayer dimensions
    return (pred[..., :H, :W],
            experts[..., :H, :W],
            gates[..., :H, :W])


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs estimation
# ─────────────────────────────────────────────────────────────────────────────
def estimate_flops(model, patch_size=256, device='cuda'):
    """
    Estimates GFLOPs for one patch_size² packed Bayer patch using torchinfo.
    Call BEFORE torch.compile — torchinfo probes many shapes, which would
    thrash TorchDynamo's compile cache.
    Gracefully skipped if torchinfo is not installed.
    """
    try:
        from torchinfo import summary

        dummy_x   = torch.zeros(1, 4, patch_size, patch_size, device=device)
        dummy_snr = torch.zeros(1, 1, patch_size, patch_size, device=device)

        stats = summary(model, input_data=(dummy_x, dummy_snr),
                        verbose=0, mode='eval')
        gflops   = stats.total_mult_adds / 1e9
        params_m = stats.total_params / 1e6
        print(f"  Parameters:             {params_m:.2f} M")
        print(f"  FLOPs (one {patch_size}×{patch_size} patch): {gflops:.2f} GFLOPs")
        return gflops
    except ImportError:
        print("  [torchinfo not installed — skipping FLOPs. Run: pip install torchinfo]")
        return None
    except Exception as e:
        print(f"  [FLOPs estimation failed: {e}]")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision('high')

    # ── Configuration ──────────────────────────────────────────────────────
    DATASET_DIR    = "/scratch/gilbreth/chen4848/datasets/Mobile-HDR"
    # Update to the phase{1,2}_best.pth from your latest training run.
    CHECKPOINT     = "models_p1_moe_Teacher_MobileHDR_20260707_2033/phase1_best.pth"
    OUTPUT_DIR     = f"test_results/{CHECKPOINT.split('/')[0]}"
    INFERENCE      = "patches"       # "full" | "patches"
    PATCH_SIZE     = 512             # Bayer-resolution patch size for patch inference
    PATCH_OVERLAP  = PATCH_SIZE // 4
    SAVE_EVERY     = 1
    METRIC_MU      = 5000            # must match training mu
    DEVICE         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    USE_COMPILE    = False

    # Fallback for legacy checkpoints without stored model_kwargs.
    MODEL_KWARGS = {
        "out_channels":          4,
        "dim":                   32,
        "num_blocks":            [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads":                 [1, 2, 4, 8],
        "se_reduction":          8,
    }
    FALLBACK_NUM_EXPERTS = 2
    # ───────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "rgb"), exist_ok=True)

    log_path   = os.path.join(OUTPUT_DIR, "log.txt")
    sys.stdout = Tee(log_path)
    print(f"Logging to: {log_path}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model, mode_tag = load_model_from_checkpoint(
        CHECKPOINT, MODEL_KWARGS, DEVICE, FALLBACK_NUM_EXPERTS)
    model.eval()
    K = model.num_experts
    print("  Weights loaded.")

    print("\nEstimating model complexity...")
    gflops = estimate_flops(model, patch_size=PATCH_SIZE, device=DEVICE)

    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("  Model compiled with TorchInductor.")
        except Exception as e:
            print(f"  Skipping compile: {e}")

    # ── GBTF demosaicing for RGB conversion ────────────────────────────────
    # Model outputs packed BGGR Bayer. GBTF converts to RGB for metrics/vis.
    gbtf = DifferentiableGBTF_BGGR().to(DEVICE)
    gbtf.eval()
    for p in gbtf.parameters():
        p.requires_grad_(False)

    # ── Dataset ────────────────────────────────────────────────────────────
    print("\nLoading test dataset...")
    test_dataset = MobileHDRDataset(
        base_dir=DATASET_DIR,
        split="test",
        transform=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"  {len(test_dataset)} test samples found.\n")

    # ── Metrics accumulators ───────────────────────────────────────────────
    metrics = {
        "psnr_noisy_rgb_mu": [],
        "psnr_rgb_linear":   [],
        "psnr_rgb_mu":       [],
        "ssim_rgb_linear":   [],
        "ssim_rgb_mu":       [],
        "pct_low_snr_pixels":[],
        "time_sec":          [],
    }
    expert_psnr_acc = [[] for _ in range(K)]
    gate_usage_acc  = [[] for _ in range(K)]

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "sample_idx",
        "psnr_noisy_rgb_mu",
        "psnr_rgb_linear", "psnr_rgb_mu", "ssim_rgb_linear", "ssim_rgb_mu",
        "pct_low_snr_pixels",
        *[f"psnr_expert{k}_mu" for k in range(K)],
        *[f"gate{k}_usage" for k in range(K)],
        "time_sec",
    ])

    print("=" * 80)
    print(f"{'#':>4}  {'PSNR-rgb-µ(noisy)':>18}  {'PSNR-rgb-lin':>12}  "
          f"{'PSNR-rgb-µ':>10}  {'gate usage':>18}  {'Time(s)':>8}")
    print("=" * 80)

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            x = sample["x"].to(DEVICE, non_blocking=True)   # [1, 4, H, W] noisy Bayer
            y = sample["y"].to(DEVICE, non_blocking=True)   # [1, 4, H, W] clean GT Bayer

            # ── SNR stats ──────────────────────────────────────────────
            snr_full = estimate_local_snr_map(x, window_size=5)
            pct_low  = (snr_full < 0.5).float().mean().item() * 100.0

            # ── Timed inference — output is denoised packed Bayer ──────
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if INFERENCE == "patches":
                y_pred, expert_outs, gates = infer_patches(
                    model, x, K, patch_size=PATCH_SIZE,
                    overlap=PATCH_OVERLAP, device=DEVICE)
            else:
                y_pred, expert_outs, gates = infer_full(model, x, device=DEVICE)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # ── Convert Bayer to RGB via GBTF for metrics / visuals ────
            # y_pred: [1, 4, H, W] denoised Bayer → rgb_pred: [1, 3, 2H, 2W]
            rgb_pred  = bayer_to_rgb(y_pred, gbtf)
            rgb_gt    = bayer_to_rgb(y,      gbtf)
            rgb_noisy = bayer_to_rgb(x,      gbtf)

            # ── RGB domain metrics ─────────────────────────────────────
            tm_pred           = hdr_tonemap(rgb_pred,  METRIC_MU)
            tm_gt             = hdr_tonemap(rgb_gt,    METRIC_MU)
            psnr_rgb_lin      = psnr(rgb_pred, rgb_gt)
            psnr_rgb_mu       = psnr(tm_pred, tm_gt)
            ssim_rgb_lin      = ssim(rgb_pred, rgb_gt)
            ssim_rgb_mu       = ssim(tm_pred, tm_gt)
            psnr_noisy_rgb_mu = psnr(hdr_tonemap(rgb_noisy, METRIC_MU), tm_gt)

            # ── Per-expert PSNR-µ (each expert's Bayer → RGB → metric) ─
            psnr_experts = [
                psnr(hdr_tonemap(bayer_to_rgb(expert_outs[:, k], gbtf), METRIC_MU), tm_gt)
                for k in range(K)
            ]
            gate_usage = gates.float().mean(dim=(0, 2, 3)).tolist()   # [K]

            # ── Accumulate ─────────────────────────────────────────────
            metrics["psnr_noisy_rgb_mu"].append(psnr_noisy_rgb_mu)
            metrics["psnr_rgb_linear"].append(psnr_rgb_lin)
            metrics["psnr_rgb_mu"].append(psnr_rgb_mu)
            metrics["ssim_rgb_linear"].append(ssim_rgb_lin)
            metrics["ssim_rgb_mu"].append(ssim_rgb_mu)
            metrics["pct_low_snr_pixels"].append(pct_low)
            metrics["time_sec"].append(elapsed)
            for k in range(K):
                expert_psnr_acc[k].append(psnr_experts[k])
                gate_usage_acc[k].append(gate_usage[k])

            csv_writer.writerow([
                i,
                f"{psnr_noisy_rgb_mu:.4f}",
                f"{psnr_rgb_lin:.4f}", f"{psnr_rgb_mu:.4f}",
                f"{ssim_rgb_lin:.4f}", f"{ssim_rgb_mu:.4f}",
                f"{pct_low:.1f}",
                *[f"{v:.4f}" for v in psnr_experts],
                *[f"{v:.4f}" for v in gate_usage],
                f"{elapsed:.3f}",
            ])
            csv_file.flush()

            usage_str = "/".join(f"{v:.2f}" for v in gate_usage)
            print(f"{i+1:>4}  {psnr_noisy_rgb_mu:>18.2f}  {psnr_rgb_lin:>12.2f}  "
                  f"{psnr_rgb_mu:>10.2f}  {usage_str:>18}  {elapsed:>8.3f}s")

            # ── Save RGB visuals ───────────────────────────────────────
            if i % SAVE_EVERY == 0:
                vis_noisy = hdr_tonemap(rgb_noisy, METRIC_MU).clamp(0, 1)
                vis_pred  = hdr_tonemap(rgb_pred,  METRIC_MU).clamp(0, 1)
                vis_gt    = hdr_tonemap(rgb_gt,    METRIC_MU).clamp(0, 1)

                # Downsample to half resolution before saving
                vis_noisy = F.interpolate(vis_noisy, scale_factor=0.5, mode='bilinear', align_corners=False)
                vis_pred  = F.interpolate(vis_pred,  scale_factor=0.5, mode='bilinear', align_corners=False)
                vis_gt    = F.interpolate(vis_gt,    scale_factor=0.5, mode='bilinear', align_corners=False)

                sep = torch.ones(1, 3, vis_gt.shape[2], 2, device=DEVICE)
                comparison = torch.cat([vis_noisy, sep, vis_pred, sep, vis_gt], dim=3)
                save_jpg(comparison[0], os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_compare.jpg"))
                save_jpg(vis_pred[0],   os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_denoised.jpg"))
                save_jpg(vis_gt[0],     os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_gt.jpg"))
                save_jpg(vis_noisy[0],  os.path.join(OUTPUT_DIR, "rgb", f"sample_{i:04d}_noisy.jpg"))

    csv_file.close()

    # ── Summary ────────────────────────────────────────────────────────────
    def avg(lst): return sum(lst) / len(lst)

    mu_label = f"µ={METRIC_MU}"

    print("\n" + "=" * 65)
    print(f"     FINAL RESULTS  [{mode_tag.upper()} mode | K={K} | mu={METRIC_MU}]")
    print("=" * 65)
    print(f"  Samples evaluated:      {len(test_dataset)}")
    print(f"  Total time:             {sum(metrics['time_sec']):.2f}s")
    print(f"  Avg time / image:       {avg(metrics['time_sec']):.3f}s")
    if gflops:
        print(f"  GFLOPs / patch:         {gflops:.2f}")
    print()
    print(f"  ── Noisy Baseline ────────────────────────────────────")
    print(f"  PSNR-{mu_label} (RGB):    {avg(metrics['psnr_noisy_rgb_mu']):.4f} dB")
    print()
    print(f"  ── RGB Domain (Bayer denoised → GBTF → RGB) ─────────")
    print(f"  PSNR-linear:            {avg(metrics['psnr_rgb_linear']):.4f} dB")
    delta_rgb = avg(metrics['psnr_rgb_mu']) - avg(metrics['psnr_noisy_rgb_mu'])
    print(f"  PSNR-{mu_label}:          {avg(metrics['psnr_rgb_mu']):.4f} dB  (delta +{delta_rgb:.2f} dB)")
    print(f"  SSIM-linear:            {avg(metrics['ssim_rgb_linear']):.4f}")
    print(f"  SSIM-{mu_label}:          {avg(metrics['ssim_rgb_mu']):.4f}")
    print()
    print(f"  ── Expert Routing ────────────────────────────────────")
    print(f"  Avg % pixels with SNR < 0.5:  {avg(metrics['pct_low_snr_pixels']):.1f}%")
    for k in range(K):
        print(f"  Expert {k}:  gate usage {avg(gate_usage_acc[k])*100:5.1f}%   "
              f"PSNR-{mu_label} {avg(expert_psnr_acc[k]):.4f} dB")
    print("=" * 65)
    print(f"\n  Full per-sample results: {csv_path}")
    print(f"  RGB visuals saved to:    {os.path.join(OUTPUT_DIR, 'rgb')}/")
    print(f"  Log saved to:            {log_path}")
    sys.stdout.close()
