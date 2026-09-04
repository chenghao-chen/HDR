"""
test_dual_MoE_two_phase.py — Benchmark for the MoE / DualSNR HDR denoisers
============================================================================
Features:
  - Auto-rebuilds the exact architecture from the checkpoint (mode,
    model_kwargs and num_experts are stored by train_A100_MoE_two_phase.py;
    falls back to MODEL_KWARGS below for legacy checkpoints)
  - Full-image or overlapping-patch inference (INFERENCE flag)
  - RGB-domain metrics (model outputs RGB directly; GT demosaiced with GBTF)
    • PSNR-linear, PSNR-µ (µ=5000), SSIM
  - Per-expert PSNR and per-pixel gate usage statistics (any number of experts)
  - Deterministic test noise (handled inside MobileHDRDataset) → results are
    reproducible across runs
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
from hdr_platform import get_accelerator, resolve_dataset_dir


# ─────────────────────────────────────────────────────────────────────────────
# Tee: mirror all print() output to both stdout and a log file simultaneously
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    """
    Replaces sys.stdout so every print() writes to both the terminal and
    a log file.  Call Tee.close() (or use as a context manager) when done.

    Usage:
        sys.stdout = Tee(log_path)
        ...
        sys.stdout.close()
    """
    def __init__(self, path: str):
        self._terminal = sys.__stdout__
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._log = open(path, "w", buffering=1)   # line-buffered

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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mode         = ckpt.get("mode", "dual")
    model_kwargs = ckpt.get("model_kwargs", fallback_kwargs)
    num_experts  = ckpt.get("num_experts", fallback_num_experts)
    print(f"  Checkpoint mode: '{mode}'  num_experts={num_experts}")
    print(f"  Model kwargs:    {model_kwargs}")

    model = build_denoiser(mode, num_experts=num_experts, **model_kwargs).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    # Strip torch.compile prefix if checkpoint was saved while compiled
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()    # benchmark loader: eval mode (enables the [.., 1] clamp)
    return model, mode


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def hdr_tonemap(x, mu=5000):
    """
    µ-law tonemapping: log1p(µ·x) / log1p(µ).
    mu=5000 matches training and the HDR imaging literature standard
    (Kalantari SIGGRAPH 2017). All metrics and visualisations use this.
    """
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
    """
    Structural Similarity — averaged over batch and channels.
    Pure PyTorch, no external dependency.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    C  = pred.shape[1]

    # Gaussian window
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
    [B, 4, h, w] packed Bayer (B, G1, G2, R) -> [B, 1, 2h, 2w] BGGR mosaic.
    PixelShuffle places channel c at cell offset (c//2, c%2):
    B→(0,0), G1→(0,1), G2→(1,0), R→(1,1) — exactly the BGGR layout.
    """
    return F.pixel_shuffle(packed, 2)


def save_jpg(tensor, path, quality=92):
    """Save a [C, H, W] float tensor as JPEG. save_image doesn't support quality kwarg."""
    to_pil_image(tensor.clamp(0, 1).cpu()).save(path, format="JPEG", quality=quality)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def infer_patches(model, noisy_bayer, num_experts, patch_size=256, overlap=32,
                  device=None):
    """
    Runs the model on a single [1, 4, H, W] Bayer image using overlapping patches.
    The model outputs sensor-resolution RGB, so each patch at (H_bayer, W_bayer)
    produces an output at (2*H_bayer, 2*W_bayer). Accumulation is done at sensor res.

    `device` defaults to the input tensor's own device, so the accumulators
    land beside the data instead of on a hardcoded 'cuda'.

    Returns: (blended [1,3,2H,2W],
              expert_outs [1,K,3,2H,2W],
              gates [1,K,2H,2W])
    """
    device = noisy_bayer.device if device is None else device
    # Autocast needs the backend's own name: 'cuda' on Polaris, 'xpu' on
    # Aurora. A literal 'cuda' raises on an Intel GPU.
    acc = get_accelerator(str(device))

    _, _, H, W = noisy_bayer.shape    # Bayer resolution
    stride = patch_size - overlap

    # Pad up to a whole number of strides, but never below patch_size: for
    # H <= overlap the stride arithmetic yields Hp < patch_size, the tile loop
    # `range(0, Hp - patch_size + 1, stride)` is empty, no tile ever runs and
    # every pixel comes out 0/1e-8 = 0 — a correctly shaped, entirely black
    # prediction that the benchmark would happily report a PSNR for.
    pad_h = max(math.ceil((H - overlap) / stride) * stride + overlap, patch_size) - H
    pad_w = max(math.ceil((W - overlap) / stride) * stride + overlap, patch_size) - W
    # Replicate, not reflect: reflect requires the pad to be strictly smaller
    # than the source dimension, so any image at most half the patch size
    # (with PATCH_SIZE=1024, anything under ~512 packed pixels) raised while
    # infer_full handled it fine.
    x_pad = F.pad(noisy_bayer, (0, pad_w, 0, pad_h), mode='replicate')
    _, _, Hp, Wp = x_pad.shape

    # Output arrays at sensor resolution (2× Bayer)
    Hs, Ws = Hp * 2, Wp * 2
    out_patch = patch_size * 2                    # output patch size in sensor pixels
    pred_sum   = torch.zeros((1, 3, Hs, Ws), device=device)
    expert_sum = torch.zeros((1, num_experts, 3, Hs, Ws), device=device)
    gate_sum   = torch.zeros((1, num_experts, Hs, Ws), device=device)
    weight_sum = torch.zeros((1, 1, Hs, Ws), device=device)

    win = torch.ones((1, 1, out_patch, out_patch), device=device)

    snr_full = estimate_local_snr_map(x_pad, window_size=5)

    for yi in range(0, Hp - patch_size + 1, stride):
        for xi in range(0, Wp - patch_size + 1, stride):
            patch    = x_pad[:, :, yi:yi+patch_size, xi:xi+patch_size]
            snr_crop = snr_full[:, :, yi:yi+patch_size, xi:xi+patch_size]

            with acc.autocast(torch.bfloat16):
                pred, experts, gates = model(patch, snr_crop)

            # Output offsets at sensor resolution
            ys, xs = yi * 2, xi * 2
            pred_sum  [:, :,    ys:ys+out_patch, xs:xs+out_patch] += pred.float() * win
            expert_sum[:, :, :, ys:ys+out_patch, xs:xs+out_patch] += experts.float() * win.unsqueeze(1)
            gate_sum  [:, :,    ys:ys+out_patch, xs:xs+out_patch] += gates.float() * win[:, 0]
            weight_sum[:, :,    ys:ys+out_patch, xs:xs+out_patch] += win

    # Every output pixel must have been covered by at least one tile. Without
    # this, an empty or short tile loop yields 0/1e-8 = 0 and the benchmark
    # reports a PSNR for a silently all-black prediction.
    H_out, W_out = H * 2, W * 2                  # crop to original sensor size
    if float(weight_sum[..., :H_out, :W_out].min()) <= 0.0:
        raise RuntimeError(
            f"infer_patches left part of the image uncovered "
            f"(H={H}, W={W}, patch_size={patch_size}, overlap={overlap}); "
            "refusing to return a partly-zero prediction")

    denom  = weight_sum + 1e-8                    # [1, 1, Hs, Ws]
    return (
        pred_sum  [...,    :H_out, :W_out] / denom[...,            :H_out, :W_out],
        expert_sum[..., :, :H_out, :W_out] / denom.unsqueeze(1)[..., :H_out, :W_out],
        gate_sum  [...,    :H_out, :W_out] / denom[...,            :H_out, :W_out],
    )


def infer_full(model, noisy_bayer, device=None):
    """
    Full-resolution inference — no patches, no blending artifacts.

    Input is [1, 4, H, W] packed Bayer (H, W divisible by 8 required).
    Output is sensor-resolution RGB: [1, 3, 2H, 2W] (after cropping padding).
    We reflect-pad the Bayer input to a multiple of 8 if needed, run inference,
    then crop the RGB output back to the true sensor size (2*H_orig, 2*W_orig).

    Returns: (blended [1,3,2H,2W], expert_outs [1,K,3,2H,2W], gates [1,K,2H,2W])
    """
    _, _, H, W = noisy_bayer.shape    # Bayer dimensions

    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    x = (F.pad(noisy_bayer, (0, pad_w, 0, pad_h), mode='reflect')
         if pad_h or pad_w else noisy_bayer)

    snr_map = estimate_local_snr_map(x, window_size=5)

    acc = get_accelerator(str(noisy_bayer.device if device is None else device))
    with acc.autocast(torch.bfloat16):
        pred, experts, gates = model(x, snr_map)

    # Crop back to the unpadded sensor size (2× the original Bayer dims)
    H_out, W_out = H * 2, W * 2
    return (pred[..., :H_out, :W_out],
            experts[..., :H_out, :W_out],
            gates[..., :H_out, :W_out])


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs estimation
# ─────────────────────────────────────────────────────────────────────────────
def estimate_flops(model, patch_size=256, device=None):
    """
    Estimates GFLOPs for one patch_size² patch using torchinfo.
    Call BEFORE torch.compile — torchinfo probes many shapes, which would
    thrash TorchDynamo's compile cache.
    Gracefully skipped if torchinfo is not installed.
    """
    try:
        from torchinfo import summary

        # Input is packed Bayer (patch_size × patch_size); SNR map is also at Bayer res.
        device = next(model.parameters()).device if device is None else device
        dummy_x   = torch.zeros(1, 4, patch_size, patch_size, device=device)
        dummy_snr = torch.zeros(1, 1, patch_size, patch_size, device=device)
        # Output will be RGB at (2*patch_size × 2*patch_size) sensor resolution.

        stats = summary(model, input_data=(dummy_x, dummy_snr),
                        verbose=0, mode='eval')
        gflops = stats.total_mult_adds / 1e9
        params_m = stats.total_params / 1e6
        print(f"  Parameters:             {params_m:.2f} M")
        print(f"  FLOPs (one {patch_size}x{patch_size} patch): {gflops:.2f} GFLOPs")
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
    # ── Device and performance flags ───────────────────────────────────────
    # Picks cuda:0 on Polaris, xpu:0 on Aurora, cpu elsewhere, and enables
    # the reduced-precision matmul path each backend offers (TF32 on the
    # A100's tensor cores; the equivalent fast path on Intel Xe).
    ACC = get_accelerator()
    ACC.enable_fast_matmul()

    # ── Configuration ──────────────────────────────────────────────────────
    # Falls back to this machine's dataset location rather than the Gilbreth
    # path this script was written against, which exists on neither ALCF system.
    DATASET_DIR    = resolve_dataset_dir("Mobile-HDR")
    # Update this to the phase{1,2}_best.pth from your latest training run.
    # Naming: models_p{PHASE}_{mode}_Teacher_MobileHDR_{timestamp}/phase{PHASE}_best.pth
    # CHECKPOINT     = "models_p1_moe_Teacher_MobileHDR_20260605_0848/phase1_best.pth"
    CHECKPOINT     = os.environ.get(
        "HDR_CHECKPOINT",
        "models_p1_moe_Teacher_MobileHDR_20260619_0059/phase1_best.pth")
    # basename(dirname(...)), not split('/')[0]: HDR_CHECKPOINT is naturally
    # given as an absolute path, and split('/')[0] is then the empty string —
    # every absolute-path run would write its log.txt, results.csv and rgb/*.jpg
    # over the previous one.
    OUTPUT_DIR     = os.path.join(
        "test_results",
        os.path.basename(os.path.dirname(os.path.abspath(CHECKPOINT))))
    INFERENCE      = "full"          # "full" | "patches"
    PATCH_SIZE     = 1024
    PATCH_OVERLAP  = PATCH_SIZE // 4
    SAVE_EVERY     = 1
    # mu=5000 matches training (train_A100_MoE_two_phase.py uses mu=5000).
    # Using a different value makes test PSNR-µ incomparable to W&B training curves.
    METRIC_MU      = 5000
    DEVICE         = ACC.device
    # torch.compile: test images vary in size, so full-res inference triggers
    # one (slow) recompile per distinct shape. Enable only for fixed-size or
    # patch-based runs.
    USE_COMPILE    = False

    # Fallback for legacy checkpoints without stored model_kwargs.
    # NOTE: checkpoints trained before SE blocks existed need "se_reduction": None.
    MODEL_KWARGS = {
        "dim":                   32,
        "num_blocks":            [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads":                 [1, 2, 4, 8],
        "se_reduction":          8,
    }
    FALLBACK_NUM_EXPERTS = 3
    # ───────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "rgb"), exist_ok=True)

    # ── Log file — mirrors all print() output to OUTPUT_DIR/log.txt ───────
    log_path   = os.path.join(OUTPUT_DIR, "log.txt")
    sys.stdout = Tee(log_path)
    print(f"Logging to: {log_path}")

    # ── Load model (architecture auto-detected from checkpoint) ────────────
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model, mode_tag = load_model_from_checkpoint(
        CHECKPOINT, MODEL_KWARGS, DEVICE, FALLBACK_NUM_EXPERTS)
    model.eval()
    K = model.num_experts
    print("  Weights loaded.")

    # ── FLOPs (BEFORE compile — torchinfo probing thrashes dynamo cache) ──
    print("\nEstimating model complexity...")
    gflops = estimate_flops(model, patch_size=PATCH_SIZE, device=DEVICE)

    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("  Model compiled with TorchInductor.")
        except Exception as e:
            print(f"  Skipping compile: {e}")

    # ── Differentiable GBTF demosaicing ────────────────────────────────────
    gbtf = DifferentiableGBTF_BGGR().to(DEVICE)
    gbtf.eval()

    # ── Dataset (deterministic noise per index — reproducible benchmarks) ──
    print("\nLoading test dataset...")
    test_dataset = MobileHDRDataset(
        base_dir=DATASET_DIR,
        split="test",
        transform=None      # no augmentation at test time
    )
    # batch_size=1: process one full image at a time
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"  {len(test_dataset)} test samples found.\n")

    # ── Metrics accumulators ───────────────────────────────────────────────
    # The model now outputs sensor-resolution RGB directly, so all metrics are
    # in the RGB domain.  RAW Bayer domain metrics are no longer applicable.
    metrics = {
        "psnr_noisy_rgb_mu": [],   # noisy baseline (after GBTF demosaic)
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

            # ── SNR stats (computed before inference) ──────────────────
            snr_full = estimate_local_snr_map(x, window_size=5)
            pct_low  = (snr_full < 0.5).float().mean().item() * 100.0

            # ── Timed inference ────────────────────────────────────────
            # Sync before and after: kernels are queued asynchronously, so
            # without these the timer measures submission, not execution.
            ACC.synchronize()
            t0 = time.perf_counter()

            if INFERENCE == "patches":
                y_pred, expert_outs, gates = infer_patches(
                    model, x, K, patch_size=PATCH_SIZE,
                    overlap=PATCH_OVERLAP, device=DEVICE)
            else:
                y_pred, expert_outs, gates = infer_full(model, x, device=DEVICE)

            ACC.synchronize()
            elapsed = time.perf_counter() - t0

            # ── Demosaic GT and noisy for comparison (y_pred is already RGB) ──
            with ACC.autocast(torch.bfloat16):
                rgb_gt    = gbtf(packed_bayer_to_mosaic(y.clamp(0, 1).float())).clamp(0, 1).float()
                rgb_noisy = gbtf(packed_bayer_to_mosaic(x.clamp(0, 1).float())).clamp(0, 1).float()

            y_pred_c = y_pred.clamp(0, 1).float()   # [1, 3, 2H, 2W] RGB

            # ── RGB domain metrics ─────────────────────────────────────
            tm_pred           = hdr_tonemap(y_pred_c, METRIC_MU)
            tm_gt             = hdr_tonemap(rgb_gt,   METRIC_MU)
            psnr_rgb_lin      = psnr(y_pred_c, rgb_gt)
            psnr_rgb_mu       = psnr(tm_pred, tm_gt)
            ssim_rgb_lin      = ssim(y_pred_c, rgb_gt)
            ssim_rgb_mu       = ssim(tm_pred, tm_gt)
            psnr_noisy_rgb_mu = psnr(hdr_tonemap(rgb_noisy, METRIC_MU), tm_gt)

            # ── Per-expert PSNR-µ + gate usage ─────────────────────────
            psnr_experts = [
                psnr(hdr_tonemap(expert_outs[:, k].clamp(0, 1).float(), METRIC_MU), tm_gt)
                for k in range(K)
            ]
            gate_usage = gates.float().mean(dim=(0, 2, 3)).tolist()  # [K]

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
                f"{psnr_rgb_lin:.4f}",  f"{psnr_rgb_mu:.4f}",
                f"{ssim_rgb_lin:.4f}",  f"{ssim_rgb_mu:.4f}",
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
                vis_pred  = hdr_tonemap(y_pred_c,  METRIC_MU).clamp(0, 1)
                vis_gt    = hdr_tonemap(rgb_gt,    METRIC_MU).clamp(0, 1)

                # Downsample to half resolution before saving (full-res is ~32 MB per file)
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
    print(f"  ── RGB Domain (direct model output) ──────────────────")
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
    sys.stdout.close()   # flush and restore stdout
