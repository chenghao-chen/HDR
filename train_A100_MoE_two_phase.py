"""
train_A100_MoE_two_phase.py — Two-phase HDR MoE Bayer denoiser training
========================================================================

Task: Bayer denoising.
  Input:  (B, 4, H, W) noisy packed BGGR Bayer in [0, 1]
  Output: (B, 4, H, W) denoised packed BGGR Bayer in [0, 1]
  GT:     clean packed BGGR Bayer — direct, exact supervision, no GBTF.

Phase 1 — Patch training  (50 epochs, 512×512 crops, batch 8)
    The model learns all noise statistics quickly with diverse patch combinations.
    Larger effective batch size → stable gradients → can use higher LR.
    Bottleneck Restormer sees 64×64 = 4096 tokens — enough global context.
    The random crop is taken inside the dataset BEFORE noise synthesis
    (≈12× less dataloader CPU than noising the full frame and cropping after).

Phase 2 — Full-res fine-tune  (30 epochs, full frames, batch 1)
    Closes the train/test distribution gap: patch statistics ≠ full-image
    statistics (boundary effects, global exposure, structured noise).
    Very small LR — adapt, don't relearn.

Loss
────
  L = L1-µ(blended, y_bayer)
    + γ · LPIPS(pseudo_RGB(blended), pseudo_RGB(y_bayer))
    + aux · Σ_k gate_k-weighted L1-µ(expert_k, y_bayer)
    + bal · load-balance                               (MoE only)

  LPIPS uses pseudo-RGB from packed Bayer:
    R = ch[3],  G = 0.5*(ch[1]+ch[2]),  B = ch[0]
  This is computed at Bayer resolution — no GBTF needed during training.
  GBTF is applied at test time only (test_dual_MoE_two_phase.py).

Model modes (MODE flag)
───────────────────────
  "moe"    MoEDenoiser — ONE shared trunk + K lightweight expert heads with
           a learned per-pixel gate conditioned on (noisy input, SNR map).
  "dual"   Legacy DualSNRDenoiser — two full teachers blended by the SNR map.
  "single" One teacher, no routing (ablation).

All modes share one forward signature:
    blended, expert_outs, gates = model(x, snr_map)

D4 augmentation for BGGR packed Bayer
──────────────────────────────────────
Each 2×2 Bayer cell holds  B(0,0) G1(0,1) G2(1,0) R(1,1)  → packed channels [0,1,2,3].
Every spatial symmetry must be paired with a channel permutation that restores
the BGGR channel-spatial correspondence:

  H-flip     flip cols   → even col ↔ odd col  → B↔G1, G2↔R    permute [1,0,3,2]
  V-flip     flip rows   → even row ↔ odd row  → B↔G2, G1↔R    permute [2,3,0,1]
  180° rot   both flips  → compose above two   → B↔R,  G1↔G2   permute [3,2,1,0]
  Transpose  swap H,W    → (r,c)→(c,r)         → G1↔G2          permute [0,2,1,3]

Applying H-flip, V-flip, and Transpose independently at 50% each gives all 8
D4 symmetries with equal probability (Phase 1 — square patches only).
Full-res images are non-square so Transpose is skipped (Phase 2).
"""

import os
import math
import time
import lpips
from datetime import datetime

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import wandb

from HDR_model_hybrid_Teacher import build_denoiser, estimate_local_snr_map
from HDR_Mobile_dataset import MobileHDRDataset

os.environ["WANDB_CACHE_DIR"] = "/scratch/gilbreth/chen4848/wandb_cache"
os.environ["WANDB_DATA_DIR"]  = "/scratch/gilbreth/chen4848/wandb_data"
os.environ["WANDB_DIR"]       = "/scratch/gilbreth/chen4848/projects/HDR-denoise"


##########################################################################
## Logging utilities
##########################################################################

def print_running_loss(running_loss, running_psnr_mu, i, print_every=20):
    if i % print_every == (print_every - 1):
        psnr_str = "  PSNR-µ = %.2f" % (running_psnr_mu / (i + 1)) if running_psnr_mu else ""
        print("\r  step %5d  loss = %10.4f%s" % (
            i + 1, running_loss / (i + 1), psnr_str), end="")

def print_epoch_loss(epoch_num, avg_loss, avg_psnr_mu, avg_psnr, phase, time_spent=None):
    ts  = "?" if time_spent is None else "%.1fs" % time_spent
    msg = ("\r[P%d] Epoch %3d  loss=%10.4f  PSNR-µ=%6.2f  PSNR=%6.2f  (%s)"
           % (phase, epoch_num, avg_loss, avg_psnr_mu, avg_psnr, ts))
    print(msg)
    return msg

def write_log(filename, s, newfile=False, end="\n"):
    if newfile:
        open(filename, "w").close()
    with open(filename, "a") as f:
        f.write("%s%s" % (s, end))

def create_folder(directory):
    os.makedirs(directory, exist_ok=True)


##########################################################################
## GPU / tensor utilities
##########################################################################

def hdr_tonemap(x, mu=5000):
    """
    µ-law tonemapping: log1p(µ·x) / log1p(µ).
    Maps [0, 1] linear HDR → [0, 1] perceptually-uniform space.
    mu=5000 is the HDR imaging literature standard (Kalantari SIGGRAPH 2017).
    """
    return torch.log1p(mu * x) / math.log1p(mu)


def batch_psnr_gpu(img, gt, data_range=1.0):
    """Linear PSNR averaged over the batch, computed entirely on GPU."""
    with torch.no_grad():
        mse = torch.mean((img.detach() - gt.detach()) ** 2, dim=[1, 2, 3])
        p   = torch.where(mse == 0,
                          torch.tensor(100.0, device=img.device),
                          10.0 * torch.log10(data_range ** 2 / mse))
        return p.mean().item()


def bayer_to_pseudo_rgb(t):
    """
    Packed BGGR [B, 4, H, W] -> pseudo-RGB [B, 3, H, W] at Bayer resolution.
    R = ch[3], G = avg(ch[1], ch[2]), B = ch[0].
    Used for LPIPS (VGG needs 3-channel input) without requiring GBTF.
    """
    return torch.stack([t[:, 3], 0.5 * (t[:, 1] + t[:, 2]), t[:, 0]], dim=1)


def collate_xy(batch):
    """
    Phase 1 collate: stacks only the tensors the loop uses. The dataset
    also returns a duplicate 'xm' alias (kept for older scripts) — stacking
    and pinning it would waste CPU and host memory every batch.
    """
    return {
        "x": torch.stack([s["x"] for s in batch]),
        "y": torch.stack([s["y"] for s in batch]),
    }


def collate_pad_to_max(batch):
    """
    Pads variable-size full-resolution images to the largest H×W in the
    batch (rounded up to a multiple of 8) so PyTorch can stack them.
    Stores original sizes in orig_h / orig_w (in packed Bayer dimensions)
    so padded pixels can be excluded from the loss via valid_mask.
    Used only in Phase 2 (full-resolution) where images differ in size.
    """
    max_h = ((max(s["x"].shape[1] for s in batch) + 7) // 8) * 8
    max_w = ((max(s["x"].shape[2] for s in batch) + 7) // 8) * 8

    def pad(t):
        _, h, w = t.shape
        return F.pad(t, (0, max_w - w, 0, max_h - h), mode="reflect")

    return {
        "x":      torch.stack([pad(s["x"]) for s in batch]),
        "y":      torch.stack([pad(s["y"]) for s in batch]),
        "orig_h": torch.tensor([s["x"].shape[1] for s in batch]),  # Bayer H
        "orig_w": torch.tensor([s["x"].shape[2] for s in batch]),  # Bayer W
    }


def build_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """
    Linear warmup for warmup_epochs, then cosine annealing to eta_min.
    If warmup_epochs == 0 returns a plain CosineAnnealingLR.
    """
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=eta_min)


##########################################################################
## Augmentation transforms
##########################################################################

def make_d4_transform(allow_transpose: bool):
    """
    Random D4 augmentation with BGGR channel permutations (see module
    docstring). The random crop is done inside the dataset (crop_size),
    so this only handles flips / transpose.

      H-flip    → permute [1, 0, 3, 2]   (B↔G1, G2↔R)
      V-flip    → permute [2, 3, 0, 1]   (B↔G2, G1↔R)
      Transpose → permute [0, 2, 1, 3]   (G1↔G2)  — square patches only,
                  so Phase 2 (non-square full frames) sets allow_transpose=False.
    """
    def transform(noisy, clean):
        if random.random() > 0.5:
            noisy = TF.hflip(noisy)[[1, 0, 3, 2]]
            clean = TF.hflip(clean)[[1, 0, 3, 2]]

        if random.random() > 0.5:
            noisy = TF.vflip(noisy)[[2, 3, 0, 1]]
            clean = TF.vflip(clean)[[2, 3, 0, 1]]

        if allow_transpose and random.random() > 0.5:
            noisy = noisy.permute(0, 2, 1)[[0, 2, 1, 3]].contiguous()
            clean = clean.permute(0, 2, 1)[[0, 2, 1, 3]].contiguous()

        return noisy, clean
    return transform


##########################################################################
## Main
##########################################################################

if __name__ == "__main__":

    # ── A100 matmul / TF32 optimisations ──────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Reproducibility ───────────────────────────────────────────────
    seed = 21
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    def seed_worker(worker_id):
        """Distinct but deterministic seed per DataLoader worker."""
        random.seed(torch.initial_seed() % 2**32)
    worker_gen = torch.Generator()
    worker_gen.manual_seed(seed)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TOP-LEVEL FLAGS  — the only lines you change between runs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PHASE       = 1        # 1 = patch training  |  2 = full-res fine-tune
    MODE        = "moe"    # "moe" | "dual" | "single"
    NUM_EXPERTS = 2        # MoE only: experts across noise levels
    USE_COMPILE = False    # torch.compile the model (A100 speedup; needs
                           # stable torch+inductor on the cluster)

    # Required for Phase 2: path to the best Phase 1 checkpoint.
    # Phase 2 loads this if no Phase 2 checkpoint exists yet.
    PHASE1_CHECKPOINT = (
        "models_p1_moe_Teacher_MobileHDR_YYYYMMDD_HHMM/phase1_best.pth"
    )
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Paths ─────────────────────────────────────────────────────────
    timestamp_str    = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag         = MODE
    save_folder      = (f"models_p{PHASE}_{mode_tag}_Teacher_"f"MobileHDR_{timestamp_str}/")
    dataset_dir      = "/scratch/gilbreth/chen4848/datasets/Mobile-HDR"
    create_folder(save_folder)

    # Checkpoint paths inside this run's save folder
    save_path      = save_folder + "latest.pth"
    best_save_path = save_folder + (f"phase{PHASE}_best.pth")
    last_path      = None          # tracks the most recently written checkpoint

    # ── Phase-specific hyperparameters ────────────────────────────────
    if PHASE == 1:
        # ── Patch training (smoke-test schedule) ─────────────────────
        # 512×512 packed patches (= 1024×1024 sensor crop).
        # Bottleneck Restormer sees 64×64 = 4096 tokens — sufficient for
        # global noise statistics without processing the entire sensor.
        # LR schedule: short 3-epoch warmup → cosine to 3e-5, keeping
        # active learning through all 50 epochs (eta_min is ~1/3 of peak).
        PATCH_SIZE     = 512
        batch_sz       = 8
        num_patch      = 8    # 8 virtual repeats per image per epoch
        lr             = 1e-4
        warmup_epochs  = 3     # short ramp; get to peak LR by epoch 3
        num_epochs     = 200
        eta_min        = 3e-5  # LR decays 1e-4→3e-5; stays active all 50 epochs
        gamma          = 0.1   # perceptual weight
        grad_clip      = 1.0
        rollback_mult  = 3.0   # tighter than Phase 2; batch 8 has low variance
        LPIPS_CROP     = 256   # Bayer-resolution crop fed to LPIPS (pseudo-RGB)

    else:
        # ── Full-resolution fine-tune ─────────────────────────────────
        PATCH_SIZE     = None  # full resolution
        batch_sz       = 1
        num_patch      = 1
        lr             = 5e-6
        warmup_epochs  = 0
        num_epochs     = 30
        eta_min        = 1e-7
        gamma          = 0.05  # near-zero perceptual; focus on pixel fidelity
        grad_clip      = 0.5   # tighter clip for fine-tune stability
        rollback_mult  = 5.0
        LPIPS_CROP     = 256

    # Shared loss weights
    mu             = 5000   # µ-law tonemapping constant (Kalantari SIGGRAPH 2017)
    aux_weight     = 0.5  if MODE in ("moe", "dual") else 0.0
    balance_weight = 0.01 if MODE == "moe" else 0.0

    start_epoch = 0
    best_psnr_mu = 0.0
    last_epoch_psnr_mu, last_epoch_loss = 0.0, 1e6

    # out_channels=4: model outputs packed BGGR Bayer (same as input format)
    model_kwargs = {
        "out_channels":          4,
        "dim":                   32,
        "num_blocks":            [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads":                 [1, 2, 4, 8],
        "se_reduction":          8,
    }

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project="hdr-dual-moe",
        name=f"run_p{PHASE}_{timestamp_str}_{mode_tag}",
        config={
            "phase":            PHASE,
            "mode":             mode_tag,
            "num_experts":      NUM_EXPERTS,
            "patch_size":       PATCH_SIZE,
            "batch_size":       batch_sz,
            "num_patch":        num_patch,
            "lr":               lr,
            "warmup_epochs":    warmup_epochs,
            "num_epochs":       num_epochs,
            "eta_min":          eta_min,
            "mu":               mu,
            "gamma":            gamma,
            "aux_weight":       aux_weight,
            "balance_weight":   balance_weight,
            "grad_clip":        grad_clip,
            "rollback_mult":    rollback_mult,
            "lpips_crop":       LPIPS_CROP,
            "seed":             seed,
            "save_folder":      save_folder,
            "compile":          USE_COMPILE,
            **model_kwargs,
        },
    )

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\nPreparing model  [phase={PHASE}  mode={mode_tag}]")
    model = build_denoiser(MODE, num_experts=NUM_EXPERTS, **model_kwargs).to(device)
    K = model.num_experts
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  mode={mode_tag}  experts={K}  params={n_params:.2f}M")

    loss_lpips = lpips.LPIPS(net="vgg").to(device)
    loss_lpips.eval()
    for p in loss_lpips.parameters():
        p.requires_grad_(False)

    # ── Logging ───────────────────────────────────────────────────────
    logfile = save_folder + f"log_p{PHASE}_{timestamp_str}.txt"
    write_log(logfile, (
        f"{'='*60}\n"
        f"  HDR Teacher — Phase {PHASE}  [{mode_tag.upper()}  K={K}]\n"
        f"{'='*60}\n"
        f"Timestamp  : {timestamp_str}\n"
        f"Dataset    : {dataset_dir}\n"
        f"Save folder: {save_folder}\n"
        f"Patch size : {PATCH_SIZE}  (None = full resolution)\n"
        f"Batch size : {batch_sz}   num_patch={num_patch}\n"
        f"LR         : {lr}  warmup={warmup_epochs}ep  "
        f"cosine→{eta_min}  total={num_epochs}ep\n"
        f"mu (tonemap): {mu}\n"
        f"Loss weights: gamma={gamma}  aux={aux_weight}  balance={balance_weight}\n"
        f"Grad clip  : {grad_clip}   rollback×{rollback_mult}\n"
        f"Model      : {model_kwargs}  params={n_params:.2f}M\n"
        f"{'='*60}\n\n"
    ), newfile=True)

    # ── Optimizer + scheduler ─────────────────────────────────────────
    # fused Adam: single multi-tensor CUDA kernel per step (A100 speedup)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     betas=(0.9, 0.999),
                                     fused=torch.cuda.is_available())
    except (TypeError, RuntimeError):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, warmup_epochs, num_epochs, eta_min)

    # ── Checkpoint resume ─────────────────────────────────────────────
    def _load_checkpoint(path, load_scheduler=True):
        """Load model + optimiser (+ scheduler) from a checkpoint file."""
        global start_epoch, last_epoch_loss, best_psnr_mu
        ckpt = torch.load(path, map_location=device, weights_only=False)
        # Strip torch.compile prefix if the checkpoint was saved compiled
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if load_scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch     = ckpt.get("epoch", 0)
        last_epoch_loss = ckpt.get("loss",  1e6)
        best_psnr_mu    = ckpt.get("best_psnr_mu", 0.0)
        print(f"  Loaded: epoch={start_epoch}  loss={last_epoch_loss:.4f}"
              f"  best_psnr_mu={best_psnr_mu:.2f}")

    if os.path.exists(save_path):
        print(f"Resuming Phase {PHASE} from {save_path}")
        _load_checkpoint(save_path)

    elif PHASE == 2 and os.path.exists(PHASE1_CHECKPOINT):
        print(f"Phase 2 init: loading Phase 1 weights from {PHASE1_CHECKPOINT}")
        _load_checkpoint(PHASE1_CHECKPOINT, load_scheduler=False)
        start_epoch     = 0
        last_epoch_loss = 1e6
        best_psnr_mu    = 0.0

    elif PHASE == 2:
        raise FileNotFoundError(
            f"Phase 2 requires a Phase 1 checkpoint.\n"
            f"Set PHASE1_CHECKPOINT to a valid path.\n"
            f"Currently: {PHASE1_CHECKPOINT}"
        )
    else:
        print("Starting Phase 1 from scratch.")

    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  Model compiled with TorchInductor.")

    model.train()

    # ── Dataset ───────────────────────────────────────────────────────
    print("Creating dataset...")

    if PHASE == 1:
        transform  = make_d4_transform(allow_transpose=True)
        crop_size  = PATCH_SIZE
        collate_fn = collate_xy
    else:
        transform  = make_d4_transform(allow_transpose=False)
        crop_size  = None
        collate_fn = collate_pad_to_max

    dataset = MobileHDRDataset(
        base_dir=dataset_dir,
        split="train",
        transform=transform,
        num_patch=num_patch,
        crop_size=crop_size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=8 if PHASE == 1 else 4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=worker_gen,
    )

    n_batches = len(dataloader)
    print(f"  {len(dataset)} virtual samples  →  {n_batches} batches/epoch")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nPhase {PHASE} training started  ({num_epochs} epochs)\n")
    training_t0 = time.time()

    for epoch in range(start_epoch, num_epochs):

        improved = False
        while not improved:
            running_loss    = 0.0
            running_psnr    = 0.0
            running_psnr_mu = 0.0
            comp_sums = {"l1_mu": 0.0, "percep": 0.0, "aux": 0.0, "balance": 0.0}
            usage_sum = torch.zeros(K, device=device)
            t1 = time.time()

            for i, sample in enumerate(dataloader):
                x = sample["x"].to(device, non_blocking=True)   # noisy packed BGGR
                y = sample["y"].to(device, non_blocking=True)   # clean packed BGGR GT
                B, C_in, H, W = x.shape      # H, W are packed Bayer dimensions

                # valid_mask at packed Bayer resolution: 1 on real pixels, 0 on padding.
                # Phase 1 patches are uniform → all-ones mask.
                # Phase 2 full-res images may be padded; orig_h/orig_w are in Bayer dims.
                if "orig_h" in sample:
                    orig_h = sample["orig_h"]
                    orig_w = sample["orig_w"]
                    valid_mask = torch.zeros(B, 1, H, W, device=device)
                    for b in range(B):
                        valid_mask[b, 0, :orig_h[b], :orig_w[b]] = 1.0
                else:
                    orig_h     = torch.full((B,), H)
                    orig_w     = torch.full((B,), W)
                    valid_mask = torch.ones(B, 1, H, W, device=device)

                with torch.no_grad():
                    snr_map = estimate_local_snr_map(x, window_size=5)

                optimizer.zero_grad(set_to_none=True)

                # ── Forward + losses ─────────────────────────────────
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                    # 1. Forward pass
                    #    y_pred:      [B, 4, H, W]    denoised packed BGGR
                    #    expert_outs: [B, K, 4, H, W] per-expert denoised Bayer
                    #    gates:       [B, K, H, W]    routing weights
                    y_pred, expert_outs, gates = model(x, snr_map)
                    C = y_pred.shape[1]   # 4 Bayer channels

                    # 2. Main L1-µ loss: denoised Bayer vs clean Bayer GT
                    tm_pred  = hdr_tonemap(y_pred,    mu=mu)
                    tm_gt    = hdr_tonemap(y.float(), mu=mu)
                    valid_px = valid_mask.sum()
                    loss_l1_mu = (((tm_pred - tm_gt).abs() * valid_mask).sum()
                                  / (valid_px * C + 1e-6))

                    # 3. Auxiliary per-expert losses
                    if aux_weight > 0:
                        tm_experts = hdr_tonemap(expert_outs, mu=mu)  # [B, K, 4, H, W]
                        err = (tm_experts - tm_gt.unsqueeze(1)).abs()
                        w   = gates.detach().unsqueeze(2) * valid_mask.unsqueeze(1)
                        loss_aux = ((w * err).sum(dim=(0, 2, 3, 4))
                                    / (w.sum(dim=(0, 2, 3, 4)) * C + 1e-6)).sum()
                    else:
                        loss_aux = x.new_zeros(())

                    # 4. Load balancing (MoE only)
                    gate_usage = ((gates * valid_mask).sum(dim=(0, 2, 3))
                                  / valid_px.clamp(min=1.0))
                    if balance_weight > 0:
                        loss_balance = K * (gate_usage ** 2).sum() - 1.0
                    else:
                        loss_balance = x.new_zeros(())


                # 5. Perceptual loss on a pseudo-RGB crop at Bayer resolution.
                # Pseudo-RGB: (R=ch3, G=avg(ch1,ch2), B=ch0) — no GBTF needed.
                # Run outside autocast so VGG stays in float32.
                min_h  = int(orig_h.min().item())
                min_w  = int(orig_w.min().item())
                crop_h = min(LPIPS_CROP, min_h)
                crop_w = min(LPIPS_CROP, min_w)
                top    = random.randint(0, min_h - crop_h)
                left   = random.randint(0, min_w - crop_w)

                def _lpips_crop(t):
                    # t: [B, 4, H, W] packed BGGR
                    c = t[:, :, top:top+crop_h, left:left+crop_w].float()
                    pseudo = torch.stack(
                        [c[:, 3], 0.5 * (c[:, 1] + c[:, 2]), c[:, 0]], dim=1)
                    return hdr_tonemap(pseudo.clamp(0, 1), mu=mu) * 2.0 - 1.0

                loss_perceptual = loss_lpips(_lpips_crop(y_pred),
                                             _lpips_crop(y)).mean()

                # 6. Total weighted loss
                ttl_loss = (loss_l1_mu
                            + gamma          * loss_perceptual
                            + aux_weight     * loss_aux
                            + balance_weight * loss_balance)

                ttl_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                # ── Running stats ────────────────────────────────────
                running_loss += ttl_loss.item()
                comp_sums["l1_mu"]   += loss_l1_mu.item()
                comp_sums["percep"]  += loss_perceptual.item()
                comp_sums["aux"]     += loss_aux.item()
                comp_sums["balance"] += loss_balance.item()
                usage_sum += gate_usage.detach().float()
                with torch.no_grad():
                    y_pred_c = y_pred.detach().float().clamp(0, 1)
                    y_gt_c   = y.float()
                    running_psnr    += batch_psnr_gpu(y_pred_c, y_gt_c)
                    running_psnr_mu += batch_psnr_gpu(
                        hdr_tonemap(y_pred_c, mu=mu),
                        hdr_tonemap(y_gt_c,   mu=mu))

                print_running_loss(running_loss, running_psnr_mu, i)

            # ── Epoch stats ───────────────────────────────────────────
            t2             = time.time()
            epoch_loss     = running_loss    / n_batches
            epoch_psnr     = running_psnr    / n_batches
            epoch_psnr_mu  = running_psnr_mu / n_batches
            epoch_usage    = (usage_sum / n_batches).tolist()
            epoch_msg = print_epoch_loss(epoch + 1, epoch_loss,
                                         epoch_psnr_mu, epoch_psnr,
                                         PHASE, t2 - t1)
            write_log(logfile, epoch_msg, end="")

            log_dict = {
                "epoch":               epoch + 1,
                "phase":               PHASE,
                "train/loss":          epoch_loss,
                "train/psnr_linear":   epoch_psnr,
                "train/psnr_mu":       epoch_psnr_mu,
                "train/lr":            optimizer.param_groups[0]["lr"],
                "train/loss_l1_mu":    comp_sums["l1_mu"]   / n_batches,
                "train/loss_percep":   comp_sums["percep"]  / n_batches,
                "train/loss_aux":      comp_sums["aux"]     / n_batches,
                "train/loss_balance":  comp_sums["balance"] / n_batches,
            }
            for k_idx, usage_k in enumerate(epoch_usage):
                log_dict[f"train/gate_usage_{k_idx}"] = usage_k
            wandb.log(log_dict)

            # ── Rollback check ────────────────────────────────────────
            improved = True
            if (last_epoch_psnr_mu > 10.0
                    and epoch_loss > rollback_mult * last_epoch_loss):
                print(f"\n  ⚠ Loss spike ({epoch_loss:.4f} > "
                      f"{rollback_mult}× {last_epoch_loss:.4f}) — reverting")
                improved = False
                if last_path and os.path.exists(last_path):
                    ckpt = torch.load(last_path, map_location=device,
                                      weights_only=False)
                    state = {k.replace("_orig_mod.", ""): v
                             for k, v in ckpt["model_state_dict"].items()}
                    (model._orig_mod if hasattr(model, "_orig_mod")
                     else model).load_state_dict(state)
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                last_epoch_psnr_mu = epoch_psnr_mu
                last_epoch_loss = epoch_loss

        scheduler.step()

        # ── Save checkpoints ──────────────────────────────────────────
        # model_kwargs / mode / num_experts stored so test script can
        # rebuild the exact architecture from the checkpoint alone.
        save_dict = {
            "epoch":                epoch + 1,
            "phase":                PHASE,
            "mode":                 mode_tag,
            "num_experts":          K,
            "model_kwargs":         model_kwargs,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss":                 epoch_loss,
            "best_psnr_mu":         best_psnr_mu,
        }
        torch.save(save_dict, save_path)
        last_path = save_path

        if epoch_psnr_mu > best_psnr_mu:
            best_psnr_mu = epoch_psnr_mu
            save_dict["best_psnr_mu"] = best_psnr_mu
            torch.save(save_dict, best_save_path)
            print(f"  ★ New best PSNR-µ: {best_psnr_mu:.2f} dB  → {best_save_path}")

    # ── Done ──────────────────────────────────────────────────────────
    total = time.time() - training_t0
    msg   = (f"\nPhase {PHASE} finished in {total/3600:.2f} h  |  "
             f"best PSNR-µ = {best_psnr_mu:.2f} dB  (Bayer domain)")
    write_log(logfile, msg)
    print(msg)

    artifact = wandb.Artifact(f"hdr_p{PHASE}_{mode_tag}_final", type="model")
    artifact.add_file(best_save_path)
    wandb.log_artifact(artifact, aliases=["best"])

    wandb.finish()

    if PHASE == 1:
        print(f"\nNext step → set  PHASE = 2  and  "
              f"PHASE1_CHECKPOINT = \"{best_save_path}\"")
