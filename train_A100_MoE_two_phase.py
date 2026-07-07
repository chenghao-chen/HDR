"""
train_A100_MoE_two_phase.py — Two-phase HDR MoE training on MobileHDR
=====================================================================

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

Model modes (MODE flag)
───────────────────────
  "moe"    MoEDenoiser — ONE shared trunk + K lightweight expert heads with
           a learned per-pixel gate conditioned on (noisy input, SNR map).
           Experts specialise on different noise levels. ~half the params
           and FLOPs of "dual" while supporting K ≥ 2 experts.
  "dual"   Legacy DualSNRDenoiser — two full teachers blended by the SNR map.
  "single" One teacher, no routing (ablation).

All modes share one forward signature:
    blended, expert_outs, gates = model(x, snr_map)
and one loss:
    L = L1-µ(blended) + γ·LPIPS + aux·Σ_k gate_k-weighted L1-µ(expert_k)
        + bal·load-balance                       (balance: moe only)

Usage
─────
  Phase 1:  set PHASE = 1, then submit job.
  Phase 2:  set PHASE = 2  AND  set PHASE1_CHECKPOINT to the best .pth from
            the Phase 1 run.  Phase 2 auto-loads it if no Phase 2 checkpoint
            exists in the current save folder.

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
Full-res images are non-square so Transpose is skipped (Phase 2); H-flip and
V-flip independently already cover the 4 dimension-preserving symmetries.
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
from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR

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
    Stores original sizes in orig_h / orig_w so padded pixels can be
    excluded from the loss via a binary valid_mask.
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
        "orig_h": torch.tensor([s["x"].shape[1] for s in batch]),
        "orig_w": torch.tensor([s["x"].shape[2] for s in batch]),
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
    # save_folder      = "models_p1_moe_Teacher_MobileHDR_20260605_0848/"
    dataset_dir      = "/scratch/gilbreth/chen4848/datasets/Mobile-HDR"
    create_folder(save_folder)

    # Checkpoint paths inside this run's save folder
    save_path      = save_folder + "latest.pth"
    best_save_path = save_folder + (f"phase{PHASE}_best.pth")
    last_path      = None          # tracks the most recently written checkpoint

    # ── Phase-specific hyperparameters ────────────────────────────────
    if PHASE == 1:
        # ── Patch training ────────────────────────────────────────────
        # 512×512 packed patches (= 1024×1024 sensor crop).
        # Bottleneck Restormer sees 64×64 = 4096 tokens — sufficient for
        # global noise statistics without processing the entire 49K-token
        # full-resolution feature map.
        PATCH_SIZE     = 512
        batch_sz       = 8
        num_patch      = 8    # 8 virtual repeats per image per epoch
        lr             = 1e-4  # higher LR safe with batch 8
        warmup_epochs  = 3     # short ramp; smoke test needs active LR quickly
        num_epochs     = 50
        eta_min        = 3e-5  # LR decays 1e-4→3e-5; stays active through epoch 50
        gamma          = 0.1   # moderate perceptual weight
        grad_clip      = 1.0
        rollback_mult  = 3.0   # tighter than Phase 2; batch 8 has low variance
        LPIPS_CROP     = 256   # sub-crop from 512×512 patch for LPIPS

    else:
        # ── Full-resolution fine-tune ─────────────────────────────────
        # batch_sz=1 is the limit for full frames on A100-80 GB.
        # Very small LR: the model is already trained — just adapt the
        # boundary/global-context statistics to full-resolution inputs.
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
    mu             = 5000   # µ-law tonemapping constant (HDR literature standard)
    aux_weight     = 0.5  if MODE in ("moe", "dual") else 0.0
    balance_weight = 0.01 if MODE == "moe" else 0.0   # anti expert-collapse

    start_epoch = 0
    best_psnr_mu = 0.0
    last_epoch_psnr_mu, last_epoch_loss = 0.0, 1e6

    model_kwargs = {
        "dim":                  32,
        "num_blocks":           [4, 4, 4, 4],
        "num_refinement_blocks": 4,
        "heads":                [1, 2, 4, 8],
        "se_reduction":         8,
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

    # Fixed GBTF demosaicing — used to produce clean RGB GT from clean Bayer
    # on GPU inside the training loop (no backprop through it).
    gbtf = DifferentiableGBTF_BGGR().to(device)
    gbtf.eval()
    for p in gbtf.parameters():
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
        # Resume this phase's own interrupted run
        print(f"Resuming Phase {PHASE} from {save_path}")
        _load_checkpoint(save_path)

    elif PHASE == 2 and os.path.exists(PHASE1_CHECKPOINT):
        # Start Phase 2 from the best Phase 1 weights.
        # Do NOT load the Phase 1 scheduler state — Phase 2 uses a
        # different schedule and the state would be meaningless.
        print(f"Phase 2 init: loading Phase 1 weights from {PHASE1_CHECKPOINT}")
        _load_checkpoint(PHASE1_CHECKPOINT, load_scheduler=False)
        start_epoch  = 0     # Phase 2 epoch counter resets
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

    # Compile AFTER any checkpoint load so weights map onto the raw module.
    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  Model compiled with TorchInductor.")

    model.train()

    # ── Dataset ───────────────────────────────────────────────────────
    print("Creating dataset...")

    if PHASE == 1:
        # Square patches: full 8-way D4. Crop happens inside the dataset
        # (before noise synthesis) — the transform only flips/transposes.
        transform  = make_d4_transform(allow_transpose=True)
        crop_size  = PATCH_SIZE
        collate_fn = collate_xy    # uniform patch size, skip duplicate 'xm'
    else:
        # Non-square full frames: 4 dimension-preserving symmetries only.
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
    # NOTE: autocast uses bfloat16 (same exponent range as fp32), so no
    # GradScaler is needed — plain backward + clip + step.
    print(f"\nPhase {PHASE} training started  ({num_epochs} epochs)\n")
    training_t0 = time.time()

    for epoch in range(start_epoch, num_epochs):

        # ── Inner loop with rollback ───────────────────────────────────
        # Re-runs the epoch if a loss spike is detected (e.g. outlier image).
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
                y = sample["y"].to(device, non_blocking=True)   # clean packed BGGR
                B, _, H, W = x.shape      # H, W are packed Bayer dimensions
                Hs, Ws = H * 2, W * 2    # sensor (output) dimensions

                # Build valid_mask at sensor resolution: 1 on real pixels, 0 on padding.
                # Phase 1 patches are uniform → mask is all ones.
                # Phase 2 full-res images may be padded by collate_pad_to_max;
                # orig_h / orig_w are in Bayer space, so multiply by 2 for sensor space.
                if "orig_h" in sample:
                    orig_h = sample["orig_h"]
                    orig_w = sample["orig_w"]
                    valid_mask = torch.zeros(B, 1, Hs, Ws, device=device)
                    for b in range(B):
                        valid_mask[b, 0, :orig_h[b] * 2, :orig_w[b] * 2] = 1.0
                else:
                    orig_h     = torch.full((B,), H)
                    orig_w     = torch.full((B,), W)
                    valid_mask = torch.ones(B, 1, Hs, Ws, device=device)

                with torch.no_grad():
                    snr_map = estimate_local_snr_map(x, window_size=5)
                    # Demosaic clean Bayer GT → clean RGB at sensor resolution.
                    # pixel_shuffle converts (B,4,H,W) packed BGGR → (B,1,2H,2W) mosaic.
                    y_rgb = gbtf(F.pixel_shuffle(y.float(), 2))   # [B, 3, Hs, Ws]

                optimizer.zero_grad(set_to_none=True)

                # ── Forward + losses ─────────────────────────────────
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                    # 1. Forward pass (unified across moe/dual/single).
                    #    y_pred:      [B, 3, Hs, Ws]    sensor-resolution RGB
                    #    expert_outs: [B, K, 3, Hs, Ws] per-expert RGB
                    #    gates:       [B, K, Hs, Ws]    routing weights at sensor res
                    y_pred, expert_outs, gates = model(x, snr_map)
                    C = y_pred.shape[1]  # 3 RGB channels

                    # 2. Main L1-µ loss on valid sensor pixels
                    tm_pred  = hdr_tonemap(y_pred, mu=mu)
                    tm_gt    = hdr_tonemap(y_rgb,  mu=mu)
                    valid_px = valid_mask.sum()
                    loss_l1_mu = (((tm_pred - tm_gt).abs() * valid_mask).sum()
                                  / (valid_px * C + 1e-6))

                    # 3. Auxiliary per-expert losses: each expert's error is
                    # weighted by its own (detached) gate so experts specialise
                    # on the pixels routed to them.
                    if aux_weight > 0:
                        tm_experts = hdr_tonemap(expert_outs, mu=mu)
                        err = (tm_experts - tm_gt.unsqueeze(1)).abs()
                        w   = gates.detach().unsqueeze(2) * valid_mask.unsqueeze(1)
                        loss_aux = ((w * err).sum(dim=(0, 2, 3, 4))
                                    / (w.sum(dim=(0, 2, 3, 4)) * C + 1e-6)).sum()
                    else:
                        loss_aux = x.new_zeros(())

                    # 4. Load balancing (moe only): K·Σ(mean gate)² is 1 when
                    # usage is uniform and grows toward K on collapse.
                    # Gates are at sensor resolution; valid_mask is also at sensor res.
                    gate_usage = ((gates * valid_mask).sum(dim=(0, 2, 3))
                                  / valid_px.clamp(min=1.0))
                    if balance_weight > 0:
                        loss_balance = K * (gate_usage ** 2).sum() - 1.0
                    else:
                        loss_balance = x.new_zeros(())

                # 5. Perceptual loss on a 256×256 random crop.
                # Run outside autocast so LPIPS VGG stays in float32.
                # y_pred and y_rgb are both [B, 3, Hs, Ws] RGB — crop directly.
                min_h  = int(orig_h.min().item()) * 2   # sensor dims = 2 × Bayer dims
                min_w  = int(orig_w.min().item()) * 2
                crop_h = min(LPIPS_CROP, min_h)
                crop_w = min(LPIPS_CROP, min_w)
                top    = random.randint(0, min_h - crop_h)
                left   = random.randint(0, min_w - crop_w)

                def _lpips_crop(t):
                    # t is [B, 3, Hs, Ws] RGB (R=ch0, G=ch1, B=ch2)
                    crop = t[:, :, top:top+crop_h, left:left+crop_w].float()
                    return hdr_tonemap(crop.clamp(0, 1), mu=mu) * 2.0 - 1.0

                loss_perceptual = loss_lpips(_lpips_crop(y_pred),
                                             _lpips_crop(y_rgb)).mean()

                # 6. Total weighted loss
                ttl_loss = (loss_l1_mu
                            + gamma          * loss_perceptual
                            + aux_weight     * loss_aux
                            + balance_weight * loss_balance)

                # ── Backward (bf16 autocast → no GradScaler needed) ──
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
                    running_psnr    += batch_psnr_gpu(y_pred_c, y_rgb)
                    running_psnr_mu += batch_psnr_gpu(
                        hdr_tonemap(y_pred_c, mu=mu),
                        hdr_tonemap(y_rgb,    mu=mu))

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
            # If the loss spikes by more than rollback_mult × last epoch's
            # loss, reload the previous checkpoint and retry the epoch.
            # This guards against outlier images or transient instability.
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

        # ── LR schedule step (accepted epochs only) ───────────────────
        # Advancing the scheduler on a rolled-back epoch would decay LR
        # even though the model weights were reverted.
        scheduler.step()

        # ── Save checkpoints ──────────────────────────────────────────
        # model_kwargs / mode / num_experts are stored so the test script
        # can rebuild the exact architecture without manual sync.
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
             f"best PSNR-µ = {best_psnr_mu:.2f} dB")
    write_log(logfile, msg)
    print(msg)

    artifact = wandb.Artifact(f"hdr_p{PHASE}_{mode_tag}_final", type="model")
    artifact.add_file(best_save_path)
    wandb.log_artifact(artifact, aliases=["best"])

    wandb.finish()

    if PHASE == 1:
        print(f"\nNext step → set  PHASE = 2  and  "
              f"PHASE1_CHECKPOINT = \"{best_save_path}\"")
