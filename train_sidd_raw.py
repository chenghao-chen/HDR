#!/usr/bin/env python
"""
train_sidd_raw.py — RAW->RAW denoising on SIDD

A separate trainer from train_A100_MoE_two_phase.py on purpose. That script
is built around the HDR joint denoise+demosaic task: it demosaics the clean
Bayer with GBTF to make an RGB target, measures L1 in the mu-law tonemapped
domain, and adds an LPIPS perceptual term. None of that applies here. SIDD
is LDR linear raw scored with plain PSNR on the mosaic, so the objective is
L1 on packed Bayer with no tonemapping, no demosaicing and no perceptual
term -- and the model keeps its mosaic (upsample=False, out_channels=4).

Branching the HDR trainer to cover both would have put a second task's
control flow through the script that produced every result in this project
so far. Keeping them apart leaves those results reproducible.

Usage
-----
    python train_sidd_raw.py
    HDR_SIDD_EPOCHS=1 HDR_DIM=16 python train_sidd_raw.py   # smoke

Environment (same pattern as the HDR trainer):
    HDR_SIDD_ROOT       dataset root holding SIDD_Medium_Raw/ and the
                        Validation*.mat blocks
    HDR_SAVE_FOLDER     run directory; resumes from latest.pth inside it
    HDR_MODE            moe | film            (single/dual have no raw variant)
    HDR_NUM_EXPERTS, HDR_DIM, HDR_NUM_BLOCKS, HDR_EXPERT_BLOCKS, HDR_FILM_HIDDEN
    HDR_SIDD_EPOCHS, HDR_SIDD_BATCH, HDR_SIDD_CROP, HDR_DATALOADER_WORKERS
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from HDR_model_hybrid_Teacher import build_denoiser, estimate_local_snr_map
from hdr_data.sidd import SIDDMediumRaw, SIDDValidationRaw


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    """Blocks are normalised to [0, 1], so the peak is 1.0."""
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


@torch.no_grad()
def evaluate(model, val_loader, device) -> float:
    """Mean per-block PSNR on the mosaic — SIDD's own protocol."""
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        snr = estimate_local_snr_map(x, window_size=5)
        pred, _, _ = model(x, snr)
        mse = ((pred - y) ** 2).flatten(1).mean(1)
        total += float(psnr_from_mse(mse).sum())
        n += x.shape[0]
    model.train()
    return total / max(n, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    root = os.environ.get(
        "HDR_SIDD_ROOT",
        "/lus/eagle/projects/lighthouse-purdue/ryanchen/datasets/SIDD")
    save_folder = os.environ.get("HDR_SAVE_FOLDER", "models_sidd_raw/")
    if not save_folder.endswith("/"):
        save_folder += "/"
    os.makedirs(save_folder, exist_ok=True)

    MODE = os.environ.get("HDR_MODE", "moe")
    NUM_EXPERTS = int(os.environ.get("HDR_NUM_EXPERTS", "2"))
    DIM = int(os.environ.get("HDR_DIM", "32"))
    _nb = os.environ.get("HDR_NUM_BLOCKS", "4")
    NUM_BLOCKS = ([int(v) for v in _nb.split(",")] if "," in _nb
                  else [int(_nb)] * 4)
    EXPERT_BLOCKS = int(os.environ.get("HDR_EXPERT_BLOCKS", "2"))
    FILM_HIDDEN = int(os.environ.get("HDR_FILM_HIDDEN", "16"))

    epochs = int(os.environ.get("HDR_SIDD_EPOCHS", "60"))
    batch_sz = int(os.environ.get("HDR_SIDD_BATCH", "16"))
    crop = int(os.environ.get("HDR_SIDD_CROP", "256"))
    workers = int(os.environ.get("HDR_DATALOADER_WORKERS", "8"))
    lr, eta_min, warmup = 2e-4, 1e-6, 3
    grad_clip = 1.0
    balance_weight = 0.001 if MODE == "moe" else 0.0
    aux_weight = 0.5 if MODE == "moe" else 0.0

    # out_channels=4 / upsample=False are what make this RAW->RAW; both live
    # in model_kwargs so the checkpoint rebuilds the same architecture.
    model_kwargs = {
        "out_channels": 4,
        "upsample": False,
        "dim": DIM,
        "num_blocks": NUM_BLOCKS,
        "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8],
        "se_reduction": 8,
        "expert_blocks": EXPERT_BLOCKS,
    }
    if MODE == "film":
        model_kwargs["film_hidden"] = FILM_HIDDEN

    model = build_denoiser(MODE, num_experts=NUM_EXPERTS, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"mode={MODE} K={NUM_EXPERTS} dim={DIM} params={n_params/1e6:.3f}M")

    train_ds = SIDDMediumRaw(root, crop_size=crop, num_patch=8)
    val_ds = SIDDValidationRaw(root)
    print(f"train: {train_ds}\nval:   {val_ds}")

    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              drop_last=True, persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=min(workers, 4), pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(train_loader))
    start_epoch, best = 0, 0.0

    latest = save_folder + "latest.pth"
    if os.path.isfile(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_epoch, best = ck["epoch"], ck.get("best_psnr", 0.0)
        print(f"resumed from epoch {start_epoch} (best {best:.2f} dB)")

    print(f"\nstarting: {epochs} epochs x {steps} steps\n")
    for epoch in range(start_epoch, epochs):
        # Linear warmup then cosine decay, matching the HDR trainer's shape.
        if epoch < warmup:
            cur_lr = lr * (epoch + 1) / warmup
        else:
            t = (epoch - warmup) / max(1, epochs - warmup)
            cur_lr = eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * t))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        t0, run_loss = time.time(), 0.0
        for i, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.no_grad():
                snr = estimate_local_snr_map(x, window_size=5)

            pred, expert_outs, gates = model(x, snr)
            loss = F.l1_loss(pred, y)

            if aux_weight > 0.0 and expert_outs.shape[1] > 1:
                # Each expert is pulled toward the target in proportion to how
                # much the gate already uses it, detached so this term shapes
                # the experts and not the routing.
                w = gates.detach().unsqueeze(2)
                aux = (w * (expert_outs - y.unsqueeze(1)).abs()).mean()
                loss = loss + aux_weight * aux
            if balance_weight > 0.0:
                usage = gates.mean(dim=(0, 2, 3))
                loss = loss + balance_weight * (
                    gates.shape[1] * (usage ** 2).sum() - 1.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            run_loss += float(loss)
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch+1} step {i+1}/{steps} "
                      f"loss {run_loss/(i+1):.5f}", flush=True)

        val_psnr = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1}] loss {run_loss/steps:.5f}  "
              f"val PSNR {val_psnr:.3f} dB  lr {cur_lr:.2e}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "model_kwargs": model_kwargs,
            "mode": MODE,
            "num_experts": NUM_EXPERTS,
            "best_psnr": max(best, val_psnr),
            "val_psnr": val_psnr,
        }
        torch.save(ckpt, latest)
        if val_psnr > best:
            best = val_psnr
            torch.save(ckpt, save_folder + "sidd_best.pth")
            print(f"  new best: {best:.3f} dB", flush=True)

    print(f"\nfinished. best val PSNR {best:.3f} dB -> {save_folder}sidd_best.pth")


if __name__ == "__main__":
    main()
