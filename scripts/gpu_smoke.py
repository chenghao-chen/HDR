#!/usr/bin/env python
"""
Prove the project runs on this machine's GPU, on the real data.

    python scripts/gpu_smoke.py                  # everything
    python scripts/gpu_smoke.py --steps 1        # quicker
    python scripts/gpu_smoke.py --skip-eval      # train path only
    python scripts/gpu_smoke.py --device cpu     # login-node dry run

The pytest suite runs CPU-only on a login node, where every gpu-marked test
auto-skips. This is the complement: a genuine train step and a
full-resolution inference pass on an actual compute node, on the actual
Mobile-HDR data, against whichever accelerator the machine has — an A100 on
Polaris, a Max 1550 tile on Aurora.

It is deliberately a *script*, not a test. What it catches is the class of
failure that never shows up on CPU:

    * a kernel that has no XPU implementation and silently falls back, or
      raises,
    * bf16 autocast producing non-finite values on one backend,
    * a memory ceiling — Polaris' A100 is 40 GB, an Aurora tile is 64 GB,
      and full-resolution inference is close to both,
    * pinned-memory and DataLoader-worker settings that only misbehave with
      a real device attached.

Exit status is 0 only if every stage passed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Run from anywhere: the project modules live in the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402

from hdr_platform import configure, resolve_dataset_dir  # noqa: E402


# Same architecture the training script builds, so the memory figures this
# reports are the ones a real run will see.
MODEL_KWARGS = dict(dim=32, num_blocks=[4, 4, 4, 4], num_refinement_blocks=4,
                    heads=[1, 2, 4, 8], se_reduction=8)


def _rule(title: str) -> None:
    print(f"\n{'#' * 4} {title} {'#' * max(4, 70 - len(title))}", flush=True)


def build_model(acc, num_experts: int):
    from HDR_model_hybrid_Teacher import build_denoiser

    model = build_denoiser("moe", num_experts=num_experts,
                           **MODEL_KWARGS).to(acc.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model      : {n_params:.2f}M params, "
          f"{model.num_experts} experts, on {acc.device}")
    return model


def train_steps(acc, model, dataset_dir: str, steps: int, batch_size: int,
                crop: int) -> None:
    """
    A real forward/backward/step on real patches.

    Runs the same sequence the training loop does — SNR map, GBTF-demosaiced
    target, autocast forward, backward, grad clip, optimiser step — because
    each of those is a separate opportunity for a backend to disagree.
    """
    from HDR_model_hybrid_Teacher import estimate_local_snr_map
    from HDR_Mobile_dataset import MobileHDRDataset
    from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR
    from train_A100_MoE_two_phase import hdr_tonemap, batch_psnr_gpu, collate_xy

    gbtf = DifferentiableGBTF_BGGR().to(acc.device).eval()

    ds = MobileHDRDataset(base_dir=dataset_dir, split="train",
                          crop_size=crop, num_patch=1)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_xy,
        **acc.dataloader_kwargs(num_workers=4))
    print(f"train set  : {len(ds)} samples, "
          f"batch {batch_size} at {crop}x{crop}")

    optimizer = torch.optim.Adam(
        model.parameters(), **acc.optimizer_kwargs(lr=1e-4))
    print(f"optimizer  : Adam(fused={acc.supports_fused_adam})")

    model.train()
    acc.reset_peak_memory()
    losses = []
    t0 = time.time()

    for i, sample in enumerate(dl):
        if i >= steps:
            break
        x = sample["x"].to(acc.device, non_blocking=True)
        y = sample["y"].to(acc.device, non_blocking=True)

        with torch.no_grad():
            snr = estimate_local_snr_map(x, window_size=5)
            y_rgb = gbtf(F.pixel_shuffle(y.float(), 2))

        optimizer.zero_grad(set_to_none=True)
        with acc.autocast(torch.bfloat16):
            pred, _experts, _gates = model(x, snr)
            loss = (hdr_tonemap(pred) - hdr_tonemap(y_rgb)).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        value = loss.item()
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(
                f"step {i}: loss is {value} — bf16 autocast is producing "
                "non-finite values on this backend.")
        losses.append(value)

        psnr = batch_psnr_gpu(
            hdr_tonemap(pred.detach().float().clamp(0, 1)), hdr_tonemap(y_rgb))
        print(f"  step {i}  in={tuple(x.shape)}  out={tuple(pred.shape)}  "
              f"loss={value:.5f}  psnr-mu={psnr:.2f}", flush=True)

    acc.synchronize()
    if not losses:
        raise RuntimeError("the DataLoader yielded no batches")
    print(f"{len(losses)} train steps in {time.time() - t0:.1f}s; "
          f"peak memory {acc.peak_memory_gb():.1f} GB")


def eval_full_frame(acc, model, dataset_dir: str) -> None:
    """
    One full-resolution inference pass — the Phase 2 / benchmark shape.

    This is the memory high-water mark of the whole project and the reason
    Phase 2 is expected to OOM on a 40 GB A100 but fit on a 64 GB Aurora tile.
    """
    from HDR_Mobile_dataset import MobileHDRDataset
    from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR
    from train_A100_MoE_two_phase import hdr_tonemap
    from test_dual_MoE_two_phase import infer_full, psnr, ssim

    gbtf = DifferentiableGBTF_BGGR().to(acc.device).eval()
    test_ds = MobileHDRDataset(base_dir=dataset_dir, split="test")
    print(f"test set   : {len(test_ds)} samples")

    sample = test_ds[0]
    x = sample["x"].unsqueeze(0).to(acc.device)
    y = sample["y"].unsqueeze(0).to(acc.device)

    model.eval()
    acc.reset_peak_memory()
    with torch.no_grad():
        t0 = time.time()
        pred, _experts, gates = infer_full(model, x, device=acc.device)
        acc.synchronize()
        elapsed = time.time() - t0

        rgb_gt = gbtf(F.pixel_shuffle(y.clamp(0, 1).float(), 2)).clamp(0, 1)

        print(f"full-res   : in={tuple(x.shape)} -> out={tuple(pred.shape)} "
              f"in {elapsed:.2f}s, peak memory {acc.peak_memory_gb():.1f} GB")

        expected = (x.shape[-2] * 2, x.shape[-1] * 2)
        if tuple(pred.shape[-2:]) != expected:
            raise RuntimeError(
                f"output is {tuple(pred.shape[-2:])}, expected {expected} "
                "(sensor resolution is 2x the packed Bayer dims)")
        if not torch.isfinite(pred).all():
            raise RuntimeError("full-resolution prediction is non-finite")

        p = pred.clamp(0, 1).float()
        print(f"metrics    : PSNR-mu {psnr(hdr_tonemap(p), hdr_tonemap(rgb_gt)):.2f} dB, "
              f"SSIM {ssim(p, rgb_gt):.4f}")
        print("             (random weights — low numbers are expected here; "
              "this checks the path runs, not that it is trained)")

        gate_sums = gates.sum(1)
        if not torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-3):
            raise RuntimeError(
                f"gates do not sum to 1 (range "
                f"{gate_sums.min():.4f}..{gate_sums.max():.4f}) — the softmax "
                "over experts is misbehaving in reduced precision.")
        print("gates      : sum to 1 across experts")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None,
                   help="cuda:0, xpu:0, cpu (default: this machine's GPU).")
    p.add_argument("--steps", type=int, default=3, help="Training steps to run.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop", type=int, default=512, help="Training patch size.")
    p.add_argument("--experts", type=int, default=2)
    p.add_argument("--dataset-dir", default=None,
                   help="Defaults to this machine's Mobile-HDR location.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip the full-resolution pass (it is the memory peak).")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Do not fail when no GPU is present. For dry runs.")
    args = p.parse_args(argv)

    rt = configure(seed=21, device=args.device)
    print(rt.banner())

    if not rt.accelerator.is_gpu and not args.allow_cpu:
        print("\nFAIL: no GPU on this node. This script is meant to run inside "
              "a job; pass --allow-cpu for a dry run on a login node.",
              file=sys.stderr)
        return 1

    dataset_dir = args.dataset_dir or resolve_dataset_dir("Mobile-HDR")
    print(f"dataset    : {dataset_dir}")
    if not os.path.isdir(os.path.join(dataset_dir, "train", "tensors")):
        print(f"\nFAIL: {dataset_dir} has no train/tensors. "
              "Stage the data with scripts/stage_data.sh.", file=sys.stderr)
        return 1

    acc = rt.accelerator
    try:
        _rule("1. Model construction")
        model = build_model(acc, args.experts)

        _rule("2. Forward / backward on real training patches")
        train_steps(acc, model, dataset_dir, args.steps, args.batch_size,
                    args.crop)

        if args.skip_eval:
            print("\n(skipping the full-resolution pass at --skip-eval)")
        else:
            _rule("3. Full-resolution inference")
            eval_full_frame(acc, model, dataset_dir)
    except Exception as exc:
        print(f"\nGPU SMOKE TEST FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print("\nGPU SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
