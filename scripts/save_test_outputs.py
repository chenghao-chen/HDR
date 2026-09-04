#!/usr/bin/env python
"""
save_test_outputs.py — look at what the denoiser actually produced
===================================================================

Runs a trained checkpoint over a test split and writes, per frame, a single
figure you can judge by eye:

    row 1   noisy input        |  our prediction   |  reference (GT)
    row 2   expert 0 output    |  expert 1 output  |  |error|

plus a gate map showing which expert the router picked where. Every panel
label carries that panel's own PSNR-mu, so the figure answers "is this
better than the input, and which expert did the work" without cross-
referencing a table.

Why per-expert panels: the blended output can look fine while the mixture
has quietly collapsed onto one expert, which makes the other one dead
weight at inference time — exactly the thing an efficiency thesis must not
ship unnoticed. Compare the per-expert PSNRs and the gate map: if one
expert is never routed to, or both experts produce the same image, the MoE
is costing parameters and buying nothing.

Usage
-----
    python scripts/save_test_outputs.py                       # best ckpt, all 28
    python scripts/save_test_outputs.py --limit 6             # a quick look
    python scripts/save_test_outputs.py --crop-size 512       # small + fast (CPU ok)
    python scripts/save_test_outputs.py --inference tiled     # if full-res OOMs
    python scripts/save_test_outputs.py --noise high          # harder test noise

Outputs land in <out>/ as:
    frame_0000_panel.jpg     the figure above
    frame_0000_gates.jpg     per-pixel routing
    results.csv              one row per frame, every metric
    summary.md               aggregates, including per-expert and gate usage

Ground truth is *demosaiced*, not measured: the dataset stores clean CFA, so
the RGB reference is GBTF applied to it. No model can beat that reference by
more than GBTF's own accuracy on noise-free input. See BENCHMARKING.md.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdr_data import build_dataset                                # noqa: E402
from hdr_data.bayer import unpack                                 # noqa: E402
from hdr_eval.checkpoints import load_model_from_checkpoint       # noqa: E402
from hdr_eval.inference import InferenceConfig, infer             # noqa: E402
from hdr_eval.metrics import psnr, ssim                           # noqa: E402
from hdr_eval.tonemap import mu_law                               # noqa: E402
from hdr_eval.runner import _default_snr_fn                       # noqa: E402
from hdr_eval.visualize import (colorize, save_gate_map,          # noqa: E402
                                save_panel_grid, to_display)
from hdr_platform import get_accelerator                          # noqa: E402


#: mu small enough that mu_law is the identity to floating point:
#: log1p(mu*x)/log1p(mu) -> x as mu -> 0. Used when the panels have already
#: been tone-mapped and must pass through the grid untouched.
_IDENTITY_MU = 1e-6


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Save per-frame test visuals for a trained HDR denoiser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default=os.environ.get(
        "HDR_CHECKPOINT",
        "models_p1_moe_Teacher_MobileHDR_polaris01/phase1_best.pth"))
    p.add_argument("--dataset", default="mobile_hdr")
    p.add_argument("--split", default="test")
    p.add_argument("--noise", default=None,
                   help="Noise preset: low/medium/high/extreme. Default: the "
                        "dataset's own deterministic test noise.")
    p.add_argument("--crop-size", type=int, default=None,
                   help="Evaluate on a centre crop this many PACKED pixels "
                        "square. Much faster, and the only sane option on a "
                        "login node. Default: full frames.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N frames.")
    p.add_argument("--out", default=None,
                   help="Output directory. Default: test_visuals/<ckpt dir>.")
    p.add_argument("--scale", type=float, default=0.25,
                   help="Downscale factor for the saved figures. Full-res "
                        "panels are 3000x4000 each and nobody opens them twice.")
    p.add_argument("--inference", choices=("full", "tiled"), default="full")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--mu", type=float, default=5000.0,
                   help="Tone-map constant. 5000 matches training; changing "
                        "it makes these numbers incomparable to the W&B curves.")
    p.add_argument("--device", default=None, help="e.g. cuda:0, cpu. Default: auto.")
    p.add_argument("--quality", type=int, default=92)
    return p.parse_args(argv)


def build_gt_demosaic(pattern="BGGR"):
    """Clean packed CFA -> the RGB reference, the same way the runner does."""
    from hdr_baselines.pipelines import DEMOSAIC_FUNCTIONS
    fn = DEMOSAIC_FUNCTIONS["gbtf"]

    def demosaic(packed):
        return fn(unpack(packed.clamp(0, 1).float()), pattern).clamp(0, 1)
    return demosaic


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        print(f"FATAL: checkpoint not found: {args.checkpoint}")
        print("Available:")
        os.system("ls -1 models_*/*.pth 2>/dev/null || echo '  (none)'")
        return 1

    acc = get_accelerator(args.device)
    device = acc.device
    out_dir = args.out or os.path.join(
        "test_visuals",
        os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint))))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  Saving test outputs")
    print("=" * 70)
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}")
    print(f"output     : {out_dir}/")

    model, info = load_model_from_checkpoint(args.checkpoint, device=device)
    K = int(getattr(model, "num_experts", 0) or 0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model      : mode={info.mode}  experts={K}  params={n_params/1e6:.2f}M")

    ds_kwargs = {"split": args.split}
    if args.noise:
        ds_kwargs["noise"] = args.noise
    if args.crop_size:
        ds_kwargs["crop_size"] = args.crop_size
    ds = build_dataset(args.dataset, **ds_kwargs)
    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"dataset    : {args.dataset}/{args.split}  {total} of {len(ds)} frames"
          f"{'  noise=' + args.noise if args.noise else ''}"
          f"{'  crop=' + str(args.crop_size) if args.crop_size else '  full-res'}")
    print(f"inference  : {args.inference}"
          + (f" tile={args.tile} overlap={args.overlap}"
             if args.inference == "tiled" else ""))
    print("-" * 70)

    gt_demosaic = build_gt_demosaic()
    cfg = InferenceConfig(mode=args.inference, tile=args.tile,
                          overlap=args.overlap,
                          amp_dtype=torch.bfloat16 if device.type in
                          ("cuda", "xpu") else None)

    fieldnames = (["frame", "psnr_mu", "psnr_linear", "ssim",
                   "noisy_psnr_mu", "gain_db", "chroma_pred", "chroma_gt",
                   "chroma_ratio"]
                  + [f"expert{k}_psnr_mu" for k in range(K)]
                  + [f"expert{k}_gate_mean" for k in range(K)]
                  + ["seconds"])
    rows = []

    for i in range(total):
        sample = ds[i]
        x = sample["x"].unsqueeze(0).to(device)
        y = sample["y"].unsqueeze(0).to(device)

        gt_rgb = gt_demosaic(y)
        noisy_rgb = gt_demosaic(x)

        t0 = time.time()
        with torch.no_grad():
            out = infer(model, x, _default_snr_fn, cfg)
        elapsed = time.time() - t0
        out = out.detach_cpu()
        gt_rgb, noisy_rgb = gt_rgb.cpu(), noisy_rgb.cpu()

        tm_gt = mu_law(gt_rgb, args.mu)
        tm_pred = mu_law(out.blended.clamp(0, 1), args.mu)
        tm_noisy = mu_law(noisy_rgb, args.mu)

        # Colour-fidelity check, independent of PSNR/SSIM: mean |channel -
        # luma|. A model that quietly collapses to R = B (see BENCHMARKING.md,
        # "The CFA phase question") can still score well on PSNR/SSIM if the
        # scene is dim, since those are luma-dominated; this is not.
        chroma_pred = float((out.blended.clamp(0, 1)
                             - out.blended.clamp(0, 1).mean(1, keepdim=True)
                             ).abs().mean())
        chroma_gt = float((gt_rgb - gt_rgb.mean(1, keepdim=True)).abs().mean())

        row = {
            "frame": i,
            "psnr_mu": psnr(tm_pred, tm_gt),
            "psnr_linear": psnr(out.blended.clamp(0, 1), gt_rgb),
            "ssim": ssim(tm_pred, tm_gt),
            "noisy_psnr_mu": psnr(tm_noisy, tm_gt),
            "chroma_pred": chroma_pred,
            "chroma_gt": chroma_gt,
            "chroma_ratio": chroma_pred / max(chroma_gt, 1e-9),
            "seconds": elapsed,
        }
        row["gain_db"] = row["psnr_mu"] - row["noisy_psnr_mu"]

        # ── the figure ────────────────────────────────────────────────
        # Every panel is tone-mapped and downscaled HERE, so the grid stacks
        # ready-made images. The error heatmap is already RGB in [0, 1] and
        # must not go through the tone curve a second time, which is why the
        # grid itself is called with a mu small enough to be the identity.
        def panel(img):
            return to_display(img, args.mu, args.scale)

        top = {
            f"noisy  {row['noisy_psnr_mu']:.2f}dB": panel(noisy_rgb),
            f"predicted  {row['psnr_mu']:.2f}dB": panel(out.blended.clamp(0, 1)),
            "reference (GT)": panel(gt_rgb),
        }

        # Error is measured through the tone curve, matching the loss and the
        # eye: a linear-light error map is all highlights.
        err = (panel(out.blended.clamp(0, 1))
               - panel(gt_rgb)).abs().mean(1, keepdim=True)

        bottom = {}
        for k in range(K):
            ek = out.experts[:, k].clamp(0, 1)
            pk = psnr(mu_law(ek, args.mu), tm_gt)
            row[f"expert{k}_psnr_mu"] = pk
            row[f"expert{k}_gate_mean"] = float(out.gates[:, k].mean())
            bottom[f"expert{k}  {pk:.2f}dB"] = panel(ek)
        err_panel = {"|error| (tone-mapped)": colorize(err, 0.0, None).unsqueeze(0)}

        stem = os.path.join(out_dir, f"frame_{i:04d}")
        grid = [top, {**bottom, **err_panel}] if bottom else [{**top, **err_panel}]
        save_panel_grid(f"{stem}_panel.jpg", grid,
                        mu=_IDENTITY_MU, scale=1.0, quality=args.quality)

        if out.gates is not None and K > 1:
            save_gate_map(f"{stem}_gates.jpg", out.gates, scale=args.scale,
                          quality=args.quality)

        rows.append(row)
        experts_txt = "  ".join(
            f"e{k}={row[f'expert{k}_psnr_mu']:.2f}" for k in range(K))
        print(f"  frame {i:3d}  PSNR-mu {row['psnr_mu']:6.2f} dB "
              f"(noisy {row['noisy_psnr_mu']:5.2f}, {row['gain_db']:+.2f})  "
              f"SSIM {row['ssim']:.4f}  chroma {row['chroma_ratio']:5.1%}  "
              f"{experts_txt}  {elapsed:.2f}s")

    if not rows:
        print("No frames evaluated.")
        return 1

    # ── results.csv ───────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    def avg(key):
        return sum(r[key] for r in rows) / len(rows)

    # ── summary.md ────────────────────────────────────────────────────
    best = max(rows, key=lambda r: r["psnr_mu"])
    worst = min(rows, key=lambda r: r["psnr_mu"])
    lines = [
        f"# Test outputs — {os.path.basename(out_dir)}",
        "",
        f"* checkpoint: `{args.checkpoint}`",
        f"* mode `{info.mode}`, {K} experts, {n_params/1e6:.2f}M parameters",
        f"* {args.dataset}/{args.split}, {len(rows)} frames, "
        f"{args.inference} inference"
        + (f", noise `{args.noise}`" if args.noise else "")
        + (f", {args.crop_size}px crops" if args.crop_size else ", full resolution"),
        f"* metrics through mu-law (mu={args.mu:g}); GT is GBTF-demosaiced clean CFA",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---|",
        f"| PSNR-mu | **{avg('psnr_mu'):.2f} dB** |",
        f"| PSNR (linear) | {avg('psnr_linear'):.2f} dB |",
        f"| SSIM | {avg('ssim'):.4f} |",
        f"| noisy input PSNR-mu | {avg('noisy_psnr_mu'):.2f} dB |",
        f"| **gain over input** | **{avg('gain_db'):+.2f} dB** |",
        f"| **chroma retention** | **{avg('chroma_ratio'):.1%}** of reference |",
        f"| seconds / frame | {avg('seconds'):.2f} |",
        "",
    ]
    if K:
        lines += ["## Experts", "",
                  "| expert | PSNR-mu alone | mean gate weight |", "|---|---|---|"]
        for k in range(K):
            lines.append(f"| {k} | {avg(f'expert{k}_psnr_mu'):.2f} dB | "
                         f"{avg(f'expert{k}_gate_mean'):.3f} |")
        gates = [avg(f"expert{k}_gate_mean") for k in range(K)]
        spread = max(avg(f"expert{k}_psnr_mu") for k in range(K)) - \
            min(avg(f"expert{k}_psnr_mu") for k in range(K))
        lines += ["", f"Gate weights sum to {sum(gates):.3f}. "
                  f"Per-expert PSNR spread is {spread:.2f} dB."]
        if min(gates) < 0.05:
            lines.append("")
            lines.append(f"> **Routing has collapsed**: expert "
                         f"{gates.index(min(gates))} receives a mean gate "
                         f"weight of {min(gates):.3f}. It is costing "
                         f"parameters and latency for nothing — drop it, or "
                         f"raise the balance loss.")
        elif spread < 0.15:
            lines.append("")
            lines.append(f"> **Experts are near-duplicates**: their solo PSNRs "
                         f"differ by only {spread:.2f} dB, so the mixture is "
                         f"buying little over a single head of the same size.")
    if avg("chroma_ratio") < 0.5:
        lines.append("")
        lines.append(f"> **Output is desaturated**: the model retains only "
                     f"{avg('chroma_ratio'):.0%} of the reference's chroma "
                     f"(mean |channel - luma|, pred vs. GT). A healthy model "
                     f"should track near 100%. This is the signature of the "
                     f"CFA-phase augmentation bug — see BENCHMARKING.md's "
                     f"\"The CFA phase question\" — not a denoising failure.")
    lines += ["", "## Range", "",
              f"* best:  frame {best['frame']} at {best['psnr_mu']:.2f} dB "
              f"(`frame_{best['frame']:04d}_panel.jpg`)",
              f"* worst: frame {worst['frame']} at {worst['psnr_mu']:.2f} dB "
              f"(`frame_{worst['frame']:04d}_panel.jpg`)",
              "",
              "Each `frame_*_panel.jpg` is: noisy | predicted | reference on "
              "the top row, and each expert's own output plus the tone-mapped "
              "absolute error on the bottom. `frame_*_gates.jpg` colours each "
              "pixel by which expert the router chose (red = expert 0, "
              "green = expert 1).",
              ]
    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("-" * 70)
    print(f"PSNR-mu {avg('psnr_mu'):.2f} dB   SSIM {avg('ssim'):.4f}   "
          f"gain {avg('gain_db'):+.2f} dB over the noisy input")
    if K:
        for k in range(K):
            print(f"  expert {k}: {avg(f'expert{k}_psnr_mu'):6.2f} dB alone, "
                  f"mean gate {avg(f'expert{k}_gate_mean'):.3f}")
    print(f"\nwrote {len(rows)} figures + results.csv + summary.md to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
