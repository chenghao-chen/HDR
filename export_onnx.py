"""
export_onnx.py — Export the MoEDenoiser to ONNX for Netron visualisation.

Usage (no checkpoint — random weights, architecture only):
    python export_onnx.py

Usage (with a checkpoint to get real weights):
    python export_onnx.py --checkpoint models_p1_moe_Teacher_MobileHDR_20260707_2033/phase1_best.pth

The exported file can be opened at https://netron.app or with the Netron
desktop app (pip install netron; python -c "import netron; netron.start('model.onnx')")
"""

import argparse
import torch
from HDR_model_hybrid_Teacher import build_denoiser, estimate_local_snr_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (optional — uses random weights if omitted)")
    parser.add_argument("--out", default="model.onnx",
                        help="Output ONNX file path (default: model.onnx)")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Packed Bayer patch size for the dummy input (default: 256)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Build model
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        mode        = ckpt.get("mode", "moe")
        num_experts = ckpt.get("num_experts", 2)
        kwargs      = ckpt.get("model_kwargs", {
            "out_channels": 4, "dim": 32,
            "num_blocks": [4, 4, 4, 4], "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8], "se_reduction": 8,
        })
        model = build_denoiser(mode, num_experts, **kwargs)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        model = build_denoiser("moe", num_experts=2, out_channels=4, dim=32,
                               num_blocks=[4, 4, 4, 4], num_refinement_blocks=4,
                               heads=[1, 2, 4, 8], se_reduction=8)
        print("Using random weights (no checkpoint supplied).")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f} M")

    # Dummy inputs — packed Bayer (B=1, 4-ch, H, W) and SNR map (B=1, 1-ch, H, W)
    P = args.patch_size
    x   = torch.zeros(1, 4, P, P, device=device)
    snr = estimate_local_snr_map(x)

    print(f"Exporting to {args.out} (opset {args.opset}, patch {P}×{P}) ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, snr),
            args.out,
            input_names=["noisy_bayer", "snr_map"],
            output_names=["denoised_bayer", "expert_outs", "gate_weights"],
            dynamic_axes={
                "noisy_bayer":    {0: "batch", 2: "height", 3: "width"},
                "snr_map":        {0: "batch", 2: "height", 3: "width"},
                "denoised_bayer": {0: "batch", 2: "height", 3: "width"},
                "expert_outs":    {0: "batch", 3: "height", 4: "width"},
                "gate_weights":   {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Saved: {args.out}")
    print("Open at https://netron.app  or run:")
    print(f"  pip install netron && python -c \"import netron; netron.start('{args.out}')\"")


if __name__ == "__main__":
    main()
