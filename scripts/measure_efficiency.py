#!/usr/bin/env python
"""
Efficiency of every SIDD RAW->RAW candidate: parameters, FLOPs and latency.

    python scripts/measure_efficiency.py --out efficiency_sidd.csv

FLOPs are counted with torch.utils.flop_counter on a packed 4x512x512 input,
i.e. one sensor megapixel, so the column reads directly as GFLOPs/MP; every
model here is fully convolutional or patch-local, so cost scales linearly
with resolution. (FlopCounterMode counts a multiply-add as 2 FLOPs.)

Latency is measured on whatever GPU is visible, batch 1, after warmup, with
CUDA events, in fp32 and in fp16 autocast, at 1 MP and at a 12 MP phone
frame. A100 latency is a proxy for on-device cost, not a measurement of it --
the rankings between models are the useful part, not the absolute numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HDR_model_hybrid_Teacher import build_denoiser  # noqa: E402

RAW = {"out_channels": 4, "upsample": False}


def moe(dim, depth=4, eb=2, K=2):
    return ("moe", K, dict(RAW, dim=dim, num_blocks=[depth] * 4,
                           num_refinement_blocks=4, heads=[1, 2, 4, 8],
                           se_reduction=8, expert_blocks=eb))


def usvd(iters, dmodel):
    return ("usvd", 1, dict(RAW, in_channels=4, num_iters=iters, d_model=dmodel,
                            share_weights=False, patch_size=8, top_k=16,
                            chunk_size=1024, checkpoint=False))


CONFIGS = {
    "unet_n2n (ref)": ("unet_n2n", 1, {"in_channels": 4, **RAW}),
    **{f"unet w{w}": ("unet_n2n", 1, {"in_channels": 4, "width": w, **RAW})
       for w in (8, 12, 16, 24, 32)},
    "moe d4": moe(4), "moe d8": moe(8), "moe d16": moe(16), "moe d32": moe(32),
    "usvd K4": usvd(4, 128), "usvd K8": usvd(8, 256),
}


def flops(run, x):
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        run(x)
    return fc.get_total_flops()


@torch.no_grad()
def latency_ms(run, x, dtype, reps=20):
    ctx = (torch.autocast("cuda", dtype=dtype) if dtype != torch.float32
           else torch.autocast("cuda", enabled=False))
    with ctx:
        for _ in range(3):
            run(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            run(x)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="efficiency_sidd.csv")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, (mode, K, kw) in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        m = build_denoiser(mode, num_experts=K, **kw).to(dev).eval()
        params = sum(p.numel() for p in m.parameters())
        x1 = torch.rand(1, 4, 512, 512, device=dev)
        # Every model takes an SNR map in the shared signature; a random one
        # is enough for cost purposes (models that ignore it ignore it).
        run = lambda t, _m=m: _m(t, torch.rand_like(t[:, :1]))
        try:
            gf = flops(run, x1) / 1e9
        except Exception as e:  # noqa: BLE001
            gf = float("nan"); print(f"{name}: flop count failed: {e}")
        row = {"model": name, "params_M": params / 1e6, "GFLOPs_per_MP": gf}
        if dev.type == "cuda":
            x12 = torch.rand(1, 4, 1504, 2016, device=dev)  # ~12.1 MP sensor
            for tag, x in (("1MP", x1), ("12MP", x12)):
                for dname, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
                    try:
                        row[f"ms_{tag}_{dname}"] = latency_ms(run, x, dt)
                    except torch.cuda.OutOfMemoryError:
                        row[f"ms_{tag}_{dname}"] = float("nan")
                        torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
        del m
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
