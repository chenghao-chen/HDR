#!/usr/bin/env python3
"""
Build the manifest for the two-family, two-benchmark learning-rate sweep.

Both earlier sweeps compared MoE against USVD at a single learning rate,
and the USVD SIDD sweep then showed that rate was worth up to 1.6 dB to one
family alone. Any cross-family claim made at a fixed rate is therefore
measuring tuning as much as architecture. This grid gives both families the
same five rates on both benchmarks so the comparison is like-for-like, and
extends above the previous maximum because every configuration was still
improving at the top of that grid -- 4e-4 was the edge of the search, not
an optimum.

    python scripts/make_lr_manifest.py --benchmark sidd --out lr_sidd.csv
    python scripts/make_lr_manifest.py --benchmark hdr  --out lr_hdr.csv
"""
from __future__ import annotations

import argparse
import csv

# Two sizes per family: one near the bottom of the shared range and one at
# each family's own top end, so a rate that only suits one scale shows up.
CONFIGS = [
    # family, label,    mode,   size args
    ("moe",  "moe_d8",   "moe",  {"dim": 8}),
    ("moe",  "moe_d32",  "moe",  {"dim": 32}),
    ("usvd", "usvd_K4",  "usvd", {"iters": 4, "dmodel": 128}),
    ("usvd", "usvd_K8",  "usvd", {"iters": 8, "dmodel": 256}),
]

LRS = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]


def lr_tag(lr: float) -> str:
    return f"{lr:g}".replace("-", "m").replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["sidd", "hdr"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="lr")
    args = ap.parse_args()

    rows = []
    for family, label, mode, size in CONFIGS:
        for lr in LRS:
            run = f"{label}_{lr_tag(lr)}"
            rows.append({
                "run": run,
                "benchmark": args.benchmark,
                "family": family,
                "mode": mode,
                "lr": f"{lr:g}",
                "dim": size.get("dim", 0),
                "iters": size.get("iters", 0),
                "dmodel": size.get("dmodel", 0),
                "folder": f"models_{args.tag}_{args.benchmark}_{run}",
            })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {args.out}  "
          f"({-(-len(rows) // 4)} nodes at 4 GPUs each)")


if __name__ == "__main__":
    main()
