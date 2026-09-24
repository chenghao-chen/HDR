#!/usr/bin/env python3
"""
Turn a list of target parameter counts into concrete model configurations.

The architecture sweep asks "at a fixed parameter budget, does routing (MoE),
continuous conditioning (FiLM), or a different trunk/expert split buy anything?"
That question only means something if every arm at a given budget really lands
on that budget, so the width has to be solved per arm rather than shared: a K=4
MoE carries four expert heads where a K=1 carries one, and matching their total
parameter counts means giving them different trunk widths.

Depth is fixed per budget by the ladder below and shared by every arm at that
budget, so within a budget the arms differ in mechanism only. See LADDER for
why the budgets are defined by walking the reference arm's width rather than
by naming round parameter counts.

    python scripts/solve_param_grid.py --out sweep_manifest.csv

Writes a CSV with one row per run: the environment every training process
needs, plus the achieved parameter count for the report to check against.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HDR_model_hybrid_Teacher import build_denoiser  # noqa: E402

# The size ladder, as (depth, reference width) rungs.
#
# The budgets are defined by walking the reference arm's width at the depth
# this project has used throughout, rather than by naming round numbers and
# solving for them. Only even widths build at all (odd ones fail inside
# pixel_shuffle) and parameters go as width squared, so the reachable sizes
# are a coarse lattice; asking for round targets forces depth to jump around
# between budgets, which would confound the size axis with a depth axis and
# make the scaling trend unreadable.
#
# The last rung is the exception that earns its place: depth 4 bottoms out
# near 95k parameters, so the ~50k point is only reachable by going shallower.
# It is kept because the low end is the interesting end for edge deployment,
# and flagged in the manifest so the report can mark it as off-ladder.
LADDER = [
    (2, 2),   # ~52k  — off-ladder, the only rung not at depth 4
    (4, 2),   # ~95k
    (4, 4),   # ~345k
    (4, 6),   # ~770k
    (4, 8),   # ~1.3M
    (4, 10),  # ~2.1M
    (4, 14),  # ~4.0M
    (4, 18),  # ~6.6M
    (4, 24),  # ~11.7M
    (4, 32),  # ~20.7M  — the current baseline
]

# The arms compared at every budget. `label` becomes part of the run folder
# name, so it has to stay filesystem-safe and stable.
#
# ref=True marks the arm whose depth solution is adopted by the whole budget.
ARMS = [
    # label        mode      K  expert_blocks  film_hidden  ref
    # No "single" arm. SingleDenoiser wraps TransUNet_Teacher_HDR, which is a
    # different network from the trunk MoEDenoiser and FiLMDenoiser share, so
    # it would confound "does conditioning help" with "is this other backbone
    # better" — and it measured 7.6 dB against ~15 dB for every other arm at a
    # comparable budget in the rehearsal. moeK1 is the honest control: the same
    # trunk and head, with a gate that is trivially 1.0.
    ("moeK1",     "moe",     1, 2,             16,          False),
    ("moeK2",     "moe",     2, 2,             16,          True),
    ("moeK3",     "moe",     3, 2,             16,          False),
    ("moeK4",     "moe",     4, 2,             16,          False),
    ("film",      "film",    1, 2,             16,          False),
    ("trunkheavy", "moe",    2, 1,             16,          False),
    ("expertheavy", "moe",   2, 4,             16,          False),
    ("moeK8",     "moe",     8, 2,             16,          False),
]

# pixel_shuffle inside the trunk needs the level widths divisible by 4, and
# odd widths fail outright, so the search walks even widths only.
DIM_CANDIDATES = list(range(2, 97, 2))


def count_params(mode, num_experts, dim, depth, expert_blocks, film_hidden):
    """Build the model on the meta device and count its parameters."""
    kwargs = dict(
        dim=dim,
        num_blocks=[depth] * 4,
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        se_reduction=8,
        expert_blocks=expert_blocks,
    )
    if mode == "film":
        kwargs["film_hidden"] = film_hidden
    with torch.device("meta"):
        model = build_denoiser(mode, num_experts=num_experts, **kwargs)
    return sum(p.numel() for p in model.parameters())


def solve_dim(mode, num_experts, depth, expert_blocks, film_hidden, target):
    """Width whose parameter count lands closest to `target` at this depth."""
    best = None
    for dim in DIM_CANDIDATES:
        try:
            n = count_params(mode, num_experts, dim, depth, expert_blocks,
                             film_hidden)
        except Exception:
            continue
        err = abs(n - target) / target
        if best is None or err < best[2]:
            best = (dim, n, err)
    return best


def size_tag(n_params):
    """Short, sortable, filesystem-safe name for a budget."""
    if n_params < 1e6:
        return f"{round(n_params / 1e3):d}k"
    return f"{n_params / 1e6:.1f}M".replace(".", "p")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="arch",
                    help="tag embedded in each run folder name")
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="warn when an arm misses its budget by more than this "
                         "fraction (default: %(default)s)")
    ap.add_argument("--out", default="sweep_manifest.csv")
    args = ap.parse_args()

    ref = next(a for a in ARMS if a[5])
    rows = []
    for depth, ref_dim in LADDER:
        # The reference arm at this rung defines the budget the other arms
        # are then fitted to, so "same budget" means the same number of
        # parameters rather than the same width.
        target = count_params(ref[1], ref[2], ref_dim, depth, ref[3], ref[4])
        tag = size_tag(target)
        for label, mode, K, eb, fh, _ in ARMS:
            got = solve_dim(mode, K, depth, eb, fh, target)
            if got is None:
                print(f"  SKIP {label} @ {tag} — no width fits", file=sys.stderr)
                continue
            dim, n, err = got
            rows.append({
                "run": f"{tag}_{label}",
                "folder": f"models_arch_{tag}_{label}_{args.tag}_polaris",
                "target_params": int(target),
                "actual_params": n,
                "rel_err": round(err, 4),
                "mode": mode,
                "num_experts": K,
                "dim": dim,
                "num_blocks": depth,
                "expert_blocks": eb,
                "film_hidden": fh if mode == "film" else "",
            })
            flag = "  <-- off budget" if err > args.tolerance else ""
            print(f"{tag:>7} {label:<12} depth={depth} dim={dim:<3} "
                  f"{n/1e6:8.4f}M (err {err*100:5.1f}%){flag}")

    with open(args.out, "w", newline="") as fh_out:
        # lineterminator="\n": csv defaults to CRLF, which leaves a stray
        # carriage return on each row's last field. The shell worker reads
        # this file with IFS=',', so that CR ends up inside an exported
        # environment variable and the trainer dies parsing it.
        writer = csv.DictWriter(fh_out, fieldnames=list(rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    off = [r for r in rows if r["rel_err"] > args.tolerance]
    print(f"\n{len(rows)} runs -> {args.out}")
    print(f"GPUs needed: {len(rows)}  ({-(-len(rows) // 4)} Polaris nodes)")
    if off:
        print(f"{len(off)} run(s) miss their budget by >{args.tolerance*100:.0f}%:")
        for r in off:
            print(f"  {r['run']}: {r['actual_params']/1e6:.4f}M "
                  f"vs {r['target_params']/1e6:.4f}M target")


if __name__ == "__main__":
    main()
