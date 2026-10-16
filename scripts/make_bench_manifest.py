#!/usr/bin/env python3
"""
Manifests for the benchmark sweeps (consumed by scripts/polaris/bench_worker.sh).

    python scripts/make_bench_manifest.py sidd --out bench_sidd.csv

SIDD design, and why:

* Long schedule. The earlier SIDD recipe was ~9.6k steps (60 epochs x 160),
  and every run was still improving when the cosine schedule reached zero;
  published supervised raw-SIDD baselines train for far longer. This uses
  200 epochs x 640 steps = 128k steps, ~13x more.
* The reference. unet_n2n is the published supervised baseline (51.19 dB on
  the SIDD validation blocks). Reproducing it here is what licenses comparing
  our models' numbers against the literature at all.
* Two learning rates per configuration, bracketing each family's optimum
  from the short-schedule LR sweep, because the optimum tends to move once
  the schedule is 13x longer.
* Three seeds per (configuration, rate). The project has never measured
  run-to-run noise; without it, no gap between models can be called real.
* Two controls on the reference only: the old short recipe, and the long
  recipe without augmentation, so the calibration result can be attributed
  to schedule vs augmentation.

All runs report the FINAL checkpoint's validation PSNR (result.json), not the
best epoch, which would be selected on the very blocks it is scored on.
"""

import argparse
import csv

SIDD_CACHE = ("/lus/eagle/projects/lighthouse-purdue/ryanchen/datasets/SIDD/"
              "cache_packed_f32")
LONG = {"HDR_SIDD_EPOCHS": 200, "HDR_SIDD_NUM_PATCH": 32, "HDR_SIDD_AUG": 1,
        "HDR_SIDD_BATCH": 16, "HDR_SIDD_CROP": 256, "HDR_SIDD_CACHE": SIDD_CACHE}

UNET = {"HDR_MODE": "unet_n2n"}
def moe(d): return {"HDR_MODE": "moe", "HDR_DIM": d, "HDR_NUM_EXPERTS": 2}
def usvd(k, dm): return {"HDR_MODE": "usvd", "HDR_USVD_ITERS": k,
                         "HDR_USVD_DMODEL": dm, "HDR_USVD_SHARE": 0}

# label, model env, learning rates, seeds
SIDD_GRID = [
    ("unet",   UNET,            [1e-4, 3e-4],   [1, 2, 3]),
    ("moed4",  moe(4),          [8e-4, 1.6e-3], [1, 2, 3]),
    ("moed8",  moe(8),          [8e-4, 1.6e-3], [1, 2, 3]),
    ("moed16", moe(16),         [4e-4, 8e-4],   [1, 2, 3]),
    ("moed32", moe(32),         [4e-4],         [1, 2]),
    ("usvdk4", usvd(4, 128),    [8e-4, 1.6e-3], [1, 2, 3]),
    ("usvdk8", usvd(8, 256),    [4e-4, 8e-4],   [1, 2, 3]),
]
SIDD_CONTROLS = [
    # The pre-fix recipe on the reference: short schedule, no augmentation.
    ("unet_oldrecipe", {**UNET, "HDR_SIDD_EPOCHS": 60, "HDR_SIDD_NUM_PATCH": 8,
                        "HDR_SIDD_AUG": 0}, 3e-4, 1),
    ("unet_noaug", {**UNET, "HDR_SIDD_AUG": 0}, 3e-4, 1),
]


# ── Mobile-HDR joint denoise+demosaic ────────────────────────────────────
#
# What "already out there" means on this task: there is no public leaderboard
# for this dataset, so the references are (a) the classical pipelines already
# scored in test_results/baseline_comparison, and (b) the standard learned
# JDD architectures -- DemosaicNet (Gharbi et al. 2016), a U-Net, DnCNN and a
# Restormer-lite -- trained here with this project's recipe and data, at two
# learning rates each so they are not handicapped by our tuning.
#
# Our side: the MoE ladder at its tuned rates; a linear-domain loss term at
# two weights, because the current model scores BELOW the un-denoised input on
# linear PSNR; and a 3x schedule, because 50 epochs x 218 steps (~11k steps)
# is the same order as the SIDD schedule that turned out under-trained.
# USVD is excluded: the LR sweep put it 3.5 dB behind MoE on this task at its
# best rate, and it is 8-17x slower than the reference at 12 MP.
def hmoe(d): return {"HDR_MODE": "moe", "HDR_DIM": d, "HDR_NUM_EXPERTS": 2}
def hbase(n): return {"HDR_MODE": "baseline", "HDR_BASELINE": n}

HDR_GRID = [
    # label,            env,                                  lrs,        seeds
    ("moed4",          hmoe(4),                               [1.6e-3],   [1, 2, 3]),
    ("moed8",          hmoe(8),                               [8e-4],     [1, 2, 3]),
    ("moed16",         hmoe(16),                              [4e-4],     [1, 2, 3]),
    ("moed32",         hmoe(32),                              [2e-4],     [1, 2, 3]),
    ("moed8_lin10",    {**hmoe(8), "HDR_LIN_WEIGHT": 10},     [8e-4],     [1, 2, 3]),
    ("moed8_lin30",    {**hmoe(8), "HDR_LIN_WEIGHT": 30},     [8e-4],     [1, 2, 3]),
    ("moed8_ep150",    {**hmoe(8), "HDR_NUM_EPOCHS": 150},    [8e-4],     [1, 2]),
    ("moed4_ep150",    {**hmoe(4), "HDR_NUM_EPOCHS": 150},    [1.6e-3],   [1]),
    ("demosaicnet",    hbase("demosaicnet"),                  [2e-4, 8e-4], [1, 2]),
    ("unetjdd",        hbase("unet"),                         [2e-4, 8e-4], [1, 2]),
    ("dncnn",          hbase("dncnn"),                        [2e-4, 8e-4], [1, 2]),
    ("restormerlite",  hbase("restormer_lite"),               [2e-4, 8e-4], [1, 2]),
]


def hdr_rows(prefix):
    rows = []
    for label, model, lrs, seeds in HDR_GRID:
        for lr in lrs:
            for s in seeds:
                run = f"{label}_lr{tag(lr)}_s{s}"
                e = {**model, "HDR_LR": lr, "HDR_SEED": s}
                rows.append((run, "hdr", f"{prefix}_{run}", env(e)))
    return rows


# ── SIDD round 2: the efficient U-Net family ──────────────────────────────
#
# Round 1 found the plain reference U-Net beats every MoE/USVD design at every
# cost, that augmentation (cell-level D4) costs 0.25 dB, and that U-Net seeds
# agree to +-0.01 dB. So this round makes the U-Net cheaper rather than trying
# to rescue the transformer designs: width from 48 (the reference, 92 GF/MP)
# down to 8 (~30x fewer FLOPs), with and without a residual output, at two
# learning rates. One seed per cell (the measured noise floor allows it), plus
# a second seed on one slice to confirm that floor holds at small widths.
NOAUG = {**LONG, "HDR_SIDD_AUG": 0}

def unet(w, res): return {"HDR_MODE": "unet_n2n", "HDR_UNET_WIDTH": w,
                          "HDR_UNET_RESIDUAL": int(res)}

def sidd_eff_rows(prefix):
    rows = []
    def add(run, e):
        rows.append((run, "sidd", f"{prefix}_{run}", env({**NOAUG, **e})))
    for w in (8, 12, 16, 24, 32, 48):
        for res in (0, 1):
            for lr in (3e-4, 1e-3):
                add(f"unet_w{w}_r{res}_lr{tag(lr)}_s1",
                    {**unet(w, res), "HDR_SIDD_LR": lr, "HDR_SEED": 1})
        add(f"unet_w{w}_r1_lr{tag(3e-4)}_s2",
            {**unet(w, 1), "HDR_SIDD_LR": 3e-4, "HDR_SEED": 2})
    # Schedule headroom: the reference was still gaining at 128k steps.
    for s_ in (1, 2):
        add(f"unet_w48_r0_lr{tag(3e-4)}_ep400_s{s_}",
            {**unet(48, 0), "HDR_SIDD_LR": 3e-4, "HDR_SEED": s_,
             "HDR_SIDD_EPOCHS": 400})
    # Fairness: round 1 trained MoE with the augmentation that hurt the U-Net.
    add("moed16_noaug_lr0p0004_s1", {**moe(16), "HDR_SIDD_LR": 4e-4, "HDR_SEED": 1})
    add("moed8_noaug_lr0p0008_s1", {**moe(8), "HDR_SIDD_LR": 8e-4, "HDR_SEED": 1})
    return rows


def tag(lr):
    return f"{lr:g}".replace("-", "m").replace(".", "p")


def env(d):
    return ";".join(f"{k}={v}" for k, v in d.items())


def sidd_rows(prefix):
    rows = []
    for label, model, lrs, seeds in SIDD_GRID:
        for lr in lrs:
            for s in seeds:
                run = f"{label}_lr{tag(lr)}_s{s}"
                e = {**LONG, **model, "HDR_SIDD_LR": lr, "HDR_SEED": s}
                rows.append((run, "sidd", f"{prefix}_{run}", env(e)))
    for label, model, lr, s in SIDD_CONTROLS:
        run = f"{label}_lr{tag(lr)}_s{s}"
        e = {**LONG, **model, "HDR_SIDD_LR": lr, "HDR_SEED": s}
        rows.append((run, "sidd", f"{prefix}_{run}", env(e)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["sidd", "sidd_eff", "hdr", "hdr_smoke"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default=None)
    a = ap.parse_args()
    if a.benchmark == "sidd":
        rows = sidd_rows(a.prefix or "models_bench_sidd")
    elif a.benchmark == "sidd_eff":
        rows = sidd_eff_rows(a.prefix or "models_eff_sidd")
    elif a.benchmark == "hdr":
        rows = hdr_rows(a.prefix or "models_bench_hdr")
    else:
        # One epoch of each new code path: learned baselines through the
        # HDR trainer, and the linear-domain loss.
        pick = {"demosaicnet_lr0p0008_s1", "unetjdd_lr0p0008_s1",
                "restormerlite_lr0p0008_s1", "moed8_lin30_lr0p0008_s1"}
        rows = [(r, b, f.replace("models_bench_hdr", "models_hdrsmoke"),
                 e + ";HDR_NUM_EPOCHS=1")
                for r, b, f, e in hdr_rows("models_bench_hdr") if r in pick]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["run", "benchmark", "folder", "env"])
        w.writerows(rows)
    print(f"{len(rows)} runs -> {a.out}  ({-(-len(rows)//4)} nodes)")


if __name__ == "__main__":
    main()
