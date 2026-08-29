#!/bin/bash -l
#PBS -N hdr_smoke
#PBS -A lighthouse-purdue
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:eagle
#PBS -o logs/
#PBS -e logs/

# SUPERSEDED: the maintained version of this job is now
#     scripts/polaris/smoke.pbs      (qsub it, or ./scripts/submit.sh smoke)
# which shares its environment setup with the Aurora equivalent in
# scripts/aurora/smoke.pbs. This file still works and is kept as the
# Polaris-only original; see RUNNING.md.

# ============================================================================
# GPU smoke test: proves the project actually runs on a Polaris A100, not just
# on the login node's CPU.
#
# The pytest suite runs CPU-only on a login node (gpu-marked tests auto-skip).
# This job runs the SAME suite on a compute node, where those tests execute for
# real, and then drives a genuine train step + a full-resolution eval pass on
# the actual Mobile-HDR data.
#
# Submit:  qsub submit_smoke_polaris.sh
# ============================================================================

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1
mkdir -p logs

PROJ_ROOT=/lus/eagle/projects/lighthouse-purdue/ryanchen
PYTHON="$PROJ_ROOT/miniforge3/envs/hdr/bin/python"

export HDR_DATASET_DIR="${HDR_DATASET_DIR:-$PROJ_ROOT/datasets/Mobile-HDR}"
export PYTHONUNBUFFERED=1
unset PYTHONPATH

# Without these OpenBLAS spawns one thread per core and dies at import.
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=================================================================="
echo "job     : ${PBS_JOBID:-interactive}"
echo "node    : $(hostname)"
echo "python  : $PYTHON"
echo "dataset : $HDR_DATASET_DIR"
echo "started : $(date)"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "################ 1. Full pytest suite (GPU tests now active) ################"
./run_tests.sh -q
suite_status=$?

echo
echo "################ 2. GPU forward/backward on real data ######################"
"$PYTHON" - <<'PYEOF'
import os, sys, time
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F

from HDR_model_hybrid_Teacher import build_denoiser, estimate_local_snr_map
from HDR_Mobile_dataset import MobileHDRDataset
from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR
from train_A100_MoE_two_phase import hdr_tonemap, batch_psnr_gpu, collate_xy

assert torch.cuda.is_available(), "no CUDA device on this compute node"
dev = torch.device("cuda:0")
print("device:", torch.cuda.get_device_name(0))

# Same architecture the training script uses.
model_kwargs = dict(dim=32, num_blocks=[4, 4, 4, 4], num_refinement_blocks=4,
                    heads=[1, 2, 4, 8], se_reduction=8)
model = build_denoiser("moe", num_experts=2, **model_kwargs).to(dev)
print("params: %.2fM  experts=%d" % (
    sum(p.numel() for p in model.parameters()) / 1e6, model.num_experts))

gbtf = DifferentiableGBTF_BGGR().to(dev).eval()

ds = MobileHDRDataset(base_dir=os.environ["HDR_DATASET_DIR"], split="train",
                      crop_size=512, num_patch=1)
dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=4,
                                 collate_fn=collate_xy)
print("train samples:", len(ds))

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

losses = []
t0 = time.time()
for i, s in enumerate(dl):
    if i >= 3:
        break
    x = s["x"].to(dev, non_blocking=True)
    y = s["y"].to(dev, non_blocking=True)
    with torch.no_grad():
        snr = estimate_local_snr_map(x, window_size=5)
        y_rgb = gbtf(F.pixel_shuffle(y.float(), 2))

    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred, experts, gates = model(x, snr)
        loss = (hdr_tonemap(pred) - hdr_tonemap(y_rgb)).abs().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    losses.append(loss.item())
    print("  step %d  in=%s  out=%s  loss=%.5f  psnr-mu=%.2f" % (
        i, tuple(x.shape), tuple(pred.shape), loss.item(),
        batch_psnr_gpu(hdr_tonemap(pred.detach().float().clamp(0, 1)),
                       hdr_tonemap(y_rgb))))

assert all(torch.isfinite(torch.tensor(l)) for l in losses), "non-finite loss"
print("3 train steps ok in %.1fs; peak GPU mem %.1f GB" % (
    time.time() - t0, torch.cuda.max_memory_allocated() / 1e9))

# ── Full-resolution eval path (the Phase-2 / benchmark shape) ────────────────
from test_dual_MoE_two_phase import infer_full, psnr, ssim

test_ds = MobileHDRDataset(base_dir=os.environ["HDR_DATASET_DIR"], split="test")
print("test samples:", len(test_ds))
sample = test_ds[0]
x = sample["x"].unsqueeze(0).to(dev)
y = sample["y"].unsqueeze(0).to(dev)
model.eval()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    t1 = time.time()
    pred, experts, gates = infer_full(model, x, device=dev)
    torch.cuda.synchronize()
    rgb_gt = gbtf(F.pixel_shuffle(y.clamp(0, 1).float(), 2)).clamp(0, 1)
    print("full-res infer: in=%s -> out=%s in %.2fs, peak mem %.1f GB" % (
        tuple(x.shape), tuple(pred.shape), time.time() - t1,
        torch.cuda.max_memory_allocated() / 1e9))
    assert pred.shape[-2:] == (x.shape[-2] * 2, x.shape[-1] * 2), "sensor-res mismatch"
    assert torch.isfinite(pred).all(), "non-finite prediction"
    p = pred.clamp(0, 1).float()
    print("untrained-model PSNR-mu %.2f dB  SSIM %.4f  (random weights: low is expected)" % (
        psnr(hdr_tonemap(p), hdr_tonemap(rgb_gt)), ssim(p, rgb_gt)))
    print("gates sum to 1: ", bool(torch.allclose(gates.sum(1), torch.ones_like(gates.sum(1)), atol=1e-3)))

print("\nGPU SMOKE TEST PASSED")
PYEOF
gpu_status=$?

echo
echo "=================================================================="
echo "pytest suite exit : $suite_status"
echo "gpu smoke exit    : $gpu_status"
echo "finished          : $(date)"
echo "=================================================================="
exit $(( suite_status || gpu_status ))
