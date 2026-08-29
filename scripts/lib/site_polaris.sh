#!/bin/bash
# ============================================================================
# scripts/lib/site_polaris.sh — ALCF Polaris
#
# Sourced by hdr_env.sh when the detected site is polaris. Defines
# hdr::site_setup, plus the queue facts the submit scripts embed as PBS
# directives.
#
# Hardware: 4x NVIDIA A100 40 GB per node, 32 physical cores (64 threads),
#           /lus/eagle. The training script was written for an 80 GB A100 —
#           see the Phase 2 note at the bottom.
# ============================================================================

# ── Queue facts (documentation for the PBS headers; not used at runtime) ────
# debug          <= 2 nodes,  <= 1 h    fastest turnaround
# debug-scaling  <= 10 nodes, <= 1 h
# prod / small   >= 10 nodes            unusable for a 1-node job
# preemptable    <= 72 h, 1+ nodes      CAN be killed and requeued mid-run
#
# The consequence: an 8-hour single-node training job has to go to
# preemptable, which is why the train script is marked rerunnable (#PBS -r y)
# and pins its save folder so a requeue resumes from latest.pth.
HDR_QUEUE_DEBUG="debug"
HDR_QUEUE_LONG="preemptable"
HDR_FILESYSTEMS="home:eagle"
HDR_SELECT_EXTRA="system=polaris"

hdr::site_setup() {
    HDR_PROJ_ROOT="${HDR_PROJ_ROOT:-/lus/eagle/projects/${HDR_PROJECT:-lighthouse-purdue}/${USER}}"

    # The conda module on Polaris is broken for this project ("module load
    # conda" fails), so the environment is a miniforge install on eagle.
    # scripts/setup/polaris_env.sh builds it.
    HDR_PYTHON="${HDR_PYTHON:-$HDR_PROJ_ROOT/miniforge3/envs/hdr/bin/python}"

    HDR_PROXY="${HDR_PROXY:-http://proxy.alcf.anl.gov:3128}"
    HDR_SMI="nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader"

    # 8 DataLoader workers x 4 threads = 32, matching the node's physical
    # core count and leaving the main process room to feed the GPU.
    hdr::setup_threads "${HDR_THREADS:-4}"

    # The training script is single-GPU. The other three A100s on the node sit
    # idle; using them would need DDP, which is a code change, not a flag.
    export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
}

# ── Phase 2 memory note ─────────────────────────────────────────────────────
# train_A100_MoE_two_phase.py records that "batch_sz=1 is the limit for full
# frames on A100-80 GB". Polaris A100s have 40 GB, so PHASE=2 (full-resolution
# fine-tune) will very likely OOM here even at batch_sz=1. Phase 1 (512x512
# patches, batch 8) is the one expected to fit.
#
# Options if Phase 2 is needed: gradient checkpointing, tiled fine-tuning, or
# run it on Aurora, whose Max 1550 tiles carry 64 GB each.
