#!/bin/bash -l
#PBS -N hdr_train
#PBS -A lighthouse-purdue
#PBS -q preemptable
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:eagle
#PBS -r y
#PBS -o logs/
#PBS -e logs/

# SUPERSEDED: the maintained version of this job is now
#     scripts/polaris/train.pbs      (qsub it, or ./scripts/submit.sh train)
# which shares its environment setup with the Aurora equivalent in
# scripts/aurora/train.pbs. This file still works and is kept as the
# Polaris-only original; see RUNNING.md.

# ============================================================================
# Polaris (ALCF) PBS port of submit_train.sh, which targeted Purdue Gilbreth
# (SLURM). Differences that matter:
#
#   queue      Polaris prod/small require >= 10 nodes, so a 1-node job can only
#              use debug (<=1 h), debug-scaling (<=1 h), or preemptable (<=72 h).
#              8 h single-node training therefore has to be preemptable.
#   -r y       preemptable jobs CAN be killed mid-run. This marks the job
#              rerunnable so PBS requeues it, and HDR_SAVE_FOLDER below is
#              pinned so the restart resumes from latest.pth instead of
#              starting a fresh timestamped run.
#   GPU        Polaris has A100 40 GB, not the 80 GB the original targeted.
#              See the Phase 2 warning at the bottom of this file.
#   network    compute nodes have no direct outbound route; W&B needs the
#              ALCF proxy, so this defaults to offline + a later sync.
#
# Submit:   qsub submit_train_polaris.sh
# Override: qsub -v HDR_SAVE_FOLDER=models_p1_moe_run2/ submit_train_polaris.sh
# ============================================================================

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1
mkdir -p logs

# ── Paths ───────────────────────────────────────────────────────────────────
PROJ_ROOT=/lus/eagle/projects/lighthouse-purdue/ryanchen
PYTHON="$PROJ_ROOT/miniforge3/envs/hdr/bin/python"

export HDR_DATASET_DIR="${HDR_DATASET_DIR:-$PROJ_ROOT/datasets/Mobile-HDR}"

# Pinned run directory. Do NOT make this timestamped: on requeue after
# preemption the script looks for <folder>/latest.pth to resume, and a new
# timestamp each attempt would silently restart training from epoch 0.
export HDR_SAVE_FOLDER="${HDR_SAVE_FOLDER:-models_p1_moe_Teacher_MobileHDR_polaris01/}"

# ── Python runtime ──────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
# Clear any inherited PYTHONPATH; miniforge warns it can shadow env packages.
unset PYTHONPATH

# OpenBLAS otherwise spawns one thread per core and dies on Polaris' core count
# ("pthread_create failed ... RLIMIT_NPROC"), which is a hard crash on import.
# Keep this small: DataLoader workers INHERIT it, and Phase 1 uses 8 workers,
# so 8 workers x 4 threads = 32 threads against the node's 64 hardware threads,
# leaving headroom for the main process feeding the GPU.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ── Weights & Biases ────────────────────────────────────────────────────────
# Compute nodes reach the internet only through the ALCF proxy.
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"

# Keep W&B scratch off /home (small quota) and on eagle.
export WANDB_CACHE_DIR="$PROJ_ROOT/wandb_cache"
export WANDB_DATA_DIR="$PROJ_ROOT/wandb_data"
export WANDB_DIR="$PROJ_ROOT/wandb_runs"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_DIR"

# Credentials come from either .env (WANDB_API_KEY) or `wandb login` (~/.netrc).
# Never commit .env — .gitignore covers it.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# Online mode blocks on network calls, so only pick it when auth actually exists.
if [[ -n "${WANDB_MODE:-}" ]]; then
    echo "WANDB_MODE preset to '$WANDB_MODE' by caller; leaving it alone."
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=online
    echo "W&B auth: WANDB_API_KEY -> online."
elif grep -qs 'machine[[:space:]]\+api\.wandb\.ai' "${NETRC:-$HOME/.netrc}"; then
    # ~/.netrc lives on /home, which is in -l filesystems, so nodes can read it.
    export WANDB_MODE=online
    echo "W&B auth: ~/.netrc (wandb login) -> online."
else
    export WANDB_MODE=offline
    echo "NOTE: no W&B credentials found -> WANDB_MODE=offline."
    echo "      Authenticate on a login node with:  wandb login"
    echo "      Or sync this run afterwards with:"
    echo "        $PROJ_ROOT/miniforge3/envs/hdr/bin/wandb sync $WANDB_DIR/wandb/offline-run-*"
fi

# ── Preflight ───────────────────────────────────────────────────────────────
echo "=================================================================="
echo "job          : ${PBS_JOBID:-interactive}"
echo "node         : $(hostname)"
echo "workdir      : $(pwd)"
echo "python       : $PYTHON"
echo "dataset      : $HDR_DATASET_DIR"
echo "save folder  : $HDR_SAVE_FOLDER"
echo "wandb mode   : $WANDB_MODE"
echo "started      : $(date)"
echo "=================================================================="

if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: interpreter not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$HDR_DATASET_DIR/train/tensors" ]]; then
    echo "FATAL: missing $HDR_DATASET_DIR/train/tensors" >&2
    exit 1
fi

if [[ -f "$HDR_SAVE_FOLDER/latest.pth" ]]; then
    echo "Found $HDR_SAVE_FOLDER/latest.pth -> training will RESUME."
else
    echo "No checkpoint in $HDR_SAVE_FOLDER -> training starts from scratch."
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi unavailable"
echo "------------------------------------------------------------------"

# ── Run ─────────────────────────────────────────────────────────────────────
# The training script is single-GPU (device "cuda:0"); the other 3 A100s on
# this node sit idle. Converting to DDP would need code changes.
"$PYTHON" train_A100_MoE_two_phase.py
status=$?

echo "------------------------------------------------------------------"
echo "exit status : $status"
echo "finished    : $(date)"

# ── Phase 2 memory warning ──────────────────────────────────────────────────
# train_A100_MoE_two_phase.py notes "batch_sz=1 is the limit for full frames
# on A100-80 GB". Polaris A100s have 40 GB, so PHASE=2 (full-resolution
# fine-tune) will very likely OOM even at batch_sz=1. Phase 1 (512x512 patches,
# batch 8) is the one expected to fit. If Phase 2 OOMs, options are gradient
# checkpointing, tiled/patch-wise fine-tuning, or an 80 GB machine.

exit $status
