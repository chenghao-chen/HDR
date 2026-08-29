#!/bin/bash -l
#PBS -N hdr_test
#PBS -A lighthouse-purdue
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -o logs/
#PBS -e logs/

# SUPERSEDED: the maintained version of this job is now
#     scripts/polaris/test.pbs      (qsub it, or ./scripts/submit.sh test)
# which shares its environment setup with the Aurora equivalent in
# scripts/aurora/test.pbs. This file still works and is kept as the
# Polaris-only original; see RUNNING.md.

# ============================================================================
# Polaris (ALCF) PBS port of submit_test.sh (Purdue Gilbreth / SLURM).
#
# Uses the debug queue: 1-2 nodes, <=1 h walltime, and it turns around far
# faster than preemptable. Evaluation is short, so it fits comfortably.
#
# Submit:   qsub submit_test_polaris.sh
# Pick a checkpoint explicitly:
#   qsub -v HDR_CHECKPOINT=models_p1_moe_Teacher_MobileHDR_polaris01/phase1_best.pth \
#        submit_test_polaris.sh
# ============================================================================

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1
mkdir -p logs

# ── Paths ───────────────────────────────────────────────────────────────────
PROJ_ROOT=/lus/eagle/projects/lighthouse-purdue/ryanchen
PYTHON="$PROJ_ROOT/miniforge3/envs/hdr/bin/python"

export HDR_DATASET_DIR="${HDR_DATASET_DIR:-$PROJ_ROOT/datasets/Mobile-HDR}"

# Defaults to the run produced by submit_train_polaris.sh. Override with -v.
export HDR_CHECKPOINT="${HDR_CHECKPOINT:-models_p1_moe_Teacher_MobileHDR_polaris01/phase1_best.pth}"

# ── Python runtime ──────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
unset PYTHONPATH
# Cap BLAS threads: unset, OpenBLAS spawns one thread per core and crashes on
# import on Polaris ("pthread_create failed ... RLIMIT_NPROC").
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ── Weights & Biases ────────────────────────────────────────────────────────
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"

export WANDB_CACHE_DIR="$PROJ_ROOT/wandb_cache"
export WANDB_DATA_DIR="$PROJ_ROOT/wandb_data"
export WANDB_DIR="$PROJ_ROOT/wandb_runs"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_DIR"

# Credentials come from either .env (WANDB_API_KEY) or `wandb login` (~/.netrc).
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
    echo "WANDB_MODE preset to '$WANDB_MODE' by caller; leaving it alone."
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=online
    echo "W&B auth: WANDB_API_KEY -> online."
elif grep -qs 'machine[[:space:]]\+api\.wandb\.ai' "${NETRC:-$HOME/.netrc}"; then
    export WANDB_MODE=online
    echo "W&B auth: ~/.netrc (wandb login) -> online."
else
    export WANDB_MODE=offline
    echo "NOTE: no W&B credentials found -> WANDB_MODE=offline. Run 'wandb login'."
fi

# ── Preflight ───────────────────────────────────────────────────────────────
echo "=================================================================="
echo "job         : ${PBS_JOBID:-interactive}"
echo "node        : $(hostname)"
echo "dataset     : $HDR_DATASET_DIR"
echo "checkpoint  : $HDR_CHECKPOINT"
echo "wandb mode  : $WANDB_MODE"
echo "started     : $(date)"
echo "=================================================================="

if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: interpreter not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$HDR_DATASET_DIR/test/tensors/with_gt" ]]; then
    echo "FATAL: missing $HDR_DATASET_DIR/test/tensors/with_gt" >&2
    exit 1
fi
# Fail fast: the original script would load the model, then die on a missing file.
if [[ ! -f "$HDR_CHECKPOINT" ]]; then
    echo "FATAL: checkpoint not found: $HDR_CHECKPOINT" >&2
    echo "Available checkpoints:" >&2
    ls -1 models_*/*.pth 2>/dev/null >&2 || echo "  (none - train first)" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi unavailable"
echo "------------------------------------------------------------------"

# ── Run ─────────────────────────────────────────────────────────────────────
"$PYTHON" test_dual_MoE_two_phase.py
status=$?

echo "------------------------------------------------------------------"
echo "exit status : $status"
echo "finished    : $(date)"
exit $status
