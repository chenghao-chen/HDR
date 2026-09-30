#!/bin/bash -l
# One rank of the USVD learning-rate sweep on SIDD. Launched by
# usvd_lr.pbs through mpiexec, never by hand.
#
# Always exits 0, for the same reason as arch_sweep_worker.sh: under mpiexec
# one non-zero rank tears down the whole job, so a single bad configuration
# would destroy the other fifteen. Failures are recorded per run instead.
set -uo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 0

MANIFEST="${HDR_SWEEP_MANIFEST:?HDR_SWEEP_MANIFEST not set}"
STATUS_DIR="${HDR_SWEEP_STATUS_DIR:?HDR_SWEEP_STATUS_DIR not set}"

RANK="${PMI_RANK:-${PALS_RANKID:-0}}"
LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}"

ROW="$(sed -n "$((RANK + 2))p" "$MANIFEST")"
ROW="${ROW%$'\r'}"
[[ -z "$ROW" ]] && exit 0

IFS=',' read -r RUN LABEL LR ITERS DMODEL SHARE FOLDER <<< "$ROW"

LOG="logs/${PBS_JOBID:-interactive}_usvdlr_${RUN}.log"
STATUS="${STATUS_DIR}/${RUN}.status"

export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
export HDR_MODE=usvd
export HDR_USVD_ITERS="$ITERS"
export HDR_USVD_DMODEL="$DMODEL"
export HDR_USVD_SHARE="$SHARE"
export HDR_SIDD_LR="$LR"
export HDR_SAVE_FOLDER="${FOLDER}/"
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export HDR_DATALOADER_WORKERS="${HDR_SWEEP_WORKERS:-4}"
export TMPDIR="${HDR_SWEEP_TMPDIR:-/tmp}"

source "${PBS_O_WORKDIR:-$(pwd)}/scripts/lib/hdr_env.sh"
hdr::init >/dev/null 2>&1

{
    echo "=== rank $RANK  run=$RUN  gpu=$LOCAL_RANK  host=$(hostname) ==="
    echo "config=$LABEL lr=$LR iters=$ITERS d_model=$DMODEL share=$SHARE"
    echo "started: $(date)"
    echo "==="
} > "$LOG" 2>&1

if "$HDR_PYTHON" train_sidd_raw.py >> "$LOG" 2>&1; then
    echo "OK" > "$STATUS"; echo "rank $RANK: $RUN OK"
else
    code=$?; echo "FAILED $code" > "$STATUS"
    echo "rank $RANK: $RUN FAILED ($code)"
fi
exit 0
