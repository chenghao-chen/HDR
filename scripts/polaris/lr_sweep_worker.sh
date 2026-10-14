#!/bin/bash -l
# One rank of the two-family learning-rate sweep. Launched by lr_sweep.pbs
# through mpiexec, never by hand.
#
# Dispatches on the manifest's benchmark column: SIDD goes to the RAW->RAW
# trainer, HDR to the joint denoise+demosaic one. Always exits 0, so a
# single failing configuration cannot tear down the other nineteen.
set -uo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 0

MANIFEST="${HDR_SWEEP_MANIFEST:?HDR_SWEEP_MANIFEST not set}"
STATUS_DIR="${HDR_SWEEP_STATUS_DIR:?HDR_SWEEP_STATUS_DIR not set}"

RANK="${PMI_RANK:-${PALS_RANKID:-0}}"
LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}"

ROW="$(sed -n "$((RANK + 2))p" "$MANIFEST")"
ROW="${ROW%$'\r'}"
[[ -z "$ROW" ]] && exit 0

IFS=',' read -r RUN BENCH FAMILY MODE LR DIM ITERS DMODEL FOLDER <<< "$ROW"

LOG="logs/${PBS_JOBID:-interactive}_lrsweep_${BENCH}_${RUN}.log"
STATUS="${STATUS_DIR}/${RUN}.status"

export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
export HDR_MODE="$MODE"
export HDR_SAVE_FOLDER="${FOLDER}/"
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export HDR_DATALOADER_WORKERS="${HDR_SWEEP_WORKERS:-4}"
export TMPDIR="${HDR_SWEEP_TMPDIR:-/tmp}"

if [[ "$MODE" == "usvd" ]]; then
    export HDR_USVD_ITERS="$ITERS"
    export HDR_USVD_DMODEL="$DMODEL"
    export HDR_USVD_SHARE=0
else
    export HDR_DIM="$DIM"
    export HDR_NUM_EXPERTS=2
fi

source "${PBS_O_WORKDIR:-$(pwd)}/scripts/lib/hdr_env.sh"
hdr::init >/dev/null 2>&1

{
    echo "=== rank $RANK  run=$RUN  bench=$BENCH  gpu=$LOCAL_RANK  host=$(hostname) ==="
    echo "family=$FAMILY mode=$MODE lr=$LR dim=$DIM iters=$ITERS dmodel=$DMODEL"
    echo "started: $(date)"
    echo "==="
} > "$LOG" 2>&1

if [[ "$BENCH" == "sidd" ]]; then
    export HDR_SIDD_LR="$LR"
    SCRIPT=train_sidd_raw.py
else
    export HDR_LR="$LR"
    SCRIPT=train_A100_MoE_two_phase.py
fi

if "$HDR_PYTHON" "$SCRIPT" >> "$LOG" 2>&1; then
    echo "OK" > "$STATUS"; echo "rank $RANK: $RUN OK"
else
    code=$?; echo "FAILED $code" > "$STATUS"
    echo "rank $RANK: $RUN FAILED ($code)"
fi
exit 0
