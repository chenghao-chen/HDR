#!/bin/bash -l
# ============================================================================
# One rank of the architecture sweep's evaluation pass: benchmark the
# checkpoint belonging to this rank's manifest row.
#
# Launched by arch_sweep_eval.pbs through mpiexec. Like the training worker it
# always exits 0, so one missing or broken checkpoint cannot tear down the
# other 79 evaluations.
# ============================================================================
set -uo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 0

MANIFEST="${HDR_SWEEP_MANIFEST:?HDR_SWEEP_MANIFEST not set}"
STATUS_DIR="${HDR_SWEEP_STATUS_DIR:?HDR_SWEEP_STATUS_DIR not set}"

RANK="${PMI_RANK:-${PALS_RANKID:-0}}"
LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}"

ROW="$(sed -n "$((RANK + 2))p" "$MANIFEST")"
# Strip a trailing CR: a manifest saved with CRLF endings would otherwise
# leave one inside the last field, which becomes an exported environment
# variable and breaks the trainer with an unreadable parse error.
ROW="${ROW%$'\r'}"
[[ -z "$ROW" ]] && exit 0

IFS=',' read -r RUN FOLDER _REST <<< "$ROW"

LOG="logs/${PBS_JOBID:-interactive}_archeval_${RUN}.log"
STATUS="${STATUS_DIR}/${RUN}.status"
CKPT="${FOLDER}/phase1_best.pth"

if [[ ! -f "$CKPT" ]]; then
    echo "SKIPPED" > "$STATUS"
    echo "rank $RANK: $RUN has no checkpoint at $CKPT"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
export HDR_CHECKPOINT="$CKPT"
export OMP_NUM_THREADS="${HDR_SWEEP_OMP_THREADS:-8}"

source "${PBS_O_WORKDIR:-$(pwd)}/scripts/lib/hdr_env.sh"
hdr::init >/dev/null 2>&1

{
    echo "=== rank $RANK  run=$RUN  gpu=$LOCAL_RANK  host=$(hostname) ==="
    echo "checkpoint: $CKPT"
    echo "started: $(date)"
    echo "==="
} > "$LOG" 2>&1

if "$HDR_PYTHON" test_dual_MoE_two_phase.py >> "$LOG" 2>&1 \
   && "$HDR_PYTHON" scripts/save_test_outputs.py --checkpoint "$CKPT" >> "$LOG" 2>&1; then
    echo "OK" > "$STATUS"
    echo "rank $RANK: $RUN OK"
else
    code=$?
    echo "FAILED $code" > "$STATUS"
    echo "rank $RANK: $RUN FAILED (exit $code), see $LOG"
fi

exit 0
