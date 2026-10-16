#!/bin/bash -l
# One rank of a benchmark sweep. Launched by bench_sweep.pbs through mpiexec.
#
# Manifest columns: run,benchmark,folder,env
#   env is ';'-separated KEY=VALUE pairs exported verbatim, so any trainer
#   setting (mode, size, lr, seed, schedule, loss weights) is data, not code.
# benchmark selects the trainer: sidd -> train_sidd_raw.py,
# hdr -> train_A100_MoE_two_phase.py.
#
# Always exits 0: a failing configuration must not tear down the others.
set -uo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 0

MANIFEST="${HDR_SWEEP_MANIFEST:?HDR_SWEEP_MANIFEST not set}"
STATUS_DIR="${HDR_SWEEP_STATUS_DIR:?HDR_SWEEP_STATUS_DIR not set}"
RANK="${PMI_RANK:-${PALS_RANKID:-0}}"
LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}"

ROW="$(sed -n "$((RANK + 2))p" "$MANIFEST")"
ROW="${ROW%$'\r'}"
[[ -z "$ROW" ]] && exit 0
IFS=',' read -r RUN BENCH FOLDER ENVS <<< "$ROW"

LOG="logs/${PBS_JOBID:-interactive}_bench_${RUN}.log"
STATUS="${STATUS_DIR}/${RUN}.status"

export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
export HDR_SAVE_FOLDER="${FOLDER}/"
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export HDR_DATALOADER_WORKERS="${HDR_SWEEP_WORKERS:-6}"
export TMPDIR="${HDR_SWEEP_TMPDIR:-/tmp}"
IFS=';' read -ra KV <<< "$ENVS"
for kv in "${KV[@]}"; do [[ -n "$kv" ]] && export "$kv"; done

source "${PBS_O_WORKDIR:-$(pwd)}/scripts/lib/hdr_env.sh"
hdr::init >/dev/null 2>&1

{ echo "=== rank $RANK run=$RUN bench=$BENCH gpu=$LOCAL_RANK host=$(hostname)"
  echo "env: $ENVS"; echo "started: $(date)"; echo "==="; } > "$LOG" 2>&1

case "$BENCH" in
    sidd) SCRIPT=train_sidd_raw.py ;;
    hdr)  SCRIPT=train_A100_MoE_two_phase.py ;;
    *)    echo "FAILED unknown-benchmark" > "$STATUS"; exit 0 ;;
esac

if "$HDR_PYTHON" "$SCRIPT" >> "$LOG" 2>&1; then
    echo "OK" > "$STATUS"; echo "rank $RANK: $RUN OK"
else
    code=$?; echo "FAILED $code" > "$STATUS"; echo "rank $RANK: $RUN FAILED ($code)"
fi
exit 0
