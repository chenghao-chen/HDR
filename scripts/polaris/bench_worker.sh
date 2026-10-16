#!/bin/bash -l
# One rank of a benchmark sweep. Launched through mpiexec by bench_sweep.pbs
# (one row per rank) or bench_pool.pbs (many rows, few ranks).
#
# Manifest columns: run,benchmark,folder,env
#   env is ';'-separated KEY=VALUE pairs exported verbatim, so any trainer
#   setting (mode, size, lr, seed, schedule, loss weights) is data, not code.
# benchmark selects the trainer: sidd -> train_sidd_raw.py,
# hdr -> train_A100_MoE_two_phase.py.
#
# Work is pulled, not assigned: each rank walks the whole manifest starting
# at its own offset and trains every run it can claim (scripts/polaris/
# claim_lib.sh). The same manifest rows can therefore be queued in several
# jobs, in different queues, and whichever gets nodes first does the work --
# a run is never trained twice at once, and a run abandoned by a preempted
# job is picked up and resumed from its latest.pth.
#
# Always exits 0: a failing configuration must not tear down the others.
set -uo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 0

MANIFEST="${HDR_SWEEP_MANIFEST:?HDR_SWEEP_MANIFEST not set}"
STATUS_DIR="${HDR_SWEEP_STATUS_DIR:?HDR_SWEEP_STATUS_DIR not set}"
RANK="${PMI_RANK:-${PALS_RANKID:-0}}"
LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}"

# shellcheck source=claim_lib.sh
source "${PBS_O_WORKDIR:-$(pwd)}/scripts/polaris/claim_lib.sh"
source "${PBS_O_WORKDIR:-$(pwd)}/scripts/lib/hdr_env.sh"
hdr::init >/dev/null 2>&1

NROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
(( NROWS > 0 )) || exit 0

run_row() {   # $1 = 0-based data row index
    local row run bench folder envs log status script code=0
    row="$(sed -n "$(( $1 + 2 ))p" "$MANIFEST")"; row="${row%$'\r'}"
    [[ -z "$row" ]] && return 0
    IFS=',' read -r run bench folder envs <<< "$row"
    claim::try "$folder" || return 0

    log="logs/${PBS_JOBID:-interactive}_bench_${run}.log"
    status="${STATUS_DIR}/${run}.status"
    case "$bench" in
        sidd) script=train_sidd_raw.py ;;
        hdr)  script=train_A100_MoE_two_phase.py ;;
        *)    echo "FAILED unknown-benchmark" > "$status"
              claim::release "$folder" fail; return 0 ;;
    esac

    # Subshell: one run's environment must not leak into the next.
    (
        export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
        export HDR_SAVE_FOLDER="${folder}/"
        export WANDB_MODE=offline OMP_NUM_THREADS=8
        export HDR_DATALOADER_WORKERS="${HDR_SWEEP_WORKERS:-6}"
        export TMPDIR="${HDR_SWEEP_TMPDIR:-/tmp}"
        IFS=';' read -ra kv <<< "$envs"
        for p in "${kv[@]}"; do [[ -n "$p" ]] && export "$p"; done
        { echo "=== rank $RANK run=$run bench=$bench gpu=$LOCAL_RANK host=$(hostname)"
          echo "env: $envs"; echo "started: $(date)"; echo "==="; } >> "$log" 2>&1
        exec "$HDR_PYTHON" "$script" >> "$log" 2>&1
    ) &
    local pid=$!
    claim::heartbeat "$folder" "$pid" &
    local hb=$!
    wait "$pid" || code=$?
    kill "$hb" 2>/dev/null; wait "$hb" 2>/dev/null

    if (( code == 0 )); then
        echo "OK" > "$status"; claim::release "$folder" ok
        echo "rank $RANK: $run OK"
    else
        echo "FAILED $code" > "$status"; claim::release "$folder" fail
        echo "rank $RANK: $run FAILED ($code)"
    fi
}

# Start at this rank's own offset so ranks spread out, then keep going
# through the rest of the manifest, claiming whatever is still free.
for (( k = 0; k < NROWS; k++ )); do
    run_row $(( (RANK + k) % NROWS ))
done
exit 0
