#!/bin/bash -l
# ============================================================================
# One rank of the architecture sweep: read this rank's row out of the manifest,
# pin it to a GPU, and train that one configuration.
#
# Launched by arch_sweep.pbs through mpiexec, never by hand. Rank N takes data
# row N (1-based past the header), so the manifest row count and the mpiexec
# rank count have to agree — the PBS script checks that before launching.
#
# This script always exits 0. Under mpiexec a single non-zero rank tears down
# the entire job, which on a 72-rank sweep would mean one bad configuration
# destroying 71 healthy runs. Failures are recorded as a status file per run
# and summarised by the PBS script after the launch returns.
# ============================================================================
set -uo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 0

MANIFEST="${HDR_SWEEP_MANIFEST:?HDR_SWEEP_MANIFEST not set}"
STATUS_DIR="${HDR_SWEEP_STATUS_DIR:?HDR_SWEEP_STATUS_DIR not set}"

# PALS (Polaris' launcher) exports PALS_*; the PMI_* names are kept as aliases.
# Fall back through both so the worker also runs under a plain mpich.
RANK="${PMI_RANK:-${PALS_RANKID:-0}}"
LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}"

# Row RANK+2 of the CSV: +1 for the header, +1 because sed is 1-based.
ROW="$(sed -n "$((RANK + 2))p" "$MANIFEST")"
# Strip a trailing CR: a manifest saved with CRLF endings would otherwise
# leave one inside the last field, which becomes an exported environment
# variable and breaks the trainer with an unreadable parse error.
ROW="${ROW%$'\r'}"
if [[ -z "$ROW" ]]; then
    echo "rank $RANK: no manifest row, nothing to do"
    exit 0
fi

IFS=',' read -r RUN FOLDER TARGET_P ACTUAL_P REL_ERR MODE NUM_EXPERTS \
    DIM NUM_BLOCKS EXPERT_BLOCKS FILM_HIDDEN <<< "$ROW"

LOG="logs/${PBS_JOBID:-interactive}_arch_${RUN}.log"
STATUS="${STATUS_DIR}/${RUN}.status"

export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
export HDR_MODE="$MODE"
export HDR_NUM_EXPERTS="$NUM_EXPERTS"
export HDR_DIM="$DIM"
export HDR_NUM_BLOCKS="$NUM_BLOCKS"
export HDR_EXPERT_BLOCKS="$EXPERT_BLOCKS"
[[ -n "$FILM_HIDDEN" ]] && export HDR_FILM_HIDDEN="$FILM_HIDDEN"
# The trainer resumes from latest.pth inside the save folder. A rehearsal run
# therefore must not write into the folder the real sweep will use, or the
# real run silently continues from a one-epoch checkpoint instead of training
# from scratch — and reports a converged-looking result for a model that was
# never trained. The suffix keeps the two apart.
export HDR_SAVE_FOLDER="${FOLDER}${HDR_SWEEP_FOLDER_SUFFIX:-}/"

# 72 processes opening W&B sessions at once is a failure mode with no upside
# here: the sweep is read back from the checkpoints, not from W&B. Offline
# runs land in the run folder and can be synced later if wanted.
export WANDB_MODE=offline

# Four ranks share a node's 32 cores, so each gets eight. The single-run
# defaults (8 dataloader workers, unpinned OMP) assume a whole node per
# process and would oversubscribe the node four times over here.
export OMP_NUM_THREADS="${HDR_SWEEP_OMP_THREADS:-8}"
export HDR_DATALOADER_WORKERS="${HDR_SWEEP_WORKERS:-4}"

source "${PBS_O_WORKDIR:-$(pwd)}/scripts/lib/hdr_env.sh"
hdr::init >/dev/null 2>&1

{
    echo "=== rank $RANK  run=$RUN  gpu=$LOCAL_RANK  host=$(hostname) ==="
    echo "mode=$MODE K=$NUM_EXPERTS dim=$DIM blocks=$NUM_BLOCKS "\
         "expert_blocks=$EXPERT_BLOCKS film_hidden=${FILM_HIDDEN:-n/a}"
    echo "target=${TARGET_P} actual=${ACTUAL_P} (rel_err=${REL_ERR})"
    echo "folder=$HDR_SAVE_FOLDER"
    echo "started: $(date)"
    echo "==="
} > "$LOG" 2>&1

if "$HDR_PYTHON" train_A100_MoE_two_phase.py >> "$LOG" 2>&1; then
    echo "OK" > "$STATUS"
    echo "rank $RANK: $RUN OK"
else
    code=$?
    echo "FAILED $code" > "$STATUS"
    echo "rank $RANK: $RUN FAILED (exit $code), see $LOG"
fi

exit 0
