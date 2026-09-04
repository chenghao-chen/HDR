#!/bin/bash
# ============================================================================
# scripts/submit.sh — one command to submit a job on whichever machine you
# are logged into.
#
#   ./scripts/submit.sh smoke                  # do this first, on both
#   ./scripts/submit.sh train
#   ./scripts/submit.sh test
#   ./scripts/submit.sh train -v HDR_SAVE_FOLDER=models_run2/
#   ./scripts/submit.sh test --site aurora --dry-run
#
# It picks scripts/<site>/<job>.pbs, and everything after the job name is
# handed to qsub untouched, so the usual overrides still work:
#
#   -q debug                 different queue
#   -l walltime=02:00:00     different walltime
#   -A other-allocation      different account
#   -v NAME=VALUE            environment for the job
#
# The point of the wrapper is that the Polaris and Aurora job scripts have
# genuinely different PBS headers — queue, filesystems, select syntax — so
# submitting the wrong one fails in a way that is not obvious from the error.
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat >&2 <<'USAGE'
usage: scripts/submit.sh <job> [--site polaris|aurora] [--dry-run] [qsub args...]

jobs:
  smoke         GPU smoke test + full pytest suite       (debug queue, ~20-30 min)
  train         two-phase training                       (long queue, 8 h)
  test          evaluation of a checkpoint                (debug queue, 30 min)
  sweep_experts K in {1,2,3,4}, one GPU each, in parallel  (debug queue, 1 h)

options:
  --site NAME   submit the scripts for NAME instead of the detected machine.
                Only useful with --dry-run: qsub cannot reach the other
                machine's scheduler from here.
  --dry-run     print the qsub command and the resolved PBS headers, submit
                nothing.
  --list        show the available jobs for every site.

Anything else is passed through to qsub verbatim.
USAGE
    exit 2
}

# ── Argument parsing ────────────────────────────────────────────────────────
JOB=""
SITE=""
DRY_RUN=0
QSUB_ARGS=()

while (( $# )); do
    case "$1" in
        -h|--help)  usage ;;
        --list)
            for d in "$SCRIPT_DIR"/*/; do
                [[ -d "$d" ]] || continue
                local_jobs=("$d"*.pbs)
                [[ -e "${local_jobs[0]}" ]] || continue
                printf '%s: %s\n' "$(basename "$d")" \
                    "$(for f in "${local_jobs[@]}"; do basename "$f" .pbs; done | tr '\n' ' ')"
            done
            exit 0 ;;
        --site)     SITE="${2:-}"; shift 2 || usage ;;
        --site=*)   SITE="${1#*=}"; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -*)         QSUB_ARGS+=("$1"); shift ;;
        *)
            if [[ -z "$JOB" ]]; then JOB="$1"; else QSUB_ARGS+=("$1"); fi
            shift ;;
    esac
done

[[ -n "$JOB" ]] || usage

# ── Site ────────────────────────────────────────────────────────────────────
if [[ -z "$SITE" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/lib/hdr_env.sh"
    SITE="$(hdr::detect_site)"
fi

if [[ "$SITE" == "local" ]]; then
    cat >&2 <<'MSG'
This host is not Polaris or Aurora, so there is no scheduler to submit to.

To run the same work in the foreground:
    python scripts/gpu_smoke.py --allow-cpu     # smoke
    ./run_tests.sh                              # test suite
    python train_A100_MoE_two_phase.py          # train
MSG
    exit 1
fi

PBS_SCRIPT="$SCRIPT_DIR/$SITE/$JOB.pbs"
if [[ ! -f "$PBS_SCRIPT" ]]; then
    echo "no such job '$JOB' for site '$SITE'" >&2
    echo "available: $(ls -1 "$SCRIPT_DIR/$SITE"/*.pbs 2>/dev/null |
                       xargs -n1 basename 2>/dev/null |
                       sed 's/\.pbs$//' | tr '\n' ' ')" >&2
    exit 2
fi

# ── Submit ──────────────────────────────────────────────────────────────────
# qsub resolves relative paths against PBS_O_WORKDIR, which is wherever it was
# invoked; the job scripts cd there. Submitting from the repo root keeps the
# run folders and logs/ where every other path in the project expects them.
cd "$REPO_DIR" || exit 1
mkdir -p logs

if (( DRY_RUN )); then
    echo "site   : $SITE"
    echo "script : $PBS_SCRIPT"
    echo "command: qsub ${QSUB_ARGS[*]:-} $PBS_SCRIPT"
    echo
    echo "PBS headers:"
    grep '^#PBS' "$PBS_SCRIPT" | sed 's/^/  /'
    exit 0
fi

if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub not found — is this a login node?" >&2
    exit 1
fi

echo "Submitting $JOB on $SITE: $PBS_SCRIPT"
exec qsub "${QSUB_ARGS[@]}" "$PBS_SCRIPT"
