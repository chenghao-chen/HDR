#!/bin/bash
# ============================================================================
# scripts/stage_data.sh — copy the datasets to the machine that needs them.
#
#   ./scripts/stage_data.sh --help
#   ./scripts/stage_data.sh --dry-run
#   ./scripts/stage_data.sh                        # Mobile-HDR, auto method
#   ./scripts/stage_data.sh --dataset kalantari2017
#   ./scripts/stage_data.sh --all
#   ./scripts/stage_data.sh --verify               # compare both ends
#
# Why this exists: Polaris and Aurora do not share a filesystem. Polaris sees
# /lus/eagle; Aurora sees /lus/flare; neither mounts the other. A checkout on
# Aurora with HDR_DATASET_DIR pointing at an eagle path simply fails with
# ENOENT, twenty minutes into the queue.
#
# Mobile-HDR is ~12 GB in 246 files. That is small enough that the transfer
# method barely matters, and large enough that you want to do it once.
#
# Methods, in the order --mode auto tries them:
#
#   local   both paths visible from this host. A plain rsync. This is the
#           case when re-staging within one machine.
#   globus  the supported way to move data between ALCF filesystems. Runs in
#           the background on the data-transfer nodes, restarts itself,
#           verifies checksums, and does not care that your ssh session ended.
#   ssh     rsync over ssh to the other machine's login node. Works, holds
#           your terminal open for the duration, and needs whatever
#           authentication that hop requires.
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
DATASETS=("Mobile-HDR")
MODE="auto"
DRY_RUN=0
VERIFY_ONLY=0
DIRECTION=""          # to-aurora | to-polaris

# Source and destination roots. Overridable, because the allocation names on
# the two machines are frequently different awards with different names.
EAGLE_ROOT="${HDR_EAGLE_ROOT:-/lus/eagle/projects/${HDR_PROJECT_POLARIS:-lighthouse-purdue}/${USER}/datasets}"
FLARE_ROOT="${HDR_FLARE_ROOT:-/lus/flare/projects/${HDR_PROJECT_AURORA:-lighthouse-purdue}/${USER}/datasets}"

# Globus collection names. `globus endpoint search ALCF` lists what your
# account can see; set these to the UUIDs it prints to skip the lookup.
GLOBUS_EAGLE="${HDR_GLOBUS_EAGLE:-}"
GLOBUS_FLARE="${HDR_GLOBUS_FLARE:-}"

SSH_HOST_AURORA="${HDR_SSH_AURORA:-aurora.alcf.anl.gov}"
SSH_HOST_POLARIS="${HDR_SSH_POLARIS:-polaris.alcf.anl.gov}"

usage() { sed -n '2,/^# ===/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ── Arguments ───────────────────────────────────────────────────────────────
while (( $# )); do
    case "$1" in
        -h|--help)     usage 0 ;;
        --dataset)     DATASETS=("${2:?}"); shift 2 ;;
        --all)         DATASETS=("Mobile-HDR" "kalantari2017" "i2-2kfps_v1"); shift ;;
        --mode)        MODE="${2:?}"; shift 2 ;;
        --to-aurora)   DIRECTION="to-aurora"; shift ;;
        --to-polaris)  DIRECTION="to-polaris"; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --verify)      VERIFY_ONLY=1; shift ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

# ── Direction ───────────────────────────────────────────────────────────────
# Default to staging *onto* whichever machine we are standing on, which is
# what you want in the overwhelmingly common case: you just logged into
# Aurora and the data is not there.
if [[ -z "$DIRECTION" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/lib/hdr_env.sh"
    case "$(hdr::detect_site)" in
        aurora)  DIRECTION="to-aurora" ;;
        polaris) DIRECTION="to-polaris" ;;
        *)       echo "Cannot tell which way to copy from this host." >&2
                 echo "Pass --to-aurora or --to-polaris." >&2
                 exit 2 ;;
    esac
fi

if [[ "$DIRECTION" == "to-aurora" ]]; then
    SRC_ROOT="$EAGLE_ROOT";  DST_ROOT="$FLARE_ROOT"
    SRC_NAME="polaris/eagle"; DST_NAME="aurora/flare"
    SRC_GLOBUS="$GLOBUS_EAGLE"; DST_GLOBUS="$GLOBUS_FLARE"
    SSH_HOST="$SSH_HOST_POLARIS"     # pull from Polaris while sitting on Aurora
else
    SRC_ROOT="$FLARE_ROOT";  DST_ROOT="$EAGLE_ROOT"
    SRC_NAME="aurora/flare"; DST_NAME="polaris/eagle"
    SRC_GLOBUS="$GLOBUS_FLARE"; DST_GLOBUS="$GLOBUS_EAGLE"
    SSH_HOST="$SSH_HOST_AURORA"
fi

echo "=================================================================="
echo "Dataset staging"
echo "  direction : $SRC_NAME  ->  $DST_NAME"
echo "  source    : $SRC_ROOT"
echo "  dest      : $DST_ROOT"
echo "  datasets  : ${DATASETS[*]}"
echo "  mode      : $MODE"
echo "=================================================================="

# ── Verification ────────────────────────────────────────────────────────────
# File count and byte total on both ends. Not a checksum — Globus already
# checksums — but it does catch the two failures that actually happen: a
# transfer that stopped halfway, and a copy of the wrong subdirectory.
summarise() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        echo "missing"
        return
    fi
    local files bytes
    files=$(find "$path" -type f | wc -l)
    bytes=$(du -sb "$path" 2>/dev/null | cut -f1)
    echo "${files} files, $(numfmt --to=iec --suffix=B "${bytes:-0}" 2>/dev/null || echo "${bytes}B")"
}

if (( VERIFY_ONLY )); then
    for ds in "${DATASETS[@]}"; do
        echo
        echo "$ds"
        echo "  source ($SRC_NAME): $(summarise "$SRC_ROOT/$ds")"
        echo "  dest   ($DST_NAME): $(summarise "$DST_ROOT/$ds")"
    done
    echo
    echo "Note: a 'missing' on the far side is expected — the other machine's"
    echo "filesystem is not mounted here. Run --verify on each machine."
    exit 0
fi

# ── Method selection ────────────────────────────────────────────────────────
resolve_mode() {
    [[ "$MODE" != "auto" ]] && { echo "$MODE"; return; }
    # Both ends visible? Then it is a local copy, whatever the labels say.
    if [[ -d "$SRC_ROOT" && -d "$(dirname "$DST_ROOT")" ]]; then
        echo "local"; return
    fi
    if command -v globus >/dev/null 2>&1; then echo "globus"; return; fi
    echo "ssh"
}
MODE="$(resolve_mode)"
echo "Using method: $MODE"
echo

run() {
    if (( DRY_RUN )); then
        printf 'DRY RUN: %s\n' "$*"
    else
        "$@"
    fi
}

# ── local ───────────────────────────────────────────────────────────────────
stage_local() {
    command -v rsync >/dev/null 2>&1 || { echo "rsync not found" >&2; return 1; }
    local ds status=0
    for ds in "${DATASETS[@]}"; do
        if [[ ! -d "$SRC_ROOT/$ds" ]]; then
            echo "SKIP $ds — not at $SRC_ROOT/$ds" >&2
            status=1
            continue
        fi
        echo ">>> $ds"
        run mkdir -p "$DST_ROOT"
        # Trailing slash on the source copies the contents into <dst>/<ds>,
        # not into <dst>/<ds>/<ds>.
        run rsync -aH --info=progress2 --partial \
            "$SRC_ROOT/$ds/" "$DST_ROOT/$ds/" || status=1
    done
    return $status
}

# ── globus ──────────────────────────────────────────────────────────────────
stage_globus() {
    if ! command -v globus >/dev/null 2>&1; then
        cat >&2 <<'MSG'
The globus CLI is not on PATH.

Either install it into any python environment:
    pip install --user globus-cli
    globus login

or run the transfer from the web UI at https://app.globus.org/file-manager
using the paths this script printed above.
MSG
        return 1
    fi

    if [[ -z "$SRC_GLOBUS" || -z "$DST_GLOBUS" ]]; then
        cat >&2 <<MSG
Globus collection IDs are not set.

Find them:
    globus endpoint search 'ALCF Eagle'
    globus endpoint search 'ALCF Flare'

then re-run with:
    HDR_GLOBUS_EAGLE=<uuid> HDR_GLOBUS_FLARE=<uuid> $0 ${DATASETS[*]/#/--dataset }

The IDs are stable, so this is a one-time lookup worth putting in your shell
profile.
MSG
        return 1
    fi

    local ds status=0
    for ds in "${DATASETS[@]}"; do
        echo ">>> $ds"
        # --recursive for a directory; --sync-level checksum makes a re-run
        # after a partial transfer copy only what differs, which is the whole
        # reason to prefer Globus for 12 GB over a link that might drop.
        run globus transfer --recursive --sync-level checksum \
            --label "hdr-$ds-$(date +%Y%m%d-%H%M)" \
            "$SRC_GLOBUS:$SRC_ROOT/$ds" \
            "$DST_GLOBUS:$DST_ROOT/$ds" || status=1
    done
    if (( ! DRY_RUN )); then
        echo
        echo "Transfers are queued and run on the data-transfer nodes; you can"
        echo "log out. Watch them with:"
        echo "    globus task list"
        echo "    globus task wait <task-id>"
    fi
    return $status
}

# ── ssh ─────────────────────────────────────────────────────────────────────
stage_ssh() {
    local ds status=0
    echo "Pulling from $SSH_HOST. This holds the terminal open for the whole"
    echo "transfer and will need whatever authentication that hop requires."
    echo
    for ds in "${DATASETS[@]}"; do
        echo ">>> $ds"
        run mkdir -p "$DST_ROOT"
        # -z is deliberately absent: these are compressed tensor files, so
        # compression costs CPU and saves nothing.
        run rsync -aH --info=progress2 --partial \
            "$SSH_HOST:$SRC_ROOT/$ds/" "$DST_ROOT/$ds/" || status=1
    done
    return $status
}

case "$MODE" in
    local)  stage_local ;;
    globus) stage_globus ;;
    ssh)    stage_ssh ;;
    *)      echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
status=$?

echo
if (( status == 0 )); then
    echo "Staging finished. Confirm with:"
    echo "    ./scripts/stage_data.sh --verify"
    echo "    $SCRIPT_DIR/../run_tests.sh -m dataset"
else
    echo "Staging reported errors (exit $status)." >&2
fi
exit $status
