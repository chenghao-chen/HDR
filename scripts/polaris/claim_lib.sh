# shellcheck shell=bash
# ============================================================================
# Run-level claims, so several jobs -- possibly in different queues -- can
# work through one backlog without ever training the same run twice at once.
#
# Why it exists: the same benchmark runs are queued both as one-node
# preemptable jobs and as a 4-node capacity job, so whichever queue gets nodes
# first does the work. Without claims, two jobs could resume the same run from
# the same latest.pth concurrently and corrupt it.
#
#   claim::try FOLDER        0 = this process now owns the run
#                            1 = run is DONE, or held by a live owner
#   claim::heartbeat FOLDER PID   (background) refresh while PID is alive
#   claim::release FOLDER ok|fail
#
# A claim is a directory under $HDR_CLAIM_DIR (mkdir is atomic, including on
# Lustre). Its heartbeat file is touched every minute while the trainer runs;
# a claim whose heartbeat is older than HDR_CLAIM_STALE_SEC (default 20 min,
# far above the longest epoch in either benchmark) belongs to a job that was
# preempted or killed, and may be taken over -- the trainers resume from
# latest.pth, so the new owner continues rather than restarts. Takeover
# renames the stale claim away first, so of two jobs racing to take the same
# stale claim, exactly one wins the subsequent mkdir.
# ============================================================================

CLAIM_DIR="${HDR_CLAIM_DIR:-claims}"
CLAIM_STALE_SEC="${HDR_CLAIM_STALE_SEC:-1200}"

claim::_path() { echo "${CLAIM_DIR}/$(basename "${1%/}")"; }

claim::done() { [[ -f "${1%/}/DONE" ]]; }

claim::_stamp() {
    echo "job=${PBS_JOBID:-interactive} host=$(hostname) pid=$$ since=$(date)" \
        > "$1/owner"
    touch "$1/heartbeat"
}

claim::_age() {   # seconds since the claim's heartbeat (or creation)
    local c="$1" now; now=$(date +%s)
    if [[ -f "$c/heartbeat" ]]; then
        echo $(( now - $(stat -c %Y "$c/heartbeat" 2>/dev/null || echo "$now") ))
    else
        echo $(( now - $(stat -c %Y "$c" 2>/dev/null || echo "$now") ))
    fi
}

claim::try() {
    local folder="${1%/}" c m
    claim::done "$folder" && return 1
    mkdir -p "$CLAIM_DIR"
    c="$(claim::_path "$folder")"
    if mkdir "$c" 2>/dev/null; then
        claim::_stamp "$c"; return 0
    fi
    (( $(claim::_age "$c") > CLAIM_STALE_SEC )) || return 1

    # Takeover. Checking staleness and then replacing the claim is not
    # atomic: a second racer that also saw "stale" could remove the claim
    # the first one has just re-created, leaving two owners. So takeovers
    # are serialised behind a steal mutex, and staleness is re-checked while
    # holding it -- a late racer then finds the claim fresh and backs off.
    m="${c}.steal"
    if ! mkdir "$m" 2>/dev/null; then
        # A mutex can only outlive its few-millisecond critical section if
        # its holder died inside it; clear such a leftover for next time.
        (( $(claim::_age "$m") > 600 )) && rm -rf "$m"
        return 1
    fi
    local won=1
    if [[ ! -d "$c" ]] || (( $(claim::_age "$c") > CLAIM_STALE_SEC )); then
        rm -rf "$c"
        if mkdir "$c" 2>/dev/null; then claim::_stamp "$c"; won=0; fi
    fi
    rm -rf "$m"
    return $won
}

claim::heartbeat() {
    local c; c="$(claim::_path "$1")"
    while kill -0 "$2" 2>/dev/null; do
        touch "$c/heartbeat" 2>/dev/null
        sleep 60
    done
}

claim::release() {
    local folder="${1%/}" c; c="$(claim::_path "$folder")"
    [[ "$2" == "ok" ]] && mkdir -p "$folder" && touch "$folder/DONE"
    # Only remove the claim if it is still ours: never delete a claim that
    # another job has since taken over.
    grep -q "pid=$$ " "$c/owner" 2>/dev/null && rm -rf "$c"
    return 0
}
