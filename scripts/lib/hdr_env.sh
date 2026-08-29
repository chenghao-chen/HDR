#!/bin/bash
# ============================================================================
# scripts/lib/hdr_env.sh — environment bootstrap shared by every job script.
#
# SOURCE this, do not execute it:
#
#     source "$(dirname "$0")/../lib/hdr_env.sh"
#     hdr::init                 # detect the machine and set everything up
#     hdr::preflight || exit 1  # refuse to start on a broken node
#     hdr::run python train_A100_MoE_two_phase.py
#
# What "everything" is:
#   * which machine this is        (polaris / aurora / local)
#   * an interpreter that has torch with the right GPU backend
#   * BLAS thread caps, without which numpy dies at import on a login node
#   * W&B scratch dirs, proxy and online/offline mode
#   * a job-log header naming the node, the device and the run directory
#
# The per-machine differences live in site_polaris.sh / site_aurora.sh. This
# file holds only what both machines agree on.
# ============================================================================

# Guard against double-sourcing: these are all idempotent, but re-running
# module loads is slow and re-printing the banner is confusing.
[[ -n "${_HDR_ENV_SOURCED:-}" ]] && return 0
_HDR_ENV_SOURCED=1

# Resolve our own location so the site files can be found regardless of where
# the calling script was invoked from. BASH_SOURCE[0] is this file even when
# sourced, which is the whole reason for using it over $0.
HDR_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HDR_SCRIPTS_DIR="$(cd "$HDR_LIB_DIR/.." && pwd)"
HDR_REPO_DIR="$(cd "$HDR_SCRIPTS_DIR/.." && pwd)"
export HDR_REPO_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
hdr::log()  { printf '%s\n' "$*"; }
hdr::warn() { printf 'WARNING: %s\n' "$*" >&2; }
hdr::die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

hdr::rule() { printf '%s\n' "------------------------------------------------------------------"; }
hdr::head() { printf '%s\n' "=================================================================="; }


# ─────────────────────────────────────────────────────────────────────────────
# Site detection
# ─────────────────────────────────────────────────────────────────────────────
# Mirrors hdr_platform.site.detect_site. Duplicated in shell on purpose: this
# runs before any interpreter is known, which is the thing it has to choose.
hdr::detect_site() {
    if [[ -n "${HDR_SITE:-}" ]]; then
        printf '%s\n' "$HDR_SITE"
        return 0
    fi

    local host="${HOSTNAME:-$(hostname 2>/dev/null)}"
    case "$host" in
        polaris*)        printf 'polaris\n'; return 0 ;;
        aurora*|uan-*)   printf 'aurora\n';  return 0 ;;
    esac

    # Compute nodes on both machines are named x4204c3s3b0n0 and give nothing
    # away, so fall back to a filesystem that exists on only one of them.
    if [[ -d /lus/flare ]]; then printf 'aurora\n';  return 0; fi
    if [[ -d /lus/eagle ]]; then printf 'polaris\n'; return 0; fi

    # A PBS job inherits the submitting host, which is always a login node.
    case "${PBS_O_HOST:-}" in
        polaris*)      printf 'polaris\n'; return 0 ;;
        aurora*|uan-*) printf 'aurora\n';  return 0 ;;
    esac

    printf 'local\n'
}


# ─────────────────────────────────────────────────────────────────────────────
# Modules
# ─────────────────────────────────────────────────────────────────────────────
# `module` is a shell function defined by the login profile. A PBS script with
# `#!/bin/bash -l` gets it; anything else may not, so make it available rather
# than failing with "module: command not found" halfway through setup.
hdr::ensure_modules() {
    if declare -F module >/dev/null 2>&1 || command -v module >/dev/null 2>&1; then
        return 0
    fi
    local init
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                /usr/share/lmod/lmod/init/bash /opt/lmod/lmod/init/bash; do
        if [[ -r "$init" ]]; then
            # shellcheck disable=SC1090
            source "$init" && return 0
        fi
    done
    hdr::warn "no module system found; site module loads will be skipped"
    return 1
}


# ─────────────────────────────────────────────────────────────────────────────
# Threads
# ─────────────────────────────────────────────────────────────────────────────
# Without a cap, OpenBLAS spawns one thread per reported core — 256 on a
# Polaris login node, 208 on an Aurora node — hits RLIMIT_NPROC and dies with
# "blas_thread_init: pthread_create failed". That is a crash on `import numpy`,
# not a slowdown.
#
# The number also has to leave room for the DataLoader: workers inherit it, so
# Phase 1's 8 workers at 4 threads each is 32 threads against a Polaris node's
# 32 physical cores, with the main process feeding the GPU on top.
hdr::setup_threads() {
    local n="${1:-${HDR_THREADS:-4}}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$n}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$n}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$n}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$n}"
    export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$n}"
}


# ─────────────────────────────────────────────────────────────────────────────
# Python
# ─────────────────────────────────────────────────────────────────────────────
hdr::setup_python() {
    export PYTHONUNBUFFERED=1
    # An inherited PYTHONPATH shadows the environment's own packages — the
    # miniforge launcher warns about exactly this — and on Aurora it can put a
    # CPU-only torch ahead of the frameworks one.
    unset PYTHONPATH

    [[ -x "${HDR_PYTHON:-}" ]] || hdr::die \
        "interpreter not found: ${HDR_PYTHON:-<unset>}
Run scripts/setup/${HDR_SITE}_env.sh to build it, or set HDR_PYTHON."
}

# Run a python module/script with the site interpreter.
hdr::python() { "$HDR_PYTHON" "$@"; }


# ─────────────────────────────────────────────────────────────────────────────
# Weights & Biases
# ─────────────────────────────────────────────────────────────────────────────
# Compute nodes on both machines reach the internet only through the ALCF
# proxy. Two failure modes this avoids:
#   * no proxy      -> wandb blocks on connect, then falls back to offline
#                      several minutes into the run
#   * /home scratch -> W&B artifacts blow through the small home quota
hdr::setup_wandb() {
    if [[ -n "${HDR_PROXY:-}" ]]; then
        export HTTP_PROXY="$HDR_PROXY"  HTTPS_PROXY="$HDR_PROXY"
        export http_proxy="$HDR_PROXY"  https_proxy="$HDR_PROXY"
        export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"
        export NO_PROXY="$no_proxy"
    fi

    export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HDR_PROJ_ROOT/wandb_cache}"
    export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$HDR_PROJ_ROOT/wandb_data}"
    export WANDB_DIR="${WANDB_DIR:-$HDR_PROJ_ROOT/wandb_runs}"
    mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_DIR" 2>/dev/null || true

    # Credentials come from .env (WANDB_API_KEY) or `wandb login` (~/.netrc).
    # .env is gitignored; never commit it.
    if [[ -f "$HDR_REPO_DIR/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$HDR_REPO_DIR/.env"
        set +a
    fi

    # Only choose online when auth actually exists — otherwise every W&B call
    # blocks on a network round trip that cannot succeed.
    if [[ -n "${WANDB_MODE:-}" ]]; then
        hdr::log "W&B  : WANDB_MODE preset to '$WANDB_MODE' by caller."
    elif [[ -n "${WANDB_API_KEY:-}" ]]; then
        export WANDB_MODE=online
        hdr::log "W&B  : WANDB_API_KEY found -> online."
    elif grep -qs 'machine[[:space:]]\+api\.wandb\.ai' "${NETRC:-$HOME/.netrc}"; then
        # ~/.netrc is on /home, which is in -l filesystems, so nodes can read it.
        export WANDB_MODE=online
        hdr::log "W&B  : ~/.netrc (wandb login) -> online."
    else
        export WANDB_MODE=offline
        hdr::log "W&B  : no credentials -> offline. Authenticate on a login node"
        hdr::log "       with 'wandb login', or sync afterwards:"
        hdr::log "         $HDR_PYTHON -m wandb sync $WANDB_DIR/wandb/offline-run-*"
    fi
}


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────
hdr::init() {
    HDR_SITE="$(hdr::detect_site)"
    export HDR_SITE

    local site_file="$HDR_LIB_DIR/site_${HDR_SITE}.sh"
    if [[ -r "$site_file" ]]; then
        # shellcheck disable=SC1090
        source "$site_file"
        hdr::site_setup
    elif [[ "$HDR_SITE" == "local" ]]; then
        # No cluster: use whatever python is on PATH and the checkout itself.
        HDR_PYTHON="${HDR_PYTHON:-$(command -v python3 || command -v python)}"
        HDR_PROJ_ROOT="${HDR_PROJ_ROOT:-$HDR_REPO_DIR}"
        HDR_SMI=""
        hdr::setup_threads 4
    else
        hdr::die "no site file for '$HDR_SITE' at $site_file"
    fi

    export HDR_PYTHON HDR_PROJ_ROOT
    export HDR_DATASET_DIR="${HDR_DATASET_DIR:-$HDR_PROJ_ROOT/datasets/Mobile-HDR}"

    hdr::setup_python
    hdr::setup_wandb

    # Job scripts cd here, so relative paths in the python code (run folders,
    # test_results/) resolve the same way from a login node and a compute node.
    cd "${PBS_O_WORKDIR:-$HDR_REPO_DIR}" || hdr::die "cannot cd to work directory"
    mkdir -p logs
}


# ─────────────────────────────────────────────────────────────────────────────
# Banner and preflight
# ─────────────────────────────────────────────────────────────────────────────
hdr::banner() {
    hdr::head
    hdr::log "job        : ${PBS_JOBID:-interactive}"
    hdr::log "site       : $HDR_SITE"
    hdr::log "node       : $(hostname)"
    hdr::log "workdir    : $(pwd)"
    hdr::log "python     : $HDR_PYTHON"
    hdr::log "dataset    : $HDR_DATASET_DIR"
    [[ -n "${HDR_SAVE_FOLDER:-}" ]] && hdr::log "save folder: $HDR_SAVE_FOLDER"
    [[ -n "${HDR_CHECKPOINT:-}"  ]] && hdr::log "checkpoint : $HDR_CHECKPOINT"
    hdr::log "wandb mode : ${WANDB_MODE:-unset}"
    hdr::log "threads    : OMP=$OMP_NUM_THREADS"
    hdr::log "started    : $(date)"
    hdr::head
    if [[ -n "${HDR_SMI:-}" ]]; then
        eval "$HDR_SMI" 2>/dev/null || hdr::warn "'$HDR_SMI' unavailable"
    fi
    hdr::rule
}

# Ten seconds of checks against twenty minutes of queue time plus a stack
# trace from inside a .to(device) call. --skip lets short jobs (the pytest
# suite) opt out of the dataset check.
hdr::preflight() {
    hdr::log "Running environment checks..."
    "$HDR_PYTHON" -m hdr_platform.doctor "$@"
    local status=$?
    hdr::rule
    if (( status != 0 )); then
        hdr::warn "environment check reported failures (exit $status)"
    fi
    return $status
}

# Run the payload, timing it and reporting the exit status the way every job
# log in this project does.
hdr::run() {
    local t0=$SECONDS
    "$@"
    local status=$?
    hdr::rule
    hdr::log "command    : $*"
    hdr::log "exit status: $status"
    hdr::log "elapsed    : $(( (SECONDS - t0) / 60 ))m $(( (SECONDS - t0) % 60 ))s"
    hdr::log "finished   : $(date)"
    return $status
}
