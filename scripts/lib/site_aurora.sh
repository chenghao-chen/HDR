#!/bin/bash
# ============================================================================
# scripts/lib/site_aurora.sh — ALCF Aurora
#
# Sourced by hdr_env.sh when the detected site is aurora. Defines
# hdr::site_setup.
#
# Hardware: 6x Intel Data Center GPU Max 1550 per node, 2 tiles each,
#           128 GB HBM per GPU (64 GB per tile), 2x Xeon CPU Max 9470C
#           (104 physical cores, 208 threads), /lus/flare.
#
# The three things that differ from Polaris and actually break code:
#
#   1. There is no CUDA. Devices are "xpu:0".., torch.cuda.is_available()
#      is False, and torch.amp.autocast(device_type="cuda") raises. The
#      hdr_platform package is what handles this; nothing here needs to.
#
#   2. eagle is not mounted. The datasets have to be staged onto flare
#      first — see scripts/stage_data.sh.
#
#   3. Torch comes from a module, not from a conda env we built. The
#      frameworks module ships PyTorch built against oneAPI with the XPU
#      backend; a pip-installed torch would be CPU-only.
# ============================================================================

# ── Queue facts (documentation for the PBS headers; not used at runtime) ────
# debug          <= 2 nodes,  <= 1 h
# debug-scaling  <= 2 nodes,  <= 1 h
# prod           1+ nodes                accepts single-node jobs, unlike
#                                        Polaris' prod, so long training runs
#                                        do NOT have to be preemptable here
HDR_QUEUE_DEBUG="debug"
HDR_QUEUE_LONG="prod"
HDR_FILESYSTEMS="home:flare"
# Aurora's PBS does not take the system= selector Polaris uses.
HDR_SELECT_EXTRA=""

hdr::site_setup() {
    # The Aurora allocation is a separate award from the Polaris/eagle one and
    # is often named differently. Set HDR_PROJECT (or HDR_PROJ_ROOT outright)
    # if "lighthouse-purdue" is not the name on this machine.
    HDR_PROJ_ROOT="${HDR_PROJ_ROOT:-/lus/flare/projects/${HDR_PROJECT:-lighthouse-purdue}/${USER}}"

    # ── Torch with the XPU backend ──────────────────────────────────────
    # The frameworks module carries PyTorch + Intel Extension for PyTorch +
    # oneCCL, all built against the oneAPI runtime on this machine. Loading it
    # is what makes torch.xpu.is_available() true.
    if hdr::ensure_modules; then
        module use /soft/modulefiles 2>/dev/null || true
        if module load "${HDR_FRAMEWORKS_MODULE:-frameworks}" 2>/dev/null; then
            hdr::log "modules: loaded ${HDR_FRAMEWORKS_MODULE:-frameworks}"
        else
            hdr::warn "could not load the '${HDR_FRAMEWORKS_MODULE:-frameworks}' module."
            hdr::warn "Check 'module avail frameworks' and set HDR_FRAMEWORKS_MODULE"
            hdr::warn "to the exact name, e.g. frameworks/2024.2.1_u1."
        fi
    fi

    # ── Interpreter ─────────────────────────────────────────────────────
    # Default to the module's python. A venv built on top of it
    # (scripts/setup/aurora_env.sh) supplies the packages frameworks does not
    # carry — lpips, wandb, opencv — while inheriting its torch.
    if [[ -z "${HDR_PYTHON:-}" && -x "$HDR_PROJ_ROOT/envs/hdr-aurora/bin/python" ]]; then
        HDR_PYTHON="$HDR_PROJ_ROOT/envs/hdr-aurora/bin/python"
        # Activating puts the venv's console scripts (wandb, pytest) on PATH
        # too; the venv is --system-site-packages, so the module's torch is
        # still the one that gets imported.
        # shellcheck disable=SC1091
        source "$HDR_PROJ_ROOT/envs/hdr-aurora/bin/activate"
    fi
    HDR_PYTHON="${HDR_PYTHON:-$(command -v python3 || command -v python)}"

    HDR_PROXY="${HDR_PROXY:-http://proxy.alcf.anl.gov:3128}"
    HDR_SMI="xpu-smi discovery"

    # ── Device hierarchy ────────────────────────────────────────────────
    # Each Max 1550 is two tiles. FLAT exposes each tile as its own XPU
    # device: 12 devices of 64 GB per node. COMPOSITE exposes 6 devices of
    # 128 GB and relies on implicit scaling across the tiles, which is the
    # slower path for a single-device workload like this one.
    #
    # This project trains on one device, so FLAT and 64 GB is the right
    # trade — and 64 GB is still more than a Polaris A100's 40 GB, which is
    # why the full-resolution Phase 2 that OOMs there is expected to fit here.
    export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"

    # 104 physical cores; the same workers x threads budget as Polaris, with
    # more room. 8 workers x 8 threads = 64, comfortably inside the node.
    hdr::setup_threads "${HDR_THREADS:-8}"

    # ── Runtime tuning ──────────────────────────────────────────────────
    # Immediate command lists cut kernel-launch latency for the many small
    # ops this model issues; it is the setting ALCF recommends for PyTorch.
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS="${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS:-1}"
    # Keep the oneAPI JIT cache off /home, whose quota is small: a torch run
    # can write hundreds of MB of compiled kernels here.
    export SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-1}"
    export SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-$HDR_PROJ_ROOT/.sycl_cache}"
    mkdir -p "$SYCL_CACHE_DIR" 2>/dev/null || true
}
