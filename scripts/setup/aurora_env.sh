#!/bin/bash
# ============================================================================
# scripts/setup/aurora_env.sh — build the Python environment on Aurora.
#
#   ./scripts/setup/aurora_env.sh            # create or update
#   ./scripts/setup/aurora_env.sh --recreate # delete and rebuild
#
# Run this on a LOGIN NODE.
#
# The shape of this is different from Polaris', and the difference is the
# whole point:
#
#   torch comes from the `frameworks` module, NOT from pip.
#
# The module ships PyTorch built against this machine's oneAPI runtime with
# the XPU backend compiled in, plus Intel Extension for PyTorch and oneCCL.
# `pip install torch` would fetch a generic wheel with no XPU support, and
# the symptom is not an install error — it is torch.xpu.is_available()
# returning False inside a job, on a node with six working GPUs.
#
# So this creates a venv WITH --system-site-packages on top of the module,
# and installs only the packages the module does not carry. The module's
# torch stays the one that gets imported.
#
# Result: $HDR_PROJ_ROOT/envs/hdr-aurora, which scripts/lib/site_aurora.sh
# picks up automatically.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROJ_ROOT="${HDR_PROJ_ROOT:-/lus/flare/projects/${HDR_PROJECT:-lighthouse-purdue}/${USER}}"
VENV_DIR="${HDR_VENV:-$PROJ_ROOT/envs/hdr-aurora}"
FRAMEWORKS_MODULE="${HDR_FRAMEWORKS_MODULE:-frameworks}"

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$HTTPS_PROXY}"
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=================================================================="
echo "Aurora environment setup"
echo "  project root : $PROJ_ROOT"
echo "  venv         : $VENV_DIR"
echo "  torch from   : module $FRAMEWORKS_MODULE (NOT pip)"
echo "=================================================================="

# ── 1. the frameworks module ────────────────────────────────────────────────
if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        [[ -r "$init" ]] && { source "$init"; break; }
    done
fi

module use /soft/modulefiles 2>/dev/null || true
if ! module load "$FRAMEWORKS_MODULE"; then
    cat >&2 <<MSG

Could not load the '$FRAMEWORKS_MODULE' module.

List what this machine actually has and pass the exact name:
    module use /soft/modulefiles
    module avail frameworks
    HDR_FRAMEWORKS_MODULE=frameworks/<version> $0
MSG
    exit 1
fi
echo "Loaded module: $FRAMEWORKS_MODULE"

BASE_PYTHON="$(command -v python3 || command -v python)"
echo "Module python: $BASE_PYTHON"

# Fail here rather than three hours into a job: if the module's torch has no
# XPU backend, nothing downstream will work and no venv can fix it.
echo
echo "Checking the module's torch ..."
"$BASE_PYTHON" - <<'PYEOF'
import sys
try:
    import torch
except Exception as exc:
    sys.exit(f"FATAL: the frameworks module has no importable torch: {exc!r}")

print("torch            :", torch.__version__)

has_xpu_ns = hasattr(torch, "xpu")
try:
    import intel_extension_for_pytorch as ipex
    print("IPEX             :", ipex.__version__)
except Exception:
    print("IPEX             : not importable "
          "(fine if torch has native XPU support)")

available = bool(has_xpu_ns and torch.xpu.is_available())
count = torch.xpu.device_count() if available else 0
print("torch.xpu        :", "present" if has_xpu_ns else "MISSING")
print("XPU available    :", available, f"({count} devices)")
if not has_xpu_ns:
    sys.exit("FATAL: this torch has no XPU support. Wrong module?")
if not available:
    print("\nNOTE: no XPU visible from this login node. That is normal — "
          "Aurora login nodes have no GPUs.\n"
          "      The smoke job is what verifies the devices for real.")
PYEOF

# ── 2. the venv ─────────────────────────────────────────────────────────────
if (( RECREATE )) && [[ -d "$VENV_DIR" ]]; then
    echo "Removing existing venv (--recreate) ..."
    rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo
    echo "Creating venv (--system-site-packages, so the module's torch is inherited) ..."
    mkdir -p "$(dirname "$VENV_DIR")"
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
else
    echo "venv already present; updating packages in place."
fi

PY="$VENV_DIR/bin/python"

# ── 3. packages, minus torch ────────────────────────────────────────────────
# Filter torch and torchvision out of requirements.txt. Installing either
# would shadow the module's XPU-enabled build with a CPU-only wheel — the
# single most likely way to end up "running on Aurora" at CPU speed.
REQ_FILTERED="$(mktemp -t hdr-req-XXXXXX.txt)"
trap 'rm -f "$REQ_FILTERED"' EXIT
grep -viE '^\s*(torch|torchvision|torchaudio)\s*([<>=!~].*)?$' \
    "$REPO_DIR/requirements.txt" > "$REQ_FILTERED"

echo
echo "Installing (torch/torchvision deliberately excluded):"
sed 's/^/  /' "$REQ_FILTERED"

"$PY" -m pip install --upgrade pip
# --no-deps on nothing here, but pip must not be allowed to pull torch in as a
# transitive dependency either; lpips declares it. It is already satisfied by
# the inherited system site-packages, so pip will leave it alone — this check
# makes sure of it afterwards.
"$PY" -m pip install -r "$REQ_FILTERED"
"$PY" -m pip install pytest

# ── 4. verify the venv did not shadow the module's torch ────────────────────
echo
echo "------------------------------------------------------------------"
"$PY" - <<PYEOF
import sys, torch
print("venv python   :", sys.executable)
print("torch         :", torch.__version__)
print("torch loaded from:", torch.__file__)
if "$VENV_DIR" in torch.__file__:
    sys.exit(
        "\nFATAL: a pip torch was installed into the venv and is now "
        "shadowing the module's XPU build.\n"
        "Fix with:\n"
        "    $PY -m pip uninstall -y torch torchvision\n"
        "then re-run this script.")
print("OK: torch still comes from the frameworks module.")
PYEOF

echo
echo "Environment ready: $PY"
echo
echo "Next:"
echo "  ./scripts/stage_data.sh --help    # eagle is not mounted here"
echo "  ./run_tests.sh                    # CPU suite on the login node"
echo "  ./scripts/submit.sh smoke         # the job that proves XPU works"
