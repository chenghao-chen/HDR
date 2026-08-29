#!/bin/bash
# ============================================================================
# scripts/setup/polaris_env.sh — build the Python environment on Polaris.
#
#   ./scripts/setup/polaris_env.sh            # create or update
#   ./scripts/setup/polaris_env.sh --recreate # delete and rebuild
#
# Run this on a LOGIN NODE. It needs the network (through the ALCF proxy) and
# takes ten to twenty minutes on a first build.
#
# Why miniforge and not `module load conda`:
#   the conda module on Polaris does not work for this project — `module load
#   conda` fails outright — so the environment is a self-managed miniforge
#   install on eagle. That also keeps it off /home, whose quota a torch+CUDA
#   environment would eat on its own.
#
# Result: $HDR_PROJ_ROOT/miniforge3/envs/hdr, which is the interpreter
# scripts/lib/site_polaris.sh points HDR_PYTHON at.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROJ_ROOT="${HDR_PROJ_ROOT:-/lus/eagle/projects/${HDR_PROJECT:-lighthouse-purdue}/${USER}}"
MINIFORGE="$PROJ_ROOT/miniforge3"
ENV_DIR="$MINIFORGE/envs/hdr"
PYTHON_VERSION="${HDR_PYTHON_VERSION:-3.11}"
# cu121 wheels match the CUDA driver on Polaris compute nodes.
TORCH_INDEX="${HDR_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

# Login nodes reach the internet only through the proxy.
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$HTTPS_PROXY}"
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"

# Without this, pip's own numpy import can die during the build on a
# 256-thread login node.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "=================================================================="
echo "Polaris environment setup"
echo "  project root : $PROJ_ROOT"
echo "  miniforge    : $MINIFORGE"
echo "  target env   : $ENV_DIR"
echo "  python       : $PYTHON_VERSION"
echo "=================================================================="

# ── 1. miniforge ────────────────────────────────────────────────────────────
if [[ ! -x "$MINIFORGE/bin/conda" ]]; then
    echo "Installing miniforge into $MINIFORGE ..."
    mkdir -p "$PROJ_ROOT"
    installer="$(mktemp -t miniforge-XXXXXX.sh)"
    trap 'rm -f "$installer"' EXIT
    curl -fsSL -o "$installer" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$installer" -b -p "$MINIFORGE"
else
    echo "miniforge already present."
fi

CONDA="$MINIFORGE/bin/conda"

# ── 2. the env ──────────────────────────────────────────────────────────────
if (( RECREATE )) && [[ -d "$ENV_DIR" ]]; then
    echo "Removing existing env (--recreate) ..."
    "$CONDA" env remove -y -p "$ENV_DIR"
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Creating env with python $PYTHON_VERSION ..."
    "$CONDA" create -y -p "$ENV_DIR" "python=$PYTHON_VERSION"
else
    echo "env already present; updating packages in place."
fi

PY="$ENV_DIR/bin/python"

# ── 3. packages ─────────────────────────────────────────────────────────────
# torch first and from the CUDA index: installing it as a dependency of
# something else pulls the default wheel, which on some mirrors is CPU-only,
# and the failure then looks like "no GPU on this node" inside a job.
echo
echo "Installing torch from $TORCH_INDEX ..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install --index-url "$TORCH_INDEX" torch torchvision

echo
echo "Installing project requirements ..."
"$PY" -m pip install -r "$REPO_DIR/requirements.txt"

echo
echo "Installing test tooling ..."
"$PY" -m pip install pytest

# ── 4. verify ───────────────────────────────────────────────────────────────
echo
echo "------------------------------------------------------------------"
"$PY" - <<'PYEOF'
import torch
print("torch          :", torch.__version__)
print("built for CUDA :", torch.version.cuda)
print("CUDA visible   :", torch.cuda.is_available(),
      "(False on a login node is expected — there are no GPUs here)")
PYEOF

echo
echo "Environment ready: $PY"
echo
echo "Next:"
echo "  ./run_tests.sh                    # CPU suite, here on the login node"
echo "  ./scripts/submit.sh smoke         # GPU smoke test in the debug queue"
