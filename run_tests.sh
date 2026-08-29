#!/bin/bash
# Run the HDR test suite on whichever machine you are on.
#
#   ./run_tests.sh                    # whole suite
#   ./run_tests.sh -k gbtf            # subset
#   ./run_tests.sh tests/test_model_moe.py -v
#   ./run_tests.sh -m "not slow"      # skip the slow ones
#   ./run_tests.sh -m gpu             # only the GPU tests (needs a job)
#
# On a login node — Polaris or Aurora — the gpu-marked tests auto-skip, and
# the dataset-marked ones skip unless the data is staged on this machine.
# To exercise those for real:
#
#   ./scripts/submit.sh smoke         # runs this suite on a compute node
#
# The interpreter and the thread caps come from scripts/lib/hdr_env.sh, which
# knows the difference between the two machines. HDR_PYTHON still overrides.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

# Sourcing hdr_env gives us HDR_PYTHON, the BLAS thread caps (without which
# OpenBLAS dies at import on a 256-thread login node) and HDR_DATASET_DIR.
# It is idempotent, so calling this from inside a job that already ran
# hdr::init costs nothing.
# shellcheck disable=SC1091
source scripts/lib/hdr_env.sh
hdr::init >/dev/null || exit 1

exec "$HDR_PYTHON" -m pytest "$@"
