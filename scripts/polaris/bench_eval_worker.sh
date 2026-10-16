#!/bin/bash -l
# One rank of the HDR benchmark evaluation: score <folder>/phase1_best.pth on
# the Mobile-HDR test split with EXACTLY the flags that produced the classical
# baseline numbers in test_results/baseline_comparison (scripts/polaris/
# baselines.pbs), so the two land in one comparable table.
#
# phase1_best.pth is selected on the training-batch PSNR, not on the test
# split, so evaluating it here involves no test-set selection.
set -uo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 0
MANIFEST="${HDR_SWEEP_MANIFEST:?}"; STATUS_DIR="${HDR_SWEEP_STATUS_DIR:?}"
RANK="${PMI_RANK:-${PALS_RANKID:-0}}"
LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}"
ROW="$(sed -n "$((RANK + 2))p" "$MANIFEST")"; ROW="${ROW%$'\r'}"
[[ -z "$ROW" ]] && exit 0
IFS=',' read -r RUN BENCH FOLDER _ENVS <<< "$ROW"
STATUS="${STATUS_DIR}/${RUN}.status"
CKPT="${FOLDER}/phase1_best.pth"
OUT="test_results/bench_hdr_eval/${RUN}"
LOG="logs/${PBS_JOBID:-interactive}_benchev_${RUN}.log"
if [[ ! -f "$CKPT" ]]; then echo "SKIPPED" > "$STATUS"; exit 0; fi

export CUDA_VISIBLE_DEVICES="$LOCAL_RANK" OMP_NUM_THREADS=8
export TMPDIR="${HDR_SWEEP_TMPDIR:-/tmp}"
source "${PBS_O_WORKDIR:-$(pwd)}/scripts/lib/hdr_env.sh"
hdr::init >/dev/null 2>&1
mkdir -p "$OUT"
if "$HDR_PYTHON" -m hdr_eval.cli --dataset mobile_hdr --split test \
       --checkpoint "${RUN}=${CKPT}" --reference gbtf \
       --inference tiled --tile 512 --ms-ssim --out "$OUT" > "$LOG" 2>&1; then
    echo "OK" > "$STATUS"; echo "rank $RANK: $RUN OK"
else
    code=$?; echo "FAILED $code" > "$STATUS"; echo "rank $RANK: $RUN FAILED ($code)"
fi
exit 0
