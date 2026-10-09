#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
if (( $# > 1 )); then echo 'Usage: bash run_all.sh [shared_warmup_checkpoint_dir]' >&2; exit 2; fi
cd -- "$REPO"
CKPT=${1:-checkpoints/json_w384_B128_warmup1000_gamma_lr}
# Preserve the host's device ordering; ELR_GPU_IDS can explicitly select eight GPUs.
IFS=',' read -r -a GPUS <<< "${ELR_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
if (( ${#GPUS[@]} != 8 )); then
  echo 'run_all.sh requires exactly eight GPU IDs (ELR_GPU_IDS or CUDA_VISIBLE_DEVICES).' >&2; exit 2
fi
for (( i=0; i<8; i++ )); do
  if [[ -z "${GPUS[i]}" ]]; then echo 'Empty GPU ID.' >&2; exit 2; fi
  for (( j=0; j<i; j++ )); do
    if [[ "${GPUS[i]}" == "${GPUS[j]}" ]]; then echo 'Duplicate GPU ID.' >&2; exit 2; fi
  done
done
if [[ ! -e "$CKPT" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" bash "$HERE/run_prefix.sh" "$CKPT"
fi
# Refuse a multi-rank warmup before launching any branches.
python3 - "$CKPT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
m = json.loads((p/'complete.json').read_text())
files = ['rank00000.pt']
if (m.get('version'), m.get('step'), m.get('world_size')) != (1, 1000, 1):
    raise SystemExit('Warmup must be a complete step1000 / single-rank checkpoint; regenerate old multi-rank warmup.')
if m.get('files') != files or not all((p/f).is_file() for f in files):
    raise SystemExit('Shared warmup rank files are incomplete.')
PY
PIDS=()
cleanup() {
  for PID in "${PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done
  for PID in "${PIDS[@]}"; do wait "$PID" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CASES=(C E A M EA EM AM EAM)
for SLOT in 0 1 2 3 4 5 6 7; do
  CASE=${CASES[SLOT]}
  GPU=${GPUS[SLOT]}
  printf 'Starting %s on GPU %s\n' "$CASE" "$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" ELR_FORK_CHECKPOINT="$CKPT" bash "$HERE/run_case.sh" "$CASE" &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then FAILED=1; fi
done
PIDS=()
if (( FAILED )); then echo 'A branch failed.' >&2; exit 1; fi
