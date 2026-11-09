#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
DATA=${1:?Usage: bash run_all.sh /absolute/data/path [output] [gpu0 gpu1 gpu2]}
OUT=${2:-runs}
G0=${3:-0}
G1=${4:-$G0}
G2=${5:-$G0}
mkdir -p "$OUT/logs"
run_job() {
  local name=$1 gpu=$2
  local extra=()
  local run
  case "$name" in
    01*) run=ref_wd0;; 02*) run=ref_wd01;; 03*) run=ref_wd03;;
    04*) run=match01_fp32;; 05*) run=match01_bf16;;
    06*) run=match03_fp32;; 07*) run=match03_bf16;;
  esac
  if [ -f "$OUT/$run/latest.pt" ]; then extra+=(--resume); fi
  echo "Starting $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$name.sh" "$DATA" "$OUT" "${extra[@]}" > "$OUT/logs/$name.log" 2>&1
  local status=$?
  echo "$name exit=$status"
  return "$status"
}
# Each chain owns one GPU. BF16 failures are recorded without blocking other chains.
chain01() {
  run_job 02_ref_wd01 "$G1" || return 1
  local failed=0
  run_job 04_match01_fp32 "$G1" || failed=1
  run_job 05_match01_bf16 "$G1" || failed=1
  return "$failed"
}
chain03() {
  run_job 03_ref_wd03 "$G2" || return 1
  local failed=0
  run_job 06_match03_fp32 "$G2" || failed=1
  run_job 07_match03_bf16 "$G2" || failed=1
  return "$failed"
}
status=0
if [ "$G0" != "$G1" ] && [ "$G0" != "$G2" ] && [ "$G1" != "$G2" ]; then
  run_job 01_ref_wd0 "$G0" & p0=$!
  chain01 & p1=$!
  chain03 & p2=$!
  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
else
  echo "Using sequential execution because GPU IDs overlap"
  run_job 01_ref_wd0 "$G0" || status=1
  chain01 || status=1
  chain03 || status=1
fi
exit "$status"
