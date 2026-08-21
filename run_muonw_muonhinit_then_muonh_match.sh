#!/usr/bin/env bash
# Run the paired MuonW->MuonH-initialization experiment end to end.
# Invoke with `bash run_muonw_muonhinit_then_muonh_match.sh`, not with torchrun:
# this coordinator launches the two DDP jobs itself.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${RANK:-}" || -n "${LOCAL_RANK:-}" ]]; then
  echo "Run this coordinator with bash, not from inside torchrun." >&2
  exit 2
fi

# The paired scripts are B=128/devB=64. Two processes reproduce the intended
# one-sequence-per-rank setup; use 1 only for a single-GPU run.
NPROC_PER_JOB="${NPROC_PER_JOB:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
RDZV_PORT="${RDZV_PORT:-29640}"
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! "$NPROC_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_JOB must be a positive integer, got: $NPROC_PER_JOB" >&2
  exit 2
fi
if (( 128 % (64 * NPROC_PER_JOB) != 0 )); then
  echo "B=128 and devB=64 require NPROC_PER_JOB=1 or 2; got $NPROC_PER_JOB." >&2
  exit 2
fi

LRNORM_DIR="experiments/lrnorm_match"
MUONW_SCRIPT="$LRNORM_DIR/train_small_batch_muonw_muonhinit_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_wsd_seed00.py"
MUONH_SCRIPT="$LRNORM_DIR/train_muonh_match_muonw_hinit_gamma_B0128_devB064_lr0p0036_wd0_wsd_seed00.py"
REFERENCE_BUILDER="$LRNORM_DIR/build_muonw_pertensor_lrnorm_reference.py"
REFERENCE_PATH="$LRNORM_DIR/reference_rmsnorm_gamma_qknorm_muonw_muonhinit_wd0_wsd_per_tensor_lr_over_norm.jsonl.gz"

BASELINE_RUN_ID="${PIPELINE_ID}_muonw_muonhinit_base"
MATCH_RUN_ID="${PIPELINE_ID}_muonh_match_muonw_muonhinit"
BASELINE_RUN_DIR="logs/${BASELINE_RUN_ID}"
MATCH_RUN_DIR="logs/${MATCH_RUN_ID}"
PIPELINE_LOG_DIR="$LRNORM_DIR/pipeline_logs/$PIPELINE_ID"
RUN_MANIFEST="$PIPELINE_LOG_DIR/manifest.tsv"
LOCK_DIR="$LRNORM_DIR/.muonw_muonhinit_pair.lock"

for required_path in "$MUONW_SCRIPT" "$MUONH_SCRIPT" "$REFERENCE_BUILDER"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file is missing: $required_path" >&2
    exit 2
  fi
done
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN: no training, reference generation, or filesystem changes will occur."
  echo "MuonW run id: $BASELINE_RUN_ID"
  echo "MuonH run id: $MATCH_RUN_ID"
  printf 'MuonW command: CUDA_VISIBLE_DEVICES=%q RUN_ID_OVERRIDE=%q %q --nnodes=1 --nproc_per_node=%q --rdzv_backend=c10d --rdzv_endpoint=%q %q\n' \
    "$GPU_IDS" "$BASELINE_RUN_ID" "$TORCHRUN" "$NPROC_PER_JOB" "127.0.0.1:${RDZV_PORT}" "$MUONW_SCRIPT"
  printf 'Reference command: %q %q --run-dir %q --output %q --reference-experiment %q --source-run-dir-label %q --embed-lr 0.0036 --muon-lr 0.00036\n' \
    "$PYTHON" "$REFERENCE_BUILDER" "$BASELINE_RUN_DIR" "$REFERENCE_PATH" \
    "${PIPELINE_ID}__muonw_muonhinit__B0128__devB064__lr0p0036__muonlr0p00036__wd0__wsd__seed0" "$BASELINE_RUN_DIR"
  printf 'MuonH command: CUDA_VISIBLE_DEVICES=%q RUN_ID_OVERRIDE=%q %q --nnodes=1 --nproc_per_node=%q --rdzv_backend=c10d --rdzv_endpoint=%q %q\n' \
    "$GPU_IDS" "$MATCH_RUN_ID" "$TORCHRUN" "$NPROC_PER_JOB" "127.0.0.1:${RDZV_PORT}" "$MUONH_SCRIPT"
  exit 0
fi
for occupied_path in \
  "$BASELINE_RUN_DIR" "logs/${BASELINE_RUN_ID}.txt" \
  "$MATCH_RUN_DIR" "logs/${MATCH_RUN_ID}.txt" \
  "$REFERENCE_PATH" "$PIPELINE_LOG_DIR"; do
  if [[ -e "$occupied_path" ]]; then
    echo "Refusing to overwrite existing artifact: $occupied_path" >&2
    exit 2
  fi
done
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another MuonW/MuonH pair pipeline appears to be running: $LOCK_DIR" >&2
  exit 2
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$PIPELINE_LOG_DIR"
printf 'phase\trun_id\tscript_or_tool\tartifact\n' > "$RUN_MANIFEST"

run_ddp_stage() {
  local phase="$1"
  local run_id="$2"
  local script="$3"
  local stage_log="$4"

  echo "==> Starting ${phase} (run_id=${run_id}, GPUs=${GPU_IDS})"
  (
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export RUN_ID_OVERRIDE="$run_id"
    "$TORCHRUN" \
      --nnodes=1 \
      --nproc_per_node="$NPROC_PER_JOB" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="127.0.0.1:${RDZV_PORT}" \
      "$script"
  ) 2>&1 | tee "$stage_log"
  printf '%s\t%s\t%s\t%s\n' "$phase" "$run_id" "$script" "logs/${run_id}" >> "$RUN_MANIFEST"
}

require_baseline_telemetry() {
  local required_file
  for required_file in tensor_norm_history.jsonl muonw_update_norm_history.jsonl; do
    if [[ ! -f "$BASELINE_RUN_DIR/$required_file" ]]; then
      echo "MuonW baseline did not produce $BASELINE_RUN_DIR/$required_file" >&2
      exit 1
    fi
  done
}

run_ddp_stage \
  "muonw_baseline" \
  "$BASELINE_RUN_ID" \
  "$MUONW_SCRIPT" \
  "$PIPELINE_LOG_DIR/muonw_baseline.out"
require_baseline_telemetry

echo "==> Building the paired per-tensor MuonW lr/norm reference"
"$PYTHON" "$REFERENCE_BUILDER" \
  --run-dir "$BASELINE_RUN_DIR" \
  --output "$REFERENCE_PATH" \
  --reference-experiment "${PIPELINE_ID}__muonw_muonhinit__B0128__devB064__lr0p0036__muonlr0p00036__wd0__wsd__seed0" \
  --source-run-dir-label "$BASELINE_RUN_DIR" \
  --embed-lr 0.0036 \
  --muon-lr 0.00036 \
  2>&1 | tee "$PIPELINE_LOG_DIR/build_reference.out"
printf '%s\t%s\t%s\t%s\n' "build_reference" "-" "$REFERENCE_BUILDER" "$REFERENCE_PATH" >> "$RUN_MANIFEST"

if [[ ! -s "$REFERENCE_PATH" ]]; then
  echo "Reference builder did not create a non-empty reference: $REFERENCE_PATH" >&2
  exit 1
fi

run_ddp_stage \
  "muonh_matching" \
  "$MATCH_RUN_ID" \
  "$MUONH_SCRIPT" \
  "$PIPELINE_LOG_DIR/muonh_matching.out"

echo
echo "Pipeline completed."
echo "MuonW telemetry: $BASELINE_RUN_DIR"
echo "MuonH telemetry: $MATCH_RUN_DIR"
echo "Reference: $REFERENCE_PATH"
echo "Manifest: $RUN_MANIFEST"
