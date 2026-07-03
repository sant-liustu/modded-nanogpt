#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# B128/devB64 scripts should run with 2 processes each. On an 8-GPU node this
# launches four independent jobs per phase, each pinned to a 2-GPU slice.
NPROC_PER_JOB="${NPROC_PER_JOB:-2}"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
LRNORM_DIR="experiments/lrnorm_match"
REFERENCE_BUILDER="$LRNORM_DIR/build_muonw_lrnorm_reference.py"
PERTENSOR_REFERENCE_BUILDER="$LRNORM_DIR/build_muonw_pertensor_lrnorm_reference.py"
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_MANIFEST="$LRNORM_DIR/muonw_lrnorm_pipeline_runs_${PIPELINE_ID}.tsv"
PIPELINE_LOG_DIR="$LRNORM_DIR/pipeline_logs/${PIPELINE_ID}"
LOCK_DIR="$LRNORM_DIR/.muonw_lrnorm_pipeline.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another MuonW lrnorm pipeline appears to be running: ${LOCK_DIR}" >&2
  echo "If this is stale, remove the lock directory after confirming no pipeline is active." >&2
  exit 1
fi

PIDS=()
cleanup() {
  local exit_code=$?
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
  rm -rf "$LOCK_DIR"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

mkdir -p "$PIPELINE_LOG_DIR"
echo -e "phase\tlabel\tgpu_ids\trun_id\tscript_or_tool\tartifact" > "$RUN_MANIFEST"

run_job() {
  local phase="$1"
  local label="$2"
  local gpu_ids="$3"
  local rdzv_port="$4"
  local script="$5"
  local run_id="${PIPELINE_ID}_${label}"
  local run_dir="logs/${run_id}"
  local job_log="${PIPELINE_LOG_DIR}/${label}.out"

  if [[ -e "$run_dir" || -e "${run_dir}.txt" ]]; then
    echo "Refusing to reuse existing log artifact for ${label}: ${run_dir}" >&2
    exit 1
  fi

  echo "==> Starting ${phase}/${label} on GPUs ${gpu_ids}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    export RUN_ID_OVERRIDE="$run_id"
    "$TORCHRUN" \
      --nnodes=1 \
      --nproc_per_node="$NPROC_PER_JOB" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="127.0.0.1:${rdzv_port}" \
      "$script"
  ) >"$job_log" 2>&1 &

  PIDS+=("$!")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$phase" "$label" "$gpu_ids" "$run_id" "$script" "$run_dir" >> "$RUN_MANIFEST"
}

wait_for_phase() {
  local phase="$1"
  local failed=0
  local pid

  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()

  if [[ "$failed" -ne 0 ]]; then
    echo "${phase} failed. Inspect logs in ${PIPELINE_LOG_DIR}" >&2
    exit 1
  fi
  echo "==> ${phase} completed"
}

require_run_outputs() {
  local label="$1"
  local run_dir="$2"

  for file in tensor_norm_history.jsonl muonw_update_norm_history.jsonl; do
    if [[ ! -f "${run_dir}/${file}" ]]; then
      echo "Missing ${file} for ${label}: ${run_dir}" >&2
      exit 1
    fi
  done
}

build_reference() {
  local label="$1"
  local run_dir="$2"
  local output="$3"
  local reference_experiment="$4"

  echo "==> Building ${label} reference from ${run_dir}"
  "$PYTHON" "$REFERENCE_BUILDER" \
    --run-dir "$run_dir" \
    --output "$output" \
    --reference-experiment "$reference_experiment"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "reference" "$label" "-" "-" "$REFERENCE_BUILDER" "$output" >> "$RUN_MANIFEST"
}

build_pertensor_reference() {
  local label="$1"
  local run_dir="$2"
  local output="$3"
  local reference_experiment="$4"
  local step_reference_json="$5"

  echo "==> Building ${label} per-tensor reference from ${run_dir}"
  "$PYTHON" "$PERTENSOR_REFERENCE_BUILDER" \
    --run-dir "$run_dir" \
    --output "$output" \
    --reference-experiment "$reference_experiment" \
    --step-reference-json "$step_reference_json"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "reference" "$label" "-" "-" "$PERTENSOR_REFERENCE_BUILDER" "$output" >> "$RUN_MANIFEST"
}

WD0_WSD_BASE="$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_wsd_seed00.py"
WD0P1_WSD_BASE="$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0p1_wsd_seed00.py"
WD0_COSINE_BASE="$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_cosine_seed00.py"
WD0P1_COSINE_BASE="$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0p1_cosine_seed00.py"

WD0_REFERENCE="$LRNORM_DIR/reference_rmsnorm_gamma_muonw_wd0_cosine_rep01_block_plus_embedding_total_lr_over_norm.jsonl"
WD0P1_REFERENCE="$LRNORM_DIR/reference_rmsnorm_gamma_muonw_wd0p1_cosine_rep01_block_plus_embedding_total_lr_over_norm.jsonl"
WD0_PERTENSOR_REFERENCE="$LRNORM_DIR/reference_rmsnorm_gamma_muonw_wd0_cosine_rep01_block_plus_embedding_per_tensor_lr_over_norm.jsonl"
WD0P1_PERTENSOR_REFERENCE="$LRNORM_DIR/reference_rmsnorm_gamma_muonw_wd0p1_cosine_rep01_block_plus_embedding_per_tensor_lr_over_norm.jsonl"

run_job "phase1_base" "muonw_wd0_wsd_base" "0,1" 29510 "$WD0_WSD_BASE"
run_job "phase1_base" "muonw_wd0p1_wsd_base" "2,3" 29511 "$WD0P1_WSD_BASE"
run_job "phase1_base" "muonw_wd0_cosine_base" "4,5" 29512 "$WD0_COSINE_BASE"
run_job "phase1_base" "muonw_wd0p1_cosine_base" "6,7" 29513 "$WD0P1_COSINE_BASE"
wait_for_phase "phase1_base"

WD0_COSINE_RUN_DIR="logs/${PIPELINE_ID}_muonw_wd0_cosine_base"
WD0P1_COSINE_RUN_DIR="logs/${PIPELINE_ID}_muonw_wd0p1_cosine_base"
require_run_outputs "muonw_wd0_cosine_base" "$WD0_COSINE_RUN_DIR"
require_run_outputs "muonw_wd0p1_cosine_base" "$WD0P1_COSINE_RUN_DIR"

build_reference \
  "muonw_wd0_cosine" \
  "$WD0_COSINE_RUN_DIR" \
  "$WD0_REFERENCE" \
  "rmsnorm_gamma_muonw__B0128__devB064__lr0p0036__muonlr0p00036__wd0__warmup1000__schedcosine__seed0__rep01"
build_reference \
  "muonw_wd0p1_cosine" \
  "$WD0P1_COSINE_RUN_DIR" \
  "$WD0P1_REFERENCE" \
  "rmsnorm_gamma_muonw__B0128__devB064__lr0p0036__muonlr0p00036__wd0p1__warmup1000__schedcosine__seed0__rep01"
build_pertensor_reference \
  "muonw_wd0_cosine_per_tensor" \
  "$WD0_COSINE_RUN_DIR" \
  "$WD0_PERTENSOR_REFERENCE" \
  "rmsnorm_gamma_muonw__B0128__devB064__lr0p0036__muonlr0p00036__wd0__warmup1000__schedcosine__seed0__rep01__per_tensor" \
  "$WD0_REFERENCE"
build_pertensor_reference \
  "muonw_wd0p1_cosine_per_tensor" \
  "$WD0P1_COSINE_RUN_DIR" \
  "$WD0P1_PERTENSOR_REFERENCE" \
  "rmsnorm_gamma_muonw__B0128__devB064__lr0p0036__muonlr0p00036__wd0p1__warmup1000__schedcosine__seed0__rep01__per_tensor" \
  "$WD0P1_REFERENCE"

run_job \
  "phase2_lrnorm" \
  "muonw_wd0_addwd_afterwarmup_cosine_lrnormmatch_totalnorm" \
  "0,1" \
  29520 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_addwd_afterwarmup_cosine_lrnormmatch_totalnorm_seed00.py"
run_job \
  "phase2_lrnorm" \
  "muonw_wd0p1_dropwd_afterwarmup_cosine_lrnormmatch_totalnorm" \
  "2,3" \
  29521 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0p1_dropwd_afterwarmup_cosine_lrnormmatch_totalnorm_seed00.py"
run_job \
  "phase2_lrnorm" \
  "muonw_wd0_match_wd0p1_cosine_lrnormmatch_fromupdate1_totalnorm" \
  "4,5" \
  29522 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_match_wd0p1_cosine_lrnormmatch_fromupdate1_totalnorm_seed00.py"
run_job \
  "phase2_lrnorm" \
  "muonw_wd0p1_match_wd0_cosine_lrnormmatch_fromupdate1_totalnorm" \
  "6,7" \
  29523 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0p1_match_wd0_cosine_lrnormmatch_fromupdate1_totalnorm_seed00.py"
wait_for_phase "phase2_lrnorm"

run_job \
  "phase3_pertensor_lrnorm" \
  "muonw_wd0_addwd_afterwarmup_cosine_lrnormmatch_pertensornorm" \
  "0,1" \
  29530 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_addwd_afterwarmup_cosine_lrnormmatch_pertensornorm_seed00.py"
run_job \
  "phase3_pertensor_lrnorm" \
  "muonw_wd0p1_dropwd_afterwarmup_cosine_lrnormmatch_pertensornorm" \
  "2,3" \
  29531 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0p1_dropwd_afterwarmup_cosine_lrnormmatch_pertensornorm_seed00.py"
run_job \
  "phase3_pertensor_lrnorm" \
  "muonw_wd0_match_wd0p1_cosine_lrnormmatch_fromupdate1_pertensornorm" \
  "4,5" \
  29532 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_match_wd0p1_cosine_lrnormmatch_fromupdate1_pertensornorm_seed00.py"
run_job \
  "phase3_pertensor_lrnorm" \
  "muonw_wd0p1_match_wd0_cosine_lrnormmatch_fromupdate1_pertensornorm" \
  "6,7" \
  29533 \
  "$LRNORM_DIR/train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0p1_match_wd0_cosine_lrnormmatch_fromupdate1_pertensornorm_seed00.py"
wait_for_phase "phase3_pertensor_lrnorm"

echo
echo "Pipeline finished."
echo "Manifest: ${RUN_MANIFEST}"
echo "Per-job logs: ${PIPELINE_LOG_DIR}"
