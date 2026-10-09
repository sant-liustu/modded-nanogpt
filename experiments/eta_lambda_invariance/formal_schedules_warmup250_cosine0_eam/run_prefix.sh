#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CKPT=${1:-checkpoints/json_w384_B128_warmup1000_gamma_lr}
if (( $# > 1 )); then echo 'Usage: bash run_prefix.sh [new_checkpoint_dir]' >&2; exit 2; fi
cd -- "$REPO"
# Always expose exactly one device, even on an eight-GPU host.
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
if (( ${#VISIBLE_GPUS[@]} < 1 )) || [[ -z "${VISIBLE_GPUS[0]}" ]]; then
  echo 'This experiment requires one visible GPU.' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[0]}"
exec torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  experiments/eta_lambda_invariance/train_gpt2_w384_muonhinit_fixednorm_jsonelr.py \
  --schedule-json "$HERE/C.json" --checkpoint-dir "$CKPT" --save-at 1000 --stop-after-save
