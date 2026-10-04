#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CKPT=${1:-checkpoints/json_w768_B128_warmup1000}
if (( $# > 1 )); then echo 'Usage: bash run_prefix.sh [new_checkpoint_dir]' >&2; exit 2; fi
cd -- "$REPO"
exec torchrun --standalone --nproc_per_node=2 \
  experiments/eta_lambda_invariance/train_gpt2_w768_muonhinit_fixednorm_jsonelr.py \
  --schedule-json "$HERE/C.json" --checkpoint-dir "$CKPT" --save-at 1000 --stop-after-save
