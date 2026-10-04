#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
if (( $# > 1 )); then echo 'Usage: bash run_all.sh [shared_warmup_checkpoint_dir]' >&2; exit 2; fi
cd -- "$REPO"
CKPT=${1:-checkpoints/json_w768_B128_warmup1000}
if [[ ! -e "$CKPT" ]]; then
  bash "$HERE/run_prefix.sh" "$CKPT"
fi
for CASE in C E A M EA EM AM EAM; do
  ELR_FORK_CHECKPOINT="$CKPT" bash "$HERE/run_case.sh" "$CASE"
done
