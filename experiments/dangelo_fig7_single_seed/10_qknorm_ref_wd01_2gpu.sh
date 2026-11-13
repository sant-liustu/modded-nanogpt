#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA=${1:?Supply absolute OpenWebText directory}
OUT=${2:-runs}
shift
if [ "$#" -gt 0 ]; then shift; fi
exec torchrun --standalone --nproc_per_node=2 train_fig7.py --data-dir "$DATA" --out "$OUT/qknorm_ref_wd01_2gpu" --seed 0 --qk-norm --micro-batch 128 --batch-size 256 --weight-decay 0.1 "$@"
