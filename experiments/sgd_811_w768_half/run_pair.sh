#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${DATA_DIR:?Set DATA_DIR to the folder containing prepared FineWeb bin shards}"
python train.py --updates 301728 --lr 0.01 --batch 4 \
  --train "$DATA_DIR/fineweb_train_*.bin" \
  --val "$DATA_DIR/fineweb_val_*.bin" \
  --output "${OUTPUT_DIR:-runs/sgd_811_w768_eta0p01_B4_half}" "$@"
