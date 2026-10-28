#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA_DIR:?Set DATA_DIR to FineWeb shard directory}"
python train.py --train "$DATA_DIR/fineweb_train_*.bin" --val "$DATA_DIR/fineweb_val_*.bin" \
  --arm sine_p2 --output "${OUTPUT_ROOT:-runs/sgd_wsd_elr}/sine_p2" "$@"
