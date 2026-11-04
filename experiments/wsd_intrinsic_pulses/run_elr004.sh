#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" train_intrinsic_pulse.py --experiment elr004 "$@"
