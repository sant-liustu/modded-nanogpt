#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec torchrun --standalone --nproc_per_node=8 train_fixednorm_w3072.py "$@"
