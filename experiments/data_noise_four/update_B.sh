#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m torch.distributed.run --standalone --nproc_per_node=2 "$here/train.py" update B "${OUT_ROOT:-$PWD/logs/data_noise_four}"
