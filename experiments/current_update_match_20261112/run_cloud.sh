#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${1:-$PWD}"
cd -- "$repo_dir"
python - "$package_dir" <<'PY'
import glob
import gzip
import json
import pathlib
import sys
import torch
package = pathlib.Path(sys.argv[1])
reference = next(package.glob('reference_*.jsonl.gz'))
with gzip.open(reference, 'rt') as stream:
    records = [json.loads(line) for line in stream]
steps = {r['update_step'] for r in records}
assert set(range(1001, 20401)) <= steps, 'Incomplete reference'
assert all('target_current_lr_update_over_norm' in r for r in records)
assert glob.glob('data/fineweb10B/fineweb_train_*.bin'), 'Missing original training shards'
assert glob.glob('data/fineweb10B/fineweb_val_*.bin'), 'Missing original validation shards'
assert torch.cuda.is_available(), 'CUDA unavailable'
print('Reference and data found. Target uses interpolated four-step baseline measurements.')
print('PyTorch', torch.__version__, 'CUDA devices', torch.cuda.device_count())
PY
mkdir -p logs
output="logs/current_update_match_$(date +%Y%m%d_%H%M%S).console.log"
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
    "$package_dir/train_current_update_match.py" 2>&1 | tee "$output"
