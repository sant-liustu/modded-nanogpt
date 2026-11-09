#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA=${1:?Supply absolute OpenWebText directory}
OUT=${2:-runs}
shift
if [ "$#" -gt 0 ]; then shift; fi
exec python -u train_fig7.py --data-dir "$DATA" --out "$OUT/ref_wd0" --seed 0 --weight-decay 0.0 "$@"
