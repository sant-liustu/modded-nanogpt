#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
bash run_pair.sh --arm prefix "$@"
bash run_pair.sh --arm lr "$@"
