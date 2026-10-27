#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Requires the completed shared prefix produced by run_lr.sh or --arm prefix.
bash run_pair.sh --arm batch "$@"
