#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if (( $# != 0 )); then echo 'Usage: bash run_all.sh (all eight runs start fresh)' >&2; exit 2; fi
for CASE in C E A M EA EM AM EAM; do
  bash "$HERE/run_case.sh" "$CASE"
done
