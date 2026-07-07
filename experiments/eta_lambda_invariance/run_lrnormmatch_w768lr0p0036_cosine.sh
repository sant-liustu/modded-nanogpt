#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -u experiments/eta_lambda_invariance/train_gpt2_mupp_w1536_lr0p0072_cosine_wd0p1_lrnormmatch_w768lr0p0036_wd0p1.py
"${PYTHON_BIN}" -u experiments/eta_lambda_invariance/train_gpt2_mupp_w1536_lr0p0036_cosine_wd0p2_lrnormmatch_w768lr0p0036_wd0p1.py
