#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CASE=${1:?Usage: bash run_case.sh C|E|A|M|EA|EM|AM|EAM}
case "$CASE" in C|E|A|M|EA|EM|AM|EAM) ;; *) echo "Unknown case: $CASE" >&2; exit 2 ;; esac
if (( $# != 1 )); then echo 'Each experiment starts fresh; no checkpoint argument is accepted.' >&2; exit 2; fi
cd -- "$REPO"
python3 - <<'PY'
import ast, pathlib
p = pathlib.Path('experiments/eta_lambda_invariance/train_gpt2_w256_muonhinit_fixednorm_jsonelr_chord_lca.py')
tree = ast.parse(p.read_text(encoding='utf-8'))
config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Hyperparameters')
seed = next(n.value for n in config.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == 'seed')
if ast.literal_eval(seed) != 0:
    raise SystemExit('This experiment suite requires the shared runner seed to remain 0.')
PY
printf 'Starting %s from step 0, seed=0, using %s\n' "$CASE" "$HERE/$CASE.json"
exec torchrun --standalone --nproc_per_node=2 \
  experiments/eta_lambda_invariance/train_gpt2_w256_muonhinit_fixednorm_jsonelr_chord_lca.py \
  --schedule-json "$HERE/$CASE.json"
