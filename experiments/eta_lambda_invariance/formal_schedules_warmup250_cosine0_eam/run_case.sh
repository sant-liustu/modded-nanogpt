#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
CASE=${1:?Usage: bash run_case.sh C|E|A|M|EA|EM|AM|EAM [runner arguments]}
case "$CASE" in C|E|A|M|EA|EM|AM|EAM) ;; *) echo "Unknown case: $CASE" >&2; exit 2 ;; esac
shift
cd -- "$REPO"
CKPT=${ELR_FORK_CHECKPOINT:-checkpoints/json_w768_B128_warmup1000}
RESUME_ARGS=(--resume "$CKPT")
for ARG in "$@"; do
  if [[ "$ARG" == --resume || "$ARG" == --resume=* ]]; then
    RESUME_ARGS=()
    CKPT=''
    break
  fi
done
python3 - "$CKPT" <<'PY'
import ast, json, pathlib, sys
p = pathlib.Path('experiments/eta_lambda_invariance/train_gpt2_w768_muonhinit_fixednorm_jsonelr.py')
tree = ast.parse(p.read_text(encoding='utf-8'))
config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Hyperparameters')
seed = next(n.value for n in config.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == 'seed')
if ast.literal_eval(seed) != 0:
    raise SystemExit('This experiment suite requires the shared runner seed to remain 0.')
if sys.argv[1]:
    checkpoint = pathlib.Path(sys.argv[1])
    manifest = json.loads((checkpoint/'complete.json').read_text())
    if manifest.get('version') != 1 or manifest.get('step') != 1000 or manifest.get('world_size') != 2:
        raise SystemExit('Shared fork requires a complete step1000 / two-rank checkpoint.')
    files = ['rank00000.pt', 'rank00001.pt']
    if manifest.get('files') != files or not all((checkpoint/f).is_file() for f in files):
        raise SystemExit('Shared warmup checkpoint rank files are incomplete.')
PY
printf 'Starting %s, seed=0, using %s (checkpoint options forwarded to runner)\n' "$CASE" "$HERE/$CASE.json"
exec torchrun --standalone --nproc_per_node=2 \
  experiments/eta_lambda_invariance/train_gpt2_w768_muonhinit_fixednorm_jsonelr.py \
  --schedule-json "$HERE/$CASE.json" "${RESUME_ARGS[@]}" "$@"
