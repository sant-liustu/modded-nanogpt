# Local end-to-end validation — 2026-10-27

Latest runner tested with width 16, depth 1, two heads, context 16, the
unchanged 50,304-token vocabulary, seed 0, LR 0.01, and 100 canonical updates.
Real prepared FineWeb tokens were copied into tiny fixture shards; a shard
boundary deliberately bisects a context. No formal-budget run was launched.

The fixed integer schedule exercised batches 4, 16, and 40 with NO gradient
accumulation. Validation used 33 contexts with batch 32, exercising a final
one-context partial batch. The test changes model size/budget, not the runner.

Passed:
- Continuous prefix + LR tail + batch tail on CPU and CUDA/BF16.
- Separate subprocess interruptions immediately after saved prefix, LR-tail,
  and batch-tail checkpoints, followed by fresh-process continuation.
- Deterministic CPU resumed weights are bitwise identical to continuous
  weights; counters, RNG state, data position and measurement rows also match
  (excluding elapsed wall time).
- Matched intrinsic-time grid, ratio, token/context counts in both arms.
- A deliberately stale log row beyond the checkpoint is discarded on resume.
- Reopening a completed run; rejecting changed scientific configuration.
- Real `run_lr.sh` then `run_batch.sh` executed through Bash, with tiny CLI
  overrides and the original local FineWeb shard paths.
- Existing `verify.py`: exact model snapshot, formal-budget schedule arithmetic,
  accumulation and validation batching equivalence, checkpoint serialization
  at simulated update 200,000, and no overwrite of that milestone.

Final tiny-run counters: prefix 78 updates / 312 contexts; LR arm 100 updates
/ 400 contexts; batch arm 82 updates / 400 contexts. Both finish at T=0.82.

Reproduce (output directory must not exist):

```bash
python verify.py
python test_end_to_end.py --output /scratch/sgd_811_e2e \
  --train-source /data/fineweb_train_000001.bin \
  --val-source /data/fineweb_val_000000.bin
```

The output directory contains report.json, subprocess logs, fixture data and
checkpoints. Tests do not alter production defaults. The initial test harness
was corrected to recognize the runner's existing configuration-mismatch error
wording; no training-logic defect was found by this suite.

Limits: this validates exercised control/data/checkpoint paths, not convergence,
loss collapse, multi-day robustness, or width-768/context-1024 batch-40 memory
capacity on a target server. The 200,000-update checkpoint is tested by its
save condition and state roundtrip, not by executing 200,000 local updates.
