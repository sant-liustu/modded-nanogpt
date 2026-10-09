# SGD LLM training experiments

Baseline: train_adamw_rmsnorm_gamma_w384_B0064_devB064.py

User-requested copy of the learnable-gamma AdamW baseline in experiments/small_batch_adamw_baseline/generated_train_scripts.

12 layers, width 384, 3 heads; tied embeddings and learnable RMSNorm gamma retained.
Global/device batch 64/64, context 1024, 10200 updates, 668,467,200 tokens (one quarter of the original width-768 source budget; about 16.48 tokens/parameter).
AdamW embedding LR 0.0036; block/gamma LR 0.0018; betas (0.9,0.95); WD 0.
WSD: warmup 500, warmdown 2900 steps. Original precision, compilation and telemetry retained.
Learning rates are inherited, not tuned for width 384. Future SGD scripts go in this directory.

Run from repository root with the existing prepared data and CUDA environment:

    python experiments/sgd_llm_training/train_adamw_rmsnorm_gamma_w384_B0064_devB064.py

No GPU training launched.
## Pure SGD sibling

`train_sgd_rmsnorm_gamma_w384_B0008_devB008.py` replaces both AdamW optimizers with SGD (momentum=0, dampening=0, nesterov=False, weight_decay=0). All trainable parameters, including embeddings and RMSNorm gamma, use SGD. Learning rates and schedules remain inherited pending tuning: embedding 0.0036, block/gamma 0.0018; 81600 updates, warmup 4000, warmdown 23200. No gradient clipping is introduced.

Update telemetry is renamed to `sgd_update_norm_history.jsonl` with `sgd_update_*` fields; it estimates the effective direction from parameter deltas, not Adam states. At zero decay it is `-(parameter_after - parameter_before) / lr`, subject to floating-point rounding.

Run from the repository root:

    python experiments/sgd_llm_training/train_sgd_rmsnorm_gamma_w384_B0008_devB008.py

No full training or convergence claim; hyperparameters have not been tuned for SGD.
SGD global/device batch is now 8/8, without accumulation. Token budget remains 668,467,200. Validation every 4000 steps and norm/update telemetry every 32 steps preserve the baseline token cadence. Learning rates remain unchanged pending tuning.

## User-selected SGD learning-rate sweep

All three scripts use B=8, T=1024, 81600 updates, warmup=4000, warmdown=23200 and seed=0, matching the 668,467,200-token AdamW baseline budget. Only peak learning rates differ. No training has been launched.

| Script suffix | Block/gamma peak LR | Embedding peak LR |
| --- | --- | --- |
| blocklr0p03.py | 0.03 | 0.06 |
| blocklr0p1.py | 0.1 | 0.2 |
| blocklr0p3.py | 0.3 | 0.6 |

Run separately from the repository root:

```sh
python experiments/sgd_llm_training/train_sgd_rmsnorm_gamma_w384_B0008_devB008_blocklr0p03.py
python experiments/sgd_llm_training/train_sgd_rmsnorm_gamma_w384_B0008_devB008_blocklr0p1.py
python experiments/sgd_llm_training/train_sgd_rmsnorm_gamma_w384_B0008_devB008_blocklr0p3.py
```

Each run retains the existing UUID-based log directory. The original untuned SGD template and AdamW baseline are preserved.
## Fixed-learning-rate batch sweep

Batch 2, 4, 8, 16 scripts all use block/gamma LR 0.1 and embedding LR 0.2 (no LR rescaling), seed 0, context 1024 and 668,467,200 tokens. Global batch equals microbatch on one GPU.

| Batch | Steps | Warmup | Warmdown | Validation interval | Norm/update interval |
| --- | --- | --- | --- | --- | --- |
| 2 | 326400 | 16000 | 92800 | 16000 | 128 |
| 4 | 163200 | 8000 | 46400 | 8000 | 64 |
| 8 | 81600 | 4000 | 23200 | 4000 | 32 |
| 16 | 40800 | 2000 | 11600 | 2000 | 16 |

Run all four sequentially, stopping on the first failure:

    python experiments/sgd_llm_training/run_sgd_batch_sweep.py

The runner sets the repository root as working directory. Each training script writes independent UUID logs. This compares batch sizes at fixed peak LR, not independently optimized performance for each batch.