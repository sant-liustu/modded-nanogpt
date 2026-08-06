# AdamW Seed Variance Runs

Four matched 20,400-step AdamW runs for estimating seed variation.

Shared configuration: batch size 128, device batch size 64, sequence length
1024, embedding LR 0.0036, block LR 0.0018, block weight decay 0, warmup 1000,
warmdown 5800, and the same data path and monitor cadence as the B128 baseline.

| Seed | Script |
| ---: | --- |
| 17 | `generated_train_scripts/train_small_batch_adamw_B0128_devB064_lr0p0036_wd0_seed000017.py` |
| 271828 | `generated_train_scripts/train_small_batch_adamw_B0128_devB064_lr0p0036_wd0_seed271828.py` |
| 314159 | `generated_train_scripts/train_small_batch_adamw_B0128_devB064_lr0p0036_wd0_seed314159.py` |
| 987654321 | `generated_train_scripts/train_small_batch_adamw_B0128_devB064_lr0p0036_wd0_seed987654321.py` |

Each script seeds Python, NumPy, PyTorch, and CUDA before model and loader
construction. Analyze final validation loss and EMA training loss as mean,
sample standard deviation, and a confidence interval across these four seeds.
