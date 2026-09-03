# Pure WD=0 rank-1 suffix ablation

This is the direct residual-ablation counterpart to the earlier rank-1
transfer runs. It asks whether replacing the late baseline ELR matrix by its
best rank-one approximation materially changes the baseline loss trajectory,
without introducing a weight-decay intervention.

## Fixed protocol

Reference baseline:

~~~text
rmsnorm_gamma_adamw__B0128__devB064__lr0p0036__blocklr0p0018__wd0
__warmup1000__warmdown5800__seed0__rep01
~~~

| Update range | 72 block matrices | Tied WTE | RMSNorm gamma |
| --- | --- | --- | --- |
| 1-1000 | Baseline WSD LR, WD=0 | Baseline 2x WSD LR, WD=0 | Baseline WSD LR, WD=0 |
| 1001-20400 | Late-baseline rank-1 ELR, WD=0 | Its own rank-1 ELR target, WD=0 | Baseline WSD LR, WD=0 |

The suffix target is rank1_targets_from_wd0_wsd_reference_post1000_dense_preupdate.jsonl.
It is the uncentered best rank-one approximation of the WD=0 baseline's 73 by
19400 raw RMS-ELR matrix over updates 1001-20400. The WTE remains a separate
target row; it is not forced to retain a fixed 2x block-LR ratio after update 1000.

## Run

From the repository root:

~~~powershell
python experiments/rank1_elr_adamw_consistency/train_rank1_elr_adamw_rmsnorm_gamma_B0128_devB064_wd0_prefix_wd0_rank1_suffix_seed00.py
~~~

For cloud execution, upload this trainer and the existing dense post-1000
target JSONL at the same relative path. No environment-variable override is
needed or accepted by the runner for the target/start boundary.

## Required check after the run

rank1_elr_history.jsonl must show experiment_mode=wd0_baseline_prefix_wd0_rank1_suffix,
rank1_elr_start_update=1001, and weight_decay=0.0 at the first active
controller record. The first 1000 optimizer updates use no rank-1 target.
