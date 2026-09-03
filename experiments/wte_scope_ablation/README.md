# WTE-scope ablation: block-only global-F with RMSNorm gamma

## Question

Does the earlier loss gap of the no-gamma block-only global-F experiment come
from excluding tied WTE from the global controller, or from its incomparable
model/optimizer configuration? Both new runners use the same RMSNorm-gamma,
WSD, seed, data order, and WD `0 -> 0.1` switch as the successful full-scope
comparison. They differ only in the treatment of tied WTE after update 1000.

The reference is:

```text
rmsnorm_gamma_adamw__B0128__devB064__lr0p0036__blocklr0p0018__wd0
__warmup1000__warmdown5800__seed0__rep01
```

For its 72 block matrices, let

```text
F_B = sqrt(sum_{i in transformer.h} ||W_i||_F^2).
```

The shared target file contains both the baseline block-only global quantity
`block_LR / F_B` and the baseline WTE ELR `WTE_LR / RMS(WTE)`.

## Runners

1. `...blockonly_globalf_wtebaseline...py`

   After warmup, only the 72 blocks are controlled to the reference
   `block_LR / F_B` and receive WD `0.1`. Tied WTE remains outside `F_B` and
   retains the reference WD=0 and its original 2x WSD LR rule. Its actual ELR
   is logged but not controlled.

2. `...blockonly_globalf_wteelrmatch...py`

   The block controller and block denominator are identical. Tied WTE still
   remains outside `F_B`, with WD=0, but its LR is set each update to

   ```text
   target_WTE_ELR * current_RMS(WTE).
   ```

   Thus the second arm changes only WTE's ELR policy relative to the first.

RMSNorm gamma stays outside the block-only denominator and keeps its ordinary
no-decay optimizer group in both arms.

## Build and run

From the repository root:

```powershell
python experiments/wte_scope_ablation/build_wte_scope_reference.py
torchrun --standalone --nproc_per_node=2 experiments/wte_scope_ablation/train_small_batch_adamw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_addwd_afterwarmup_blockonly_globalf_wtebaseline_seed00.py
torchrun --standalone --nproc_per_node=2 experiments/wte_scope_ablation/train_small_batch_adamw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_addwd_afterwarmup_blockonly_globalf_wteelrmatch_seed00.py
```

The generated target JSONL is required by both cloud runs. The builder is
local-only and reconstructs it from the formal baseline telemetry.

## Interpretation

* If the baseline-policy arm fails but WTE-ELR rescue collapses, WTE's own
  ELR alignment is sufficient; WTE need not enter the block denominator.
* If both fail, the issue is not merely WTE ELR drift: its contribution to the
  global state or a wider cross-block interaction remains relevant.
* If the baseline-policy arm already collapses, WTE scope alone cannot explain
  the old no-gamma block-only result.
