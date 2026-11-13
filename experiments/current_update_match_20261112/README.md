# Current-step aggregate relative-update matching

This supplements the old wd 0.1 -> 0 intervention. The reference keeps wd=0.1. The new run sets wd=0 from update 1001 and matches eta_t * ||u_t|| / ||w_(t-1)||. Initialization seed 0, global batch 128, device batch 64, 20,400 steps, warmup 1,000 and warmdown 5,800 are inherited from the original script.

The scope is the original aggregate of 73 non-gamma block and tied embedding tensors, with embedding LR twice block LR. It is not per-tensor matching. Adam's adaptive update excludes decoupled weight decay. The controller predicts the current bias-corrected Adam update from current gradients and previous moments without modifying optimizer state, then sets LR before the optimizer step.

The bundled reference contains baseline norms recorded every FOUR steps and linearly interpolated. This is an interpolated-reference experiment, not exact matching to a measured every-step reference. For exact reference matching, regenerate the reference with every-step parameter and update norm logging. Each entry must use parameter norms before update t and Adam update norms from update t.

From the original training repository root, with the same FineWeb shards and cloud CUDA environment, run:

```bash
torchrun --standalone --nproc_per_node=2 /path/to/current_update_match_20261112/train_current_update_match.py
```

Upload the script and bundled gzip-compressed reference JSONL together. Keep the original data directory relative to the repository root. The entrypoint inherits the original validation, optimizer, model and compile settings.

`current_update_match_history.jsonl` records target, predicted and realized current-step relative updates and relative matching error. Realized updates are inferred from actual parameter displacement, excluding weight decay, so finite-precision subtraction adds measurement error. Inspect this error before interpreting loss discrepancy. Compare loss against the original wd=0.1 reference and existing aggregate ELR matching run over steps 1001-20400.

Local verification covers syntax and a synthetic CUDA fused AdamW prospective-update calculation. Full model training and cloud DDP have not been run.

The reference is gzip compressed losslessly. The training script and launcher read it directly.
