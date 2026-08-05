# Norm-control schedules without learnable Q/K gamma

## Experiment question

Test whether the schedule-collapse behavior attributed to learnable RMSNorm
gamma persists when Q/K RMSNorm remains functionally present but its affine
gamma is fixed to one.

## Controlled comparison

The four existing AdamW gamma scripts are copied directly. The new siblings
preserve the model width/depth, tied embedding, 20,400 updates, seed, data,
optimizer settings, per-tensor norm-control targets, four norm schedules,
LR/norm matching, monitoring, and checkpoint behavior.

The sole architecture change is the RMSNorm gamma split:

- Q/K RMSNorm uses `weight=None` and has no trainable parameter;
- attention pre-norm, MLP pre-norm, and final RMSNorm retain learnable gamma;
- expected learnable gamma count is 25 rather than 49 for the 12-layer model.

## Verification

1. Compare every schedule function with its gamma parent.
2. Verify no `q_norm.weight` or `k_norm.weight` occurs in named parameters or
   optimizer state while all 25 other gamma tensors remain.
3. Run all four scripts with the local tiny-data CUDA configuration.
4. Verify common capture norms, checkpoint output, and LR/norm ratios.
