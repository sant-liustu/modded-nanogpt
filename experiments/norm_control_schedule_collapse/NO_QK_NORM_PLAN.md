# No-QK-Norm AdamW Schedule Ablation

## Question

Does removing Q/K RMSNorm itself change the behavior of the four established
AdamW norm-control schedules, relative to the existing variants where Q/K
RMSNorm is present but has no learnable gamma?

## Experimental variable

- Control: the four existing `gamma_no_qk_gamma` scripts.
- Variant: remove `Attention.q_norm`, `Attention.k_norm`, and their forward
  applications.
- Preserve all attention pre-norm, MLP pre-norm, and final RMSNorm modules and
  their learnable gamma parameters.

## Invariants

- Dataset/model dimensions, seed, 20,400 updates, and B128/devB64 defaults.
- AdamW parameter groups and learning rates.
- Constant, cosine-wave, linear-down, and linear-up schedules.
- Per-tensor norm-control targets and LR/norm matching.
- Monitoring and checkpoint cadence.

## Implementation

Copy each `train_gpt2_gamma_no_qk_gamma_*` entry point to a corresponding
`train_gpt2_gamma_no_qk_norm_*` entry point. Make only the architecture change
above plus remove the now-obsolete Q/K-gamma assertion.

## Verification

- Compile all four entry points.
- Compare normalized ASTs against their parents and confirm changes are limited
  to the header, Q/K norm construction/application, and obsolete assertion.
- Run tiny local smoke configurations and verify finite loss, 25 learnable
  RMSNorm gamma tensors, norm-control initialization, and checkpoint behavior.

## Review focus

Review the `Attention.__init__` and `Attention.forward` changes. All schedule and
optimizer sections should remain unchanged.
