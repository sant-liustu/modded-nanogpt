# Untied per-tensor norm heterogeneity A/B

Source: train_gpt2_gamma_adam_hardnorm_pertensor_rmselr_assignmentA_muonhinit_B0128_devB128.py.
Authorized by the user's 2026-10-09 request to copy Assignment A into untied A/B runners.
Status: implemented; awaiting human review. No formal training has been run.

Both new scripts are full copies of Assignment A. They differ only in their
assignment config and identifying header. Existing tied files are unchanged.

- Embedding and output head have separate parameter storage. At initialization,
  embedding copies the head values to preserve the tied baseline's initial
  function, initial RMS, and random-number stream. They train independently.
- Control covers 74 matrices: 72 block matrices, embedding, and output head.
  Each has a separate Adam group with zero weight decay, ELR control, hard norm
  projection, and telemetry.
- Original 73 norm assignments are preserved from A seed 20260903 and B seed
  20260904. The new head uses linear_up in A and linear_down in B.
- Both use the same heterogeneous ELR target. Every original target value is
  preserved; the head receives the embedding target at each of 20400 updates.
- Seed 0, data order, gamma WSD schedule, batch 128, single GPU, no accumulation,
  architecture and remaining settings follow the original.

Run from the repository root, sequentially on one GPU:

```powershell
python experiments/norm_control_schedule_collapse/train_gpt2_gamma_adam_hardnorm_pertensor_rmselr_untied_assignmentA_muonhinit_B0128_devB128.py
python experiments/norm_control_schedule_collapse/train_gpt2_gamma_adam_hardnorm_pertensor_rmselr_untied_assignmentB_muonhinit_B0128_devB128.py
```

Logs and final checkpoints use the inherited logs/<uuid>/ layout.
Compare untied A against untied B using matching steps and the same loss
smoothing as the original experiment. Do not require tied and untied loss
curves to coincide. A tiny smoke test is not evidence of loss collapse.
Untying adds one vocab-by-width matrix and its optimizer/gradient storage;
full-size batch-128 memory and torch.compile are not validated by the smoke.

Validation and target regeneration:

```powershell
python experiments/norm_control_schedule_collapse/build_untied_pertensor_targets.py
python experiments/norm_control_schedule_collapse/smoke_untied_pertensor_hardnorm.py
```

The smoke checks all 20400 formal ELR rows, A/B source equivalence, independent
weight storage with equal initial values, four real CUDA updates per arm on a
tiny model, all norm trajectories and ELR telemetry, and multi-rank rejection.
The tiny model uses eager execution and synthetic tokens.
