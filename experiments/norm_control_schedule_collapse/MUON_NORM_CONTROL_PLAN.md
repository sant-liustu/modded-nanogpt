# Muon gamma norm-control schedule addendum

## Approval and scope

- Status: approved for implementation.
- Approval source: user instruction on 2026-08-03 to start from the four
  learnable-gamma, monitored, 20,400-step AdamW norm-control scripts and create
  four Muon variants.
- Branch: `experiment/norm-control-muon-schedules`.
- Base: `experiment/norm-control-tied-embedding @ 749a30a`.

## Invariants

The four source scripts remain unchanged.  Each Muon sibling must preserve:

- model architecture, tied embedding, and learnable RMSNorm gamma;
- FineWeb10B data, `B=128`, per-device `B=64`, `T=1024`, and 20,400 updates;
- seed, validation cadence, tensor/update monitoring cadence, spectral-norm
  monitoring, and checkpoint behavior;
- the exact delayed/cosine/linear-down/linear-up norm schedule and its LR
  matching behavior;
- AdamW for the tied embedding/head and RMSNorm gamma.

Only transformer block matrices move from AdamW to Muon.  Muon uses the
corrected block learning rate `0.00036`, momentum `0.95`, Nesterov momentum,
five Newton-Schulz steps, and zero weight decay for every Muon matrix.

## Files

The implementation adds four self-contained sibling scripts named by inserting
`_muon_` after `train_gpt2_gamma_` in the corresponding source filename.
No shared optimizer abstraction is introduced.

## Verification

1. Compile all four scripts.
2. Verify source/variant invariants and exact schedule bodies.
3. Verify complete, non-overlapping optimizer coverage.
4. Verify Muon receives only 2D transformer-block parameters while the tied
   embedding/head and gamma parameters remain on AdamW.
5. Run an extracted lightweight Muon step when the local CUDA runtime permits.
