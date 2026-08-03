# Plain SignSGD gamma norm-control schedules

## Experiment definition

The four Muon gamma norm-control scripts are the direct comparison baseline.
Each SignSGD sibling preserves the model, tied embedding, learnable RMSNorm
gamma, 20,400 updates, norm schedule, LR/norm matching, monitoring, disabled
spectral-norm estimation, and checkpoint cadence.

Only optimizer 2 changes:

- transformer-block matrices use state-free plain SignSGD;
- the update is `parameter -= learning_rate * sign(gradient)`;
- block learning rate remains `0.00036`, matching the unit-RMS Muon update
  scale;
- momentum, Nesterov, shape scaling, and weight decay are absent;
- tied embedding/head and RMSNorm gamma remain on AdamW.

The optimizer is implemented directly in each self-contained training script.
No shared optimizer module is used.

## Verification

1. Parse and compile all four scripts.
2. Compare schedule functions with their Muon parents.
3. Check exact optimizer coverage and state-free signed updates.
4. Run all four scripts with the local tiny-data smoke configuration.
5. Verify common capture norms and LR/norm ratios after schedule divergence.
