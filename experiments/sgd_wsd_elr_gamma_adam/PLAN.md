# Gamma Adam intervention

Keep the existing pure-SGD suite intact. New three-arm suite changes all 1D
RMSNorm gamma parameters to Adam, betas(.9,.95), eps1e-8, WD0. Matrix SGD,
norm projection and sine compensation remain unchanged. Historical gamma
peak LR .0018 at B128 motivates sqrt-scaled .0003182 at B4; user selects
.0003 as the new default, not validated as optimal.
Gamma warms up with the current14791-step warmup then stays constant. This
changes BOTH optimizer and gamma tail schedule relative to the SGD suite.
CLI --gamma-lr supports explicit smaller-rate comparisons; no automatic batch
scaling. All arms use identical gamma settings and a separate output root.

Maintain val256/every500, every-step losses, stable_end and200k checkpoints.
Save and restore Adam step/first/second moments. Verify exact CPU resumed
weights, Adam state and logs, schedule invariants and short CUDA runs. Existing
SGD checkpoints cannot be resumed into this suite. No formal training locally.

## Gradient/update monitoring extension
Reuse packed per-tensor gradient stats and pre/post optimizer snapshots from
F:/shared_elr_init_suite_20261014/train_muon.py. Default cadence500, zero disables.
Measure all unique tensors including gamma; add dot products/radial components.
Distinguish optimizer-only displacement (before projection) and total displacement
(after projection). Save raw delta and historical direction norm delta/LR only
for optimizer-only rows. Buffer files and recover to checkpoint like step logs.
Verify on/off exact weights, Adam state, RNG and losses; resumed monitor rows;
SGD direction vs raw gradient, gamma unchanged by matrix projection. Snapshot
cost is one extra FP32 model plus per-tensor scratch at measured updates.

User refinement: retain only L2/RMS plus identifying metadata. Reuse gradient
L2 from overall norm; derive RMS on CPU. Reuse gamma displacement across
projection stages. Remove maxima/dots/radial/relative/delta-over-LR metrics.

User cadence revision: both per-tensor monitors default to every1000 updates
plus final. Validation remains every500 and whole-model step logs every update.
