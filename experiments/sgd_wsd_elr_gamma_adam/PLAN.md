# Gamma Adam intervention

Keep the existing pure-SGD suite intact. New three-arm suite changes all 1D
RMSNorm gamma parameters to Adam, betas(.9,.95), eps1e-8, WD0. Matrix SGD,
norm projection and sine compensation remain unchanged. Historical gamma
peak LR .0018 is reused as a starting setting, not validated as optimal for B4.
Gamma warms up with the current14791-step warmup then stays constant. This
changes BOTH optimizer and gamma tail schedule relative to the SGD suite.
CLI --gamma-lr supports explicit smaller-rate comparisons; no automatic batch
scaling. All arms use identical gamma settings and a separate output root.

Maintain val256/every500, every-step losses, stable_end and200k checkpoints.
Save and restore Adam step/first/second moments. Verify exact CPU resumed
weights, Adam state and logs, schedule invariants and short CUDA runs. Existing
SGD checkpoints cannot be resumed into this suite. No formal training locally.
