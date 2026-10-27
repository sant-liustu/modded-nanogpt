# Performance change, authorized 2026-10-27

Keep model, data order, SGD math, batching, schedule and logging cadence fixed.
Compare the existing update against a candidate on identical FP32 gradients
and fixed real FineWeb batches. Preserve finite-loss checks before backward
and finite-gradient checks before parameter updates. Test injected NaN/Inf.

Candidate: use the single already-required CPU loss scalar for both the
finite check and train-loss accounting, replacing a separate isfinite kernel
and scalar extraction. Replace square/sum per parameter with foreach L2 norms
and an aggregate L2 norm; diagnostics may differ by rounding, gradients must
not be changed. Retain ordinary SGD step exactly.

Also remove duplicate checkpoint writes at phase endpoints and on reopening
already completed phases. Keep first endpoint save and 200k milestone intact.
Do not modify or restart any remote running process. Benchmark locally;
local GPU timings do not establish a speedup on the user's server.
