# Two-draw correctness validation, 2026-10-28

Production defaults in both runners: 1024 sequences per independent draw,
micro-batch64, 16 accumulations, two draws with seed and seed+1. Existing
measurement times and initial/current RMS metadata are preserved.

verify_fisher_accumulation.py PASS: two-draw values exactly match independent
single-draw calls with the recorded seeds; each draw resets its gradient buffer.
Mean and sample standard error match (S1+S2)/2 and abs(S1-S2)/2. Group values
are aggregated within each draw before computing uncertainty. Existing masked
accumulation versus full-objective, state-isolation and schedule checks pass.

Real FineWeb single-CUDA eager correctness tests, w128/d1/vocab50304, four
training updates, T8, four diagnostic sequences split 2x2, two draws:
- AdamW: local_checks/fisher_train_20261028_192305/report.json
- Muon: local_checks/fisher_train_muon_20261028_192331/report.json

Both PASS exact off/on model, optimizer, scheduler, RNG, loader, next batch and
norm-target equality, initial/current RMS checks and exact midpoint resume.
Tests assert two seeds, two values and correct mean/SE in every parameter/group.
Source directory has been synchronized with the tested review implementation.

Only small correctness tests requested; no long-run, full-length memory,
distributed/compiled or statistical-precision validation. Two-draw SE describes
pseudo-label noise conditional on fixed inputs and is itself noisy.

Reproduce: python verify_fisher_accumulation.py, then use FISHER_TEST_RUNNER
train.py or train_muon.py and FISHER_TEST_LENGTH=8 with python verify_fisher.py.
