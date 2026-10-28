# Fisher model snapshots, 2026-10-28

Both real FineWeb single-CUDA eager tests PASS, w128/d1/T8, four updates,
four probe sequences split 2x2, two Fisher draws:
- AdamW: local_checks/fisher_train_20261028_203335/report.json
- Muon: local_checks/fisher_train_muon_20261028_203401/report.json

Verified: every probe at test steps 0/2/4 has a model-only file linked in JSONL;
snapshot step/input hash/initial RMS match logs; per-tensor RMS agrees with saved
weights; final weights exactly match complete training checkpoint; no completed
snapshot has a leftover .tmp. Snapshot contains no optimizer/RNG/loader state.
Weights strictly load into a freshly constructed matching GPT; embedding/head
remain tied, and a forward pass is finite.

Off-arm omits --checkpoint-dir/--save-at and verifies the automatic final full
checkpoint under the unique run directory. On-arm retains an explicit midpoint
full checkpoint; resumed final state exactly matches uninterrupted training.
Off/on model/optimizer/scheduler/RNG/loaders/next-batch/norm-target states remain
exactly identical. Initial/current RMS metadata checks continue to pass.

Only small script-correctness tests were run; no new two-GPU, full-length memory,
throughput or statistical-accuracy validation. Snapshot writes execute on rank 0
inside the existing Fisher barriers; default final full checkpoint path is
broadcast to all ranks when needed. Old source/config checkpoint guards remain.
