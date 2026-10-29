# lr0p0018 AdamW + fixed RMS validation, 2026-10-30

Legacy LR source verified directly:
experiments/eta_lambda_invariance/train_gpt2_mupp_w768_lr0p0018_cosine_wd0p1.py.
Embedding peak .0018, block AND gamma .0009; no RMS multiplication; 1000 warmup,
20400 total, cosine floor .1, preserving zero-based optimizer iteration indexing.
WD .1 for matrices, WD0 for gamma, no clipping, AdamW betas .9/.95 and eps1e-8.

PASS verify_ordinary_adamw.py:
- Every one of 20400 actual-LR schedule values matches the old formula.
- Three actual optimizer-construction/application/projection updates exactly
  match a separate plain PyTorch AdamW reference on a tiny tied-head GPT.
- Different tensor RMS values do not change the assigned actual LR.

PASS small real FineWeb CUDA eager w128/d1/vocab50304/B1/T8, four updates,
Fisher four sequences / micro2 / accumulation2 / two draws / token-chunk4:
- absolute mode: local_checks/fisher_train_20261030_020132/report.json
- old rms_elr mode: local_checks/fisher_train_20261030_020158/report.json

Both retain exact Fisher off/on model/optimizer/scheduler/RNG/loader/next-batch/
norm-target equality; snapshots load, RMS agrees, midpoint resume exactly matches.
Absolute-mode logs verify per-group actual LR against the schedule, AdamW
betas/eps/decay, and matrix projection relative errors below 1e-6.
Named old ELR launchers reject --lr-mode overrides; new entry rejects ELR flags.

Only small local correctness checks; no full cloud two-GPU/NCCL/compiled run or
convergence claim. Current model/init/data are preserved, so this is NOT an exact
reproduction of the historical baseline or the sharpness paper.
