# Plain AdamW without norm control, 2026-10-30

User correction: ordinary AdamW must have free parameter norms. New recommended
entry train_adamw_w768_d12_lr0p0018.py fixes arm=fanin_wd/lr_mode=absolute.
The existing trainer's projection routine returns immediately for this arm;
no change to the trainer, model/init/data, LR groups or Fisher implementation.
The separately named fixed entry remains only as a fixed-RMS ablation.

PASS verify_ordinary_adamw.py (default free-norm arm):
- All 20400 LR values match the historical lr0p0018 baseline.
- Three runtime updates exactly match a separate plain PyTorch AdamW reference,
  with no projection on either side; matrix RMS changes from initialization.

PASS real FineWeb single-CUDA eager w128/d1/vocab50304/B1/T8 four updates,
probe four sequences/micro2/accumulation2/two draws/token-chunk4:
local_checks/fisher_train_20261030_115439/report.json.
- Metadata fixed_norm=false, no norm_control_history.jsonl generated.
- At least one matrix's final RMS differs from its initial RMS.
- Actual per-group LR equals the absolute schedule, independent of RMS.
- Exact Fisher off/on training-state and midpoint-resume equality.
- Fisher/RMS records, model snapshot loading and default final full checkpoint
  all continue to pass.

Entry rejects --arm overrides to avoid accidentally enabling norm projection.
Local tests are small script-correctness checks; no full two-GPU compiled run.
