# Plain AdamW gamma weight decay, 2026-10-30

The free-norm train_adamw_w768_d12_lr0p0018.py entry explicitly sets BOTH
--weight-decay 0.1 and --gamma-weight-decay 0.1. All its trainable groups now
use WD0.1, including RMSNorm gamma; gamma_wd metadata records the resolved value.
The underlying trainer defaults gamma_weight_decay=0 to preserve old controls.
LR groups/schedule, free norms, model/init/data and Fisher/snapshots are retained.

PASS verify_ordinary_adamw.py: actual optimizer groups all assert WD0.1; three
updates exactly match separate plain PyTorch AdamW with WD0.1 for every tensor.
All legacy LR values still match, and matrix RMS evolves freely.

PASS real FineWeb single-CUDA eager w128/d1/T8 four-update correctness test:
local_checks/fisher_train_20261030_120135/report.json. Actual checkpoint groups
all assert WD0.1; metadata gamma_wd=.1, fixed_norm=false, no projection history,
changed matrix RMS, exact Fisher off/on and resumed state equality, and snapshots
all checked. This is small correctness evidence, not full cloud training evidence.
