# Old lr0p0018 AdamW baseline + current fixed-initial-RMS model

Run from repository root on two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 experiments/shared_elr_init_suite/train_adamw_fixed_w768_d12_lr0p0018.py
```

This preserves the CURRENT w768/d12 model (QK RMSNorm, RoPE, ReLU squared,
tied head), fan-in initialization, FineWeb data, B128/T1024, per-device B64,
seed0, 20400 updates, and current distributed chunked Fisher/snapshots.
It uses ordinary torch.optim.AdamW with actual per-group learning rates that
do NOT depend on RMS. Matrix RMS is projected to its own step-zero RMS after
each update; RMSNorm gamma is trainable and not projected.

The user's old baseline was located and verified in:
experiments/eta_lambda_invariance/train_gpt2_mupp_w768_lr0p0018_cosine_wd0p1.py.
The lr0p0018 name denotes EMBEDDING LR, not a common LR for all parameters:

| group | peak actual LR | weight decay |
|---|---:|---:|
| tied embedding/head | 0.0018 | 0.1 |
| Transformer block matrices | 0.0009 | 0.1 |
| RMSNorm gamma (including final norm) | 0.0009 | 0 |

AdamW betas=(.9,.95), eps=1e-8, fused=True; no gradient clipping, matching that
old baseline. Warmup1000, then cosine to 10% peak. The old zero-based optimizer
iteration indexing is preserved exactly (the last used LR is just above the
floor; the scheduler reaches the floor after that update). All 20400 scheduled
steps were audited against the legacy formula. Gamma follows this SAME cosine
curve; it does not use the old ELR runner's separate LR.0018 constant plateau.

CLI --learning-rate controls the BLOCK/GAMMA peak (default .0009), and
--embedding-lr-multiplier defaults2. These are absolute LRs. --min-lr-ratio
defaults.1 and --grad-clip defaults0. ELR JSON and ELR-only flags are rejected
by this entry. --schedule-json is not accepted in absolute mode.
Original train_fanin_fixed_*_elr*.py entries retain rms_elr mode and policies.

Schedule.json has kind=absolute_lr and explicit block, embedding, gamma LRs.
For compatibility, tensor_rms_elr_history.jsonl keeps its filename but logs
lr_policy=baseline_absolute_lr and target_rms_elr=null; actual_rms_elr is a
DIAGNOSTIC ratio only, not the LR-setting rule. Schedule metadata describes
the selected policy. Checkpoint config includes lr_mode and LR hyperparameters;
the source/config guards reject incompatible old checkpoints.

This is the old actual-LR baseline with CURRENT-model norm control added.
It is not a full paper reproduction (data/model/init/batch remain current),
nor a free-norm baseline. Decay is applied during AdamW before matrix projection.

Verify the golden update and the old LR schedule:

```bash
python experiments/shared_elr_init_suite/verify_ordinary_adamw.py
```

Small real FineWeb runner tests (PowerShell):

```powershell
$env:FISHER_TEST_RUNNER='train.py'
$env:FISHER_TEST_LENGTH='8'
$env:FISHER_TEST_LR_MODE='absolute'
python verify_fisher.py
$env:FISHER_TEST_LR_MODE='rms_elr'
python verify_fisher.py
```

Tests verify group actual LR independence from matrix RMS, norm projection,
Fisher off/on isolation, snapshots, and exact resumed state. Only small local
CUDA eager checks are claimed; full cloud two-GPU compiled performance remains
unvalidated for this new entry.
