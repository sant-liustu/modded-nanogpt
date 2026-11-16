# Muon post-update projection matching
Run entry points from the repository root using torchrun with two processes.
Defaults: 124M, FineWeb, 20400 updates, global batch 128, device batch 64,
seed 0. Experiment 1 uses weight decay 0 for the projected reference and
0.1 for the MuonW matching run. Experiment 2 defaults to weight decay 0. Projection captures RMS at update 1000 and rescales
73 matrices after subsequent optimizer updates. Gamma is excluded.

The inherited aggregate metric is base_lr * sum(numel_i/RMS_i)/sum(numel_i).
Embedding LR remains 10 * base_lr. This base-LR proxy differs from averaging
ELRs using actual per-tensor LRs. Hyperball references use raw-update RMS.
The legacy metadata metric tag is retained for compatibility.

Experiment 1: run muon_to_norm_control.py --output results/norm_reference,
then rerun with --reference results/norm_reference/reference.jsonl and a fresh
--output. Experiment 2: norm_control_to_muonh.py --reference followed by the
existing experiments/hyperball_weighted_proxy_match/hyperball_weighted_proxy_reference.jsonl
and a fresh --output. Both accept --config for hyperparameter overrides.

Verification: all three entry paths completed eight steps on one CUDA GPU
with a tiny two-layer model. Projected matching against the existing tiny
Hyperball fixture had maximum relative aggregate error 2.22e-16.
Full 124M training and two-GPU DDP remain unverified.
Locally located old norm-control reference data use AdamW.

## Cloud entry point

Run python experiments/muon_norm_projection_match/run_cloud.py from the repository root.
No experiment config is needed. Requires two GPUs, existing project dependencies, and
FineWeb shards under data/fineweb10B. Runs the reference and both matching experiments
sequentially. Use --dry-run to inspect the fixed commands and configurations.
