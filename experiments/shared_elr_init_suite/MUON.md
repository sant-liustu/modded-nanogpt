# Fixed-RMS Muon ELR sweep

Run from the repository root, for example:

    torchrun --standalone --nproc_per_node=2 experiments/shared_elr_init_suite/train_fanin_fixed_muon_w768_d12_elr0p03_embed5x.py

Four entries select block ELR peaks 0.03 / 0.1 / 0.3 / 1; tied embedding/head ELR is 5 times block ELR at every step. Model: width768, depth12, 2 GPUs, per-device batch64. Initialization matches the AdamW fanin_fixed baseline. Matrix RMS is projected to its initial value after every update; WD=0. Block matrices use Muon (momentum0.95, Nesterov, 5 Newton-Schulz steps); tied embedding/head and gamma use AdamW. Gamma warms up to absolute LR0.0018 over 1000 steps, then stays constant without norm projection. Matrix ELR warms up for 1000 steps then cosine decays to zero at step20400.

Keep train_muon.py beside the four entries. Gradient monitoring defaults to EVERY update (--gradient-norm-every 1). gradient_norm_history.jsonl contains per-parameter L2, RMS, maximum absolute gradient, and finite checks after DDP/accumulation averaging and before optimizer update. Statistics are packed on GPU and transferred to CPU once per recorded update. optimizer_update_norm_history.jsonl identifies the optimizer type. schedule.json contains base block ELR; tensor_rms_elr_history.jsonl contains actual targets including the 5x embedding multiplier.

DDP uses replicated Muon momentum over already synchronized gradients, preserving complete per-rank checkpoint state. AdamW checkpoints are incompatible. Local validation: synthetic single-CUDA eager 4-update training at all four peaks, fixed-RMS error below 1e-7; packed gradient metrics match previous metrics within floating point tolerance, including NaN/Inf checks. Formal two-GPU/compiled training has not been validated.