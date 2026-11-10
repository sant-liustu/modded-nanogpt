# Single-seed Figure 7 cloud package

Python 3.10+, PyTorch 2.x, NumPy. Input: GPT-2-tokenized uint16 train.bin and val.bin.

```bash
pip install torch numpy
# Three GPUs: references parallel, then each matching chain follows its reference.
nohup bash run_all.sh /absolute/path/openwebtext runs 0 1 2 > launcher.log 2>&1 &
# One GPU: all seven runs sequentially.
nohup bash run_all.sh /absolute/path/openwebtext runs 0 > launcher.log 2>&1 &
```

All seven runs use seed 0. Each individual numbered script accepts DATA OUTPUT [extra training options]. Example:

```bash
CUDA_VISIBLE_DEVICES=0 bash 02_ref_wd01.sh /data/openwebtext runs
CUDA_VISIBLE_DEVICES=1 bash 04_match01_fp32.sh /data/openwebtext runs
```

Only start a matching script after its reference is complete. Logs: runs/logs. Each training run records metrics.csv, eval.csv, elr.npy, config.json, latest.pt. Launcher resumes from latest.pt automatically. Resume repeats up to 5000 updates since last checkpoint. A BF16 branch may diverge as in the paper; inspect the corresponding log. Other chains continue. Final nonzero exit indicates at least one failed run.



Paper: https://arxiv.org/abs/2310.04415
Upstream: https://github.com/tml-epfl/why-weight-decay
`model.py` is downloaded unmodified from the upstream main branch on 2026-11-09. Its SHA256 is in provenance.json. Upstream license is included.

Model and optimizer follow upstream defaults: 12 layers, 12 heads, width 768, sequence length 256, vocabulary 50304, no linear/LN biases, learnable LayerNorm gains, tied token embeddings/output head, dropout 0, init scale .02 with upstream residual scaling. AdamW beta=(.9,.95), default epsilon 1e-8, upstream name-based decay groups, fused AdamW when CUDA supports it. Peak LR .0006, 400 warmup updates, cosine toward .00006 over 50000 updates, clipping 1.0, global batch 256, microbatch 256, gradient accumulation 1. No EMA is needed for the Figure 7 raw-model comparison.

Reference runs default to BF16 autocast with FP32 parameter storage, matching upstream. Matching runs set WD=0 and use `lr[t] = reference_elr[t] * current_global_norm[t]`. Global norm uses upstream `model.weight_norm()` (includes LN, shared parameters counted once). LR and norm are sampled before each update. All optimizer groups receive the same LR.

The matching controller and its precise timing are reconstructed from the paper's ELR definition; the public training script does not contain this controller. The paper does not specify reference interpolation or exact norm sampling timing. This implementation saves every step and requires no interpolation. Source code defaults, rather than an unverified historical experimental commit, define architecture details.

Training and evaluation use separate reproducible RNG streams. Runs share initialization and training batch sequence. Evaluation uses a fixed 1000-sequence sample per split every 200 updates. This improves paired measurement but differs from upstream's shared training/evaluation RNG and repeatedly resampled eval sets. `eval.csv` contains pre-update evaluated loss; `metrics.csv` contains every update's minibatch loss, LR, pre-update norm, ELR and clipping norm. Plot these separately. No smoothing is applied. `elr.npy` is the complete controller target trajectory. Failed runs stop on nonfinite values and retain the last checkpoint.

Upstream enables TF32 even with `dtype=float32`; we preserve this default. Use `--strict-fp32` to disable TF32 and record it as an additional variant. Modern CUDA/PyTorch kernels may differ from the original environment.

The full suite includes BF16 reference WD=0/.1/.3 and WD=0 matching runs in FP32/BF16 targeting .1/.3. BF16 matching may fail, as reported by the paper; run individual commands to continue other branches after failure. No claim of cloud training validation is made by the local smoke check.

## Gamma cosine / norm excluding gamma

New pair (seed 0, microbatch=global batch=256):

```bash
bash 08_gamma_ref_wd01.sh /data/openwebtext runs
bash 09_gamma_match01_fp32.sh /data/openwebtext runs
```

Run reference first. Reference uses WD=.1 on matrices, BF16. Match uses WD=0, FP32. Both keep LayerNorm gamma on the original 400-step warmup + cosine schedule (peak .0006, floor .00006), with no WD. Matrix LR in match is target ELR times current global L2 norm excluding all LayerNorm gamma. Reference records the same norm convention. Tied embedding is counted once. Logs add gamma_lr and gamma_norm. Old all-parameter-norm targets are rejected. The original seven launchers retain their behavior. These two new launchers are independent of run_all.sh.

## Two-GPU gamma pair

```bash
CUDA_VISIBLE_DEVICES=0,1 bash 10_gamma_ref_wd01_2gpu.sh /data/openwebtext runs
CUDA_VISIBLE_DEVICES=0,1 bash 11_gamma_match01_fp32_2gpu.sh /data/openwebtext runs
```

Each GPU uses microbatch 128; global batch 256; accumulation 1. Reference and matching remain sequential. Global sampled indices are partitioned across ranks (not duplicated); DDP averages gradients before global gradient clipping. Norm is computed on each full replica, not summed across replicas. Training loss is averaged across ranks. Evaluation uses identical fixed samples on each rank; only rank 0 writes logs and checkpoints. Checkpoints include per-rank random states and require the same world size on resume. New reference is required for the new world-size/microbatch configuration. Single-GPU launchers retain their prior defaults.

Validation of DDP update: Python syntax and single-process CPU tiny run passed; global sampling partition checked. Windows torchrun failed during rendezvous because local PyTorch lacks libuv; alternate Gloo file-store initialization did not complete and was stopped. Two-process training, resume, CUDA/NCCL, compiled training and throughput remain unverified.
