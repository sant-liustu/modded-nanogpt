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

Model and optimizer follow upstream defaults: 12 layers, 12 heads, width 768, sequence length 256, vocabulary 50304, no linear/LN biases, learnable LayerNorm gains, tied token embeddings/output head, dropout 0, init scale .02 with upstream residual scaling. AdamW beta=(.9,.95), default epsilon 1e-8, upstream name-based decay groups, fused AdamW when CUDA supports it. Peak LR .0006, 400 warmup updates, cosine toward .00006 over 50000 updates, clipping 1.0, global batch 256. No EMA is needed for the Figure 7 raw-model comparison.

Reference runs default to BF16 autocast with FP32 parameter storage, matching upstream. Matching runs set WD=0 and use `lr[t] = reference_elr[t] * current_global_norm[t]`. Global norm uses upstream `model.weight_norm()` (includes LN, shared parameters counted once). LR and norm are sampled before each update. All optimizer groups receive the same LR.

The matching controller and its precise timing are reconstructed from the paper's ELR definition; the public training script does not contain this controller. The paper does not specify reference interpolation or exact norm sampling timing. This implementation saves every step and requires no interpolation. Source code defaults, rather than an unverified historical experimental commit, define architecture details.

Training and evaluation use separate reproducible RNG streams. Runs share initialization and training batch sequence. Evaluation uses a fixed 1000-sequence sample per split every 200 updates. This improves paired measurement but differs from upstream's shared training/evaluation RNG and repeatedly resampled eval sets. `eval.csv` contains pre-update evaluated loss; `metrics.csv` contains every update's minibatch loss, LR, pre-update norm, ELR and clipping norm. Plot these separately. No smoothing is applied. `elr.npy` is the complete controller target trajectory. Failed runs stop on nonfinite values and retain the last checkpoint.

Upstream enables TF32 even with `dtype=float32`; we preserve this default. Use `--strict-fp32` to disable TF32 and record it as an additional variant. Modern CUDA/PyTorch kernels may differ from the original environment.

The full suite includes BF16 reference WD=0/.1/.3 and WD=0 matching runs in FP32/BF16 targeting .1/.3. BF16 matching may fail, as reported by the paper; run individual commands to continue other branches after failure. No claim of cloud training validation is made by the local smoke check.
