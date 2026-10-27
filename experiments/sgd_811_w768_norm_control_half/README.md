# Norm-controlled width-768 SGD 8-1-1 pair

Standalone counterpart of the plain-SGD half-budget experiment. Nothing in the
plain version is modified. Published alongside the plain experiment in experiments/sgd_811_w768_norm_control_half.

## Norm control definition

For each unique trainable 2D tensor W, store r_W = RMS(W_initial). After EACH
optimizer step, replace W by W * r_W/RMS(W). Targets are per tensor, not a common
RMS across layers. Tied embedding/head is processed once. All 1D RMSNorm gamma
vectors remain learnable and are not projected. This is projected SGD, without
tangent-gradient removal or LR compensation. Momentum=0, WD=0, no clipping.

Control starts at initialization, including the shared prefix. Train a NEW
controlled prefix; never use a checkpoint produced by the plain runner. Targets
are stored in every checkpoint, restored and validated, not recomputed from a
resumed checkpoint. A policy marker prevents accidental baseline checkpoint use.
Global parameter_norm need not be constant because gamma remains free.

## Fixed scientific configuration

Same model snapshot as plain version: width 768, depth 12, 6 heads, learned
RMSNorm/QK normalization, RoPE, ReLU squared, tied embeddings; context 1024.
FineWeb GPT-2 prepared uint16 shards, seed 0, FP32 parameters, BF16 autocast.
Uniform initial LR 0.01 and batch 4. No added warmup. Full current batch per
update by default; optional --microbatch enables gradient accumulation.

| Stage | LR arm, batch 4 | Batch arm, LR .01 | Canonical updates |
|---|---|---|---|
| Shared prefix | .01 | 4 | 241386 |
| Middle | .0025 | 16 | 30172 |
| Last | .001 | 40 | 30170 |

Each trajectory consumes 1,235,877,888 tokens. LR arm: 301728 updates; batch arm:
251946 updates. Shared prefix plus both tails: 312288 actual updates. Pairing
of T, data and B/eta is unchanged; projection happens 4/10 times more often in
the LR arm per paired grid in the tail. This may prevent exact loss collapse.

## Run on a single CUDA GPU

```bash
export DATA_DIR=/data/fineweb10B
export OUTPUT_DIR=runs/sgd_811_w768_normcontrol_eta0p01_B4_half
bash run_lr.sh
bash run_batch.sh
```

Or `bash run_pair.sh` runs prefix, LR tail, batch tail sequentially. The batch
entry requires the completed controlled prefix. Resume uses the latest .pt and
direct mmap data offsets; do not change data paths or contract settings.

Need at least 1,235,877,889 training tokens including the last target. No implicit
data cycling or download. Python 3.10+, NumPy and CUDA PyTorch 2.5+ required.

Validation: fixed first 512 contexts, 32 per forward. Default measurement and
latest-checkpoint cadence is 1000 shared macros, plus start/fork/end; independent
prefix_step200000.pt retained. JSONL records the original loss/progress/norm
fields plus norm_control_max_relative_error. Checkpoint norm_targets gives the
target RMS for every controlled tensor. Checkpoints are approximately 494 MB.

## Interpretation and validation

Compare LR versus batch WITHIN the controlled pair. Comparing controlled versus
plain loss is also useful, but does not isolate a scale symmetry: projecting
V/O/MLP/embedding can alter the represented function. Passing local tests is
not evidence of convergence or collapse. No formal training is started here.

Run `python verify_norm_control.py` for the standalone small-model CPU tests.

Local result (2026-10-27): width16/depth1/context16, 100 canonical updates,
synthetic token shards, full batches 4/16/40, validation batch32 with a final
partial batch. Continuous and separately interrupted/resumed runs have bitwise
equal final CPU weights and equal measurement records excluding timing. Maximum
recorded matrix RMS relative error: 2.384185791015625e-7. See verification.json.
The 200k checkpoint trigger and target persistence are tested without performing
200k training updates. Formal width-768 GPU training is not validated by this test.

## Per-update training loss

Each optimizer update is retained in `{arm}_train.jsonl` (prefix, lr, batch),
with train_loss, updates, macro, substep, T, contexts, tokens, lr and batch.
Rows are buffered and flushed at validation boundaries (default 1000 macros),
at milestones, and at arm completion. Validation frequency is unchanged.
For a full trajectory, concatenate prefix_train.jsonl with the desired tail.
On resume, rows beyond the checkpoint update are discarded and regenerated.
A crash can lose the buffered rows since the last save; resuming from the
checkpoint recomputes those updates. Old checkpoints cannot recover losses
from steps completed before this logging feature was added.
