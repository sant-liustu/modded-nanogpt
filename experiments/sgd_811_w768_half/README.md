# Width-768 SGD: paired 8-1-1 schedules

User configuration: width 768 / depth 12 / six heads / context 1024,
existing RMSNorm-QK/RoPE/ReLU-squared model, seed 0. Uniform LR 0.01,
initial global batch 4. Plain SGD; no momentum, weight decay, clipping or
weight-norm control. BF16 autocast, FP32 parameters. No added warmup.

Budget is half of `scaling_collapse_model_only_20261022/train_fixednorm_w768.py`:
9,429 * 128 * 1024 = **1,235,877,888 tokens per trajectory**.
At batch 4 this is **301,728 canonical updates**. 80/10/10 stages are integer
rounded. See PLAN.md for the exact pairing and integerization specification.
The executed multipliers are exactly **1, 4, 10**, with no alternating values.
LR-arm levels are **0.01, 0.0025, 0.001**, at fixed batch 4; batch-arm
levels are **4, 16, 40**, at fixed LR 0.01. Both arms match exactly at macro
boundaries. This is an integer-multiplier 8-1-1 variant, not the paper's
middle level 0.01/sqrt(10). Round each 10% tail to the nearest complete
group of 4 or 10 reference updates and assign the remainder to the prefix.
For the formal budget the phase lengths are 241,386 / 30,172 / 30,170
canonical updates. LR arm: 301,728 updates; batch arm: 251,946 updates.
Shared prefix plus both tails: 312,288 actual optimizer updates.
Old balanced-schedule checkpoints have a different schedule hash and cannot
be resumed as this experiment. Use a fresh output directory.

## Server execution (single CUDA GPU)

Two entry points, run in this order (same DATA_DIR and OUTPUT_DIR):

```bash
cd experiments/sgd_811_w768_half
export DATA_DIR=/data/fineweb10B
export OUTPUT_DIR=runs/sgd_811_w768_eta0p01_B4_half
bash run_lr.sh
bash run_batch.sh
```

`run_lr.sh` trains/resumes the shared prefix and then the LR tail.
`run_batch.sh` loads that completed shared prefix and trains the batch tail.
Do not launch the batch entry until the prefix is complete. Alternatively,
`bash run_pair.sh` runs all stages sequentially. None of these commands download
data. Both entry points retain the 200,000-update prefix checkpoint.

Requirements: Python 3.10+, NumPy, PyTorch 2.5+ with CUDA and BF16 support.
Use the same prepared GPT-2 FineWeb uint16 bin format as the existing suite.
At least 1,235,877,889 training tokens are needed (the extra token is for the
final next-token target). No data download or implicit repetition is performed.

```bash
python train.py --updates 301728 --output runs/sgd_811_w768_eta0p01_B4_half \
  --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --preflight
python train.py --updates 301728 --output runs/sgd_811_w768_eta0p01_B4_half \
  --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin'
```

Default `--arm all` runs one common prefix, then the LR tail, then reloads
the same prefix and runs the batch tail. This saves duplicating prefix work.
You can instead use `--arm prefix`, then separately `--arm lr` and
`--arm batch`. Each arm automatically resumes its latest saved checkpoint.
Do not run two processes targeting the same arm/output directory.
Keep data paths, file metadata, microbatch and evaluation settings unchanged
when resuming; the checkpoint validates the experiment contract.

Checkpoint policy: `prefix.pt`, `lr.pt`, `batch.pt` each retain the latest
state, saved at evaluation ticks (default every 1000 macros) and arm endpoints.
Additionally `prefix_step200000.pt` permanently retains the common-prefix
state immediately after optimizer update 200,000 (T=2000; 819,200,000 tokens).
It is independent of evaluation cadence and is never overwritten. This step
precedes the fork at update 241,386, so one milestone serves both trajectories.
Each FP32 model checkpoint is about 494 MB; momentum-free SGD has no moment
buffers. Saved state includes model weights, update/macro counters, data cursor,
random states and experiment contract. Resume seeks the mmap tape directly,
without replaying consumed batches; LR is reconstructed from the saved macro.
To explicitly resume the milestone, copy it as `prefix.pt` into a NEW output
directory and use that output with the otherwise identical command. Keep the
same data paths/metadata. This preserves the original experiment and logs.

By default, omit `--microbatch`: each update processes its full global batch
in one forward/backward (4, then 16/40 in the batch-schedule arm). There is no
gradient accumulation in this mode. Optionally set `--microbatch 4` to cap each
forward/backward and accumulate a correctly averaged global-batch gradient.
No DDP is implemented. Existing checkpoints retain their microbatch contract;
when resuming an older microbatch-4 run, explicitly pass `--microbatch 4`.

Validation: same first 512 contexts (524,288 predicted tokens) of the validation tape, default every
1000 common macros and each arm endpoint. Each validation forward uses
`--validation-batch-size` contexts (default 32), independently of training
microbatch, so default evaluation takes 16 forwards.
JSONL reports validation CE,
last-update train CE, gradient and parameter norms, LR, batch, tokens and T.
The first record of each tail is the identical shared-prefix evaluation.
Use the same T grid to compare `lr.jsonl` and `batch.jsonl`; don't compare
their optimizer-step counters as if they were equal.

## Local verification only

`python verify.py` checks the model snapshot against the local source, schedule
invariants, shard-boundary token continuity and microbatch accumulation. On a
server without the original source file, that snapshot comparison is skipped.
`python train.py --updates 20 --output smoke_gpu --eval-every 8 --validation-contexts 2 --microbatch 1 --validation-batch-size 1`
was run on the predecessor implementation locally with the full 768 model. The half-budget package has invariant and checkpoint tests; no new GPU training was started. This is a functional
smoke test, not a scientific 8-1-1/collapse or convergence result: its tiny
budget compresses the schedule tail into one macro.

The local machine only has one 100M-token training shard, insufficient for the
formal budget. No formal training is running. Checkpoints/smoke outputs are
excluded from the portable source archive and Git.
