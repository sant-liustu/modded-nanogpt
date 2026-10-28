# AdamW / Muon: two independent accumulated raw Fisher estimates

Both train.py (AdamW) and train_muon.py (Muon), including their existing ELR
launchers, now enable the requested protocol by default:

- 1024 GLOBAL sequences per draw, micro-batch 64 per rank, 16 GLOBAL micro-batches, sequence length 1024.
- With two ranks: each processes 512 sequences / 8 micro-batches, then SUM-reduces gradients BEFORE squaring.
- Two independently seeded estimates per measurement (fisher_draws=2 by default).
- Each estimate independently accumulates 16 micro-batches, then squares once.
- Same fixed inputs in both estimates; only pseudo-label draws differ.
- Steps 0, 100, 1000, 2000, then every 1000; also final and resume points.
- Raw per-token Fisher, with initial/current per-parameter RMS preserved.

Explicit flags (same for either launcher's original training command):

```text
--fisher-every 1000 --fisher-draws 2 --fisher-batch-size 64 --fisher-accumulation-steps 16 --fisher-sequence-length 1024
```

Use --fisher-every 0 to disable. The GLOBAL sequence count is micro-batch size times
accumulation steps; accumulation steps must be divisible by world size. These flags configure diagnostics, not the training batch.
Other optimizer, ELR and norm-control settings remain unchanged.

## Exact calculation

For each draw, freeze weights. A dedicated loader supplies 16 different consecutive batches,
kept on CPU and reused at every measurement. For each micro-batch, move only
that batch to GPU, forward, sample independent categorical pseudo-labels at
valid positions, and differentiate sum(CE) / N, where N is the TOTAL valid-token
count across all 16 micro-batches. Accumulate gradients in private FP32 buffers.
Release each graph immediately. After all 16 backward passes, square the sum
ONCE: trace_j = N * sum(g_j**2); mean_fisher_j = trace_j / numel_j.
Squaring/reducing uses FP64. This is NOT an average of 16 squared gradients.
With length 1024 and no mask, N=1,048,576, not 1024.

Each draw costs 16 GLOBAL micro-batch forwards/backwards. On two ranks each
rank performs 8, followed by SUM gradient communication; two draws perform 16
per rank in total. The diagnostic uses FP32 eager on all training ranks. Training
may use mixed precision; speedup depends on compute versus communication. A 64x1024
FP32 diagnostic can require substantial GPU memory. Local shortened-sequence
tests do not validate the memory footprint of the full-length configuration.

## Output and interpretation

fisher_history.jsonl stores step, input hash (inputs plus masks/targets), sequence
count, micro-batch size, accumulation steps, valid tokens, precision and time.
Each unique trainable tensor has mean_fisher, trace, numel, shape, rms and
initial_rms. Groups Emb/QK/VO/FFN/Norm use parameter-count-weighted averages.
Tied embedding/head is counted once and names retained in aliases.

Each tensor and group records mean_fisher_draws=[S1,S2], mean_fisher=(S1+S2)/2,
mc_standard_error=abs(S1-S2)/2, and trace=numel*mean_fisher. Group values are
aggregated per draw before computing SE, preserving parameter covariances.
For world size R, draw k on rank r uses seed=fisher_seed+k*R+r. Seeds are
recorded in draw_seeds_by_rank[draw][rank]; draw_seeds records the logging rank.
Each draw starts with zero private gradient buffers. They reuse fixed inputs and model
weights. The SE measures only pseudo-label randomness conditional on inputs;
with two draws it is noisy and does not quantify input/population uncertainty.
An explicit --fisher-draws 1 still works and reports SE=null. Labels are
model-sampled, not true labels. Increasing sequence count alone is not a
demonstrated precision guarantee; no confidence interval is claimed.
No EMA, optimizer preconditioning, tangent projection or gain relaxation.

fisher_initial_rms.json and checkpoints retain actual step-zero RMS; resume
restores it. RMS=sqrt(sum(W**2)/numel), not Frobenius norm. Offline, multiply
raw per-tensor mean_fisher by current rms**2 for that RMS-scaled coordinate,
or initial_rms**2 for the initial fixed coordinate. Aggregate only AFTER
per-tensor scaling. Norm vector logging does not imply its LR is RMS-scaled.

## Isolation and compatibility

Uses autograd.grad and private accumulation buffers, never parameter .grad.
Restores model modes and torch RNG; dedicated inputs do not advance training
or validation loaders. All ranks construct the same fixed global probe inputs
on CPU and take disjoint contiguous sequence partitions; the input hash refers
to the full global inputs. Private local gradients use the GLOBAL valid-token
denominator, SUM all-reduce, and only then square. Do not average local Fisher
values or divide the global gradient by world size again. Rank 0 writes logs;
seconds records the slowest rank and peak memory fields refer to rank 0. Measurement resets the CUDA peak-memory counter and
is included in wall-clock training time. Old-version checkpoints remain subject
to strict source/config compatibility checks; do not bypass them.

## Validation

python verify_fisher_accumulation.py checks accumulated gradients against a
single combined loss, including unequal masks; distinguishes square-of-sum
from sum-of-squares, checks state isolation and the default measurement schedule.

PowerShell runner checks (real FineWeb, CUDA, on/off and resume comparisons):

```powershell
$env:FISHER_TEST_WIDTH='128'
$env:FISHER_TEST_DEPTH='1'
$env:FISHER_TEST_LENGTH='8'
$env:FISHER_TEST_RUNNER='train.py'
python verify_fisher.py
$env:FISHER_TEST_RUNNER='train_muon.py'
python verify_fisher.py
```

Current script-correctness tests default to 4 sequences / micro-batch 2 / 2
accumulations, two draws, and shorter sequences. Production defaults remain
1024 / 64 / 16. They do not validate two-GPU compiled training,
full-length memory feasibility, or population Fisher accuracy.

## Paper context

https://arxiv.org/html/2502.19002v2#A1.SS1 specifies B=1024 for Eq.3, but does not
unambiguously describe the sequence/token unit or repeated-estimation count.
Our explicit sequence protocol is the user's requested setting, not a claim of
exact reproduction of undisclosed implementation details. Figure 3 shows
0,100,1000,10000,20000 from 50,000-step GPT-2-small / LLaMA-0.25B training.
We retain raw arithmetic trace means (Eq.4), not mean(log(diagonal Fisher)).

## Distributed verification and two-GPU launch

python verify_fisher_distributed.py uses two CPU/Gloo processes and compares
both draws against a serial combined objective with identical per-rank labels,
including uneven valid-token masks. This verifies communication algebra and
state isolation, not two-GPU NCCL or compiled-training performance.

From this directory, run either job separately:

```bash
torchrun --standalone --nproc_per_node=2 train_fanin_fixed_w768_d12_elr0p06.py
torchrun --standalone --nproc_per_node=2 train_fanin_fixed_muon_w768_d12_elr0p1_embed5x.py
```

Both training and Fisher run on two ranks. Existing data paths may need local
CLI overrides. Old-version checkpoint source/config checks remain enforced.
