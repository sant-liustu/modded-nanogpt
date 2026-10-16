# Fixed-norm width ladder at matched TPP

Five standalone trainers derived from the shared ELR initialization suite. Width ladder:

| Script | Width | Unique parameters | Total updates | Consumed tokens | Actual TPP | GPUs / device batch |
|---|---:|---:|---:|---:|---:|---|
| train_fixednorm_w384.py | 384 | 40,563,072 | 6,189 | 811,204,608 | 19.998599 | 1 / 128 |
| train_fixednorm_w768.py | 768 | 123,590,400 | 18,858 | 2,471,755,776 | 19.999577 | 2 / 64 |
| train_fixednorm_w1536.py | 1536 | 417,047,040 | 63,636 | 8,340,897,792 | 19.999897 | 4 / 32 |

The common target is TPP=20.0, anchored to w768 with 123590400 unique trainable parameters (2471808000 target tokens). Use the SAME `--tpp` argument on all entries to change it. Total updates round to the nearest full global batch (TPP differences above are rounding only). Parameter counts include all trainable gamma and count tied embedding/head once.

Global B128 and context1024 are fixed. Depth12, head dimension128 and vocab50304 are fixed. Width doubling increases parameters by about 3.05x then 3.37x, so token budgets do NOT simply double or quadruple.

## Preserved behavior

Only `fanin_fixed` is allowed. All block matrices use std=1/sqrt(fan_in), with no extra residual depth initialization factor. Tied embedding/head std0.02. Input multiplier1; logits/(width/768); RoPE, squared ReLU, learnable RMSNorm including QK norms; attention scale1/sqrt(head_dim). All 73 matrices project to their own initial RMS after each update. Matrix WD0; gamma WD0 and no gamma projection.

Matrix actual LR = shared ELR(update) * current matrix RMS; embedding participates in the same ELR, without an extra embedding LR factor. Default peak ELR follows e_i=0.06*sqrt(18858/T_i), anchored at width768: width384=0.10473427355087156, width768=0.06, width1536=0.03266236513438636 (use describe-only for full-precision generated values). Thus e_i^2*T_i is constant. Gamma peak absolute LR stays0.0018 across widths, warms up over round(T*1000/20400) then stays constant. This inverse-square-root scaling is an experimental hypothesis, not an established equivalence to AdamW timescale. Defaults are generated for TPP20; explicit --peak-elr remains available.

Matrix ELR uses linear warmup over round(10% total updates), immediately followed by linear decay to zero over the remaining updates; no constant plateau. Width384:619+5570; width768:1886+16972; width1536:6364+57272. Only --schedule linear is accepted, and external schedule JSON is rejected. Gamma retains its previous separate warmup (303/924/3119 updates) then constant absolute LR; only matrix ELR shape changed. Forward multipliers remain unchanged; this is an ELR/fixed-norm experiment, not an exact reproduction of the papers' parameterization.

## Data budget vs corpus size

`--input-bin` supplies the corpus. These scripts increase CONSUMED training tokens by increasing updates; they do not download or select more unique shards. The inherited loader cycles the supplied shards and may skip a short tail at shard boundaries. For single-pass training supply enough distinct tokens for the largest run plus boundary overhead; use a consistent shuffled corpus/prefix policy across widths. The startup DATA_BUDGET record reports available tokens and nominal corpus passes. These are not an audit of consumed sample order.

## Inspect without launching training

```powershell
python train_fixednorm_w384.py --describe-only
python train_fixednorm_w768.py --describe-only
python train_fixednorm_w1536.py --describe-only
python train_fixednorm_w384.py --describe-only --tpp 20
```

## Training (Linux/CUDA host, run from this directory)

```bash
torchrun --standalone --nproc_per_node=1 train_fixednorm_w384.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w384 --run-name w384
torchrun --standalone --nproc_per_node=2 train_fixednorm_w768.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w768 --run-name w768
torchrun --standalone --nproc_per_node=4 train_fixednorm_w1536.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w1536 --run-name w1536
```

Choose fresh checkpoint directories. Inherited complete-state checkpoint/resume behavior remains; the new per-width source hashes/config must match. Do not resume old source checkpoints. Periodic validation remains disabled by default (`--val-loss-every 0`); a validation glob is still needed by the inherited loader.

## Verification

Local pre-publication verification checked actual meta-device model parameter counts at all full widths; source AST equality for model, initialization and fixed-RMS functions; all-step two-segment ELR values and separate gamma warmup policy; tied weights; CPU optimizer/projection execution on a tiny model; full-budget arithmetic and schedule endpoints. All five trainer copies were byte-compared against locally verified versions before publication. This is not full GPU/DDP/Inductor/data/checkpoint validation. No formal training is launched by this task.

## Depth variants (width768)

Two standalone entries preserve the same fan-in fixed-RMS model construction as yesterday. Only depth and the resulting parameter budget/ELR/schedule durations change; no 3/depth residual multiplier or depth-dependent initialization is added. Depth6 defaults to 1 rank/device batch128; depth24 defaults to 4 ranks/device batch32; global batch128. Use gradient accumulation with a smaller --device-batch-size if memory requires it.

| Script | Depth | Parameters | Updates | Tokens | Actual TPP | Peak ELR | Warmup / decay |
|---|---:|---:|---:|---:|---:|---:|---|
| train_fixednorm_w768_d6.py | 6 | 81112320 | 12377 | 1622278144 | 20.000391 | 0.07406130676167542 | 1238 / 11139 |
| train_fixednorm_w768_d24.py | 24 | 208546560 | 31822 | 4170973184 | 20.000201 | 0.046188656539899973 | 3182 / 28640 |

Both use peak ELR squared times updates=67.8888, the same width768/depth12 anchor. Gamma stays at absolute peak LR0.0018, warmup607/1560 updates respectively, then constant. All matrices including tied embedding are fixed to their own initial RMS; gamma is outside projection. Controlled matrix count changes with depth:37/145 respectively.

```bash
python train_fixednorm_w768_d6.py --describe-only
python train_fixednorm_w768_d24.py --describe-only
torchrun --standalone --nproc_per_node=1 train_fixednorm_w768_d6.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w768_d6
torchrun --standalone --nproc_per_node=4 train_fixednorm_w768_d24.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w768_d24
```

Verification covers all five entries; formal multi-GPU runs have not been launched.

## Eight-GPU execution plan

Two independent lanes run concurrently:

- large: GPUs0-3 run w1536, 4 ranks/device B32.
- remaining stage1: GPUs4-7 run w768/depth24, 4 ranks/device B32.
- remaining stage2 (only after stage1 succeeds): GPUs4-5 w768/depth12 (2 ranks/device B64), GPU6 w768/depth6 (1 rank/device B128), GPU7 w384/depth12 (1 rank/device B128), concurrently.

One command starts both lanes:

```bash
python launch.py --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/formal01 --dry-run
python launch.py --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/formal01
```

Or two terminals, with DIFFERENT fresh output roots:

```bash
bash run_large.sh --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/large01
bash run_remaining.sh --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/remaining01
```

Do not combine the all-lanes command with separate lane commands on the same GPUs. --gpu-ids accepts eight unique indices into the outer CUDA-visible inventory; default0..7. Each torchrun job has its own standalone rendezvous. Console logs, allocation manifest, checkpoints and result summary are under the fresh output root. A failed stage blocks its next stage and stops peers within that stage; the other lane remains independent. Formal GPU training has not been launched. `python verify_launcher.py` tests CPU subprocess sequencing, failure barriers, remapping and allocations.


## Optimized training scripts (2026-10-16)

All five trainers now sample diagnostics every 1000 updates by default, including
parameter norms, inferred Adam updates, LR/RMS records, and projection records.
Non-sampled updates skip diagnostic parameter snapshots and file writes. Loss,
validation, and checkpoint policies are unchanged. Existing tensor/Adam diagnostic
interval flags remain configurable; LR/projection diagnostics use interval 1000.

Matrix LR uses scheduled ELR multiplied by the saved fixed target RMS. Targets are
cached on CPU after checkpoint restoration; ordinary updates do not rescan weights
for LR. Gamma retains its absolute LR schedule. Projection still runs every update,
compiled with fullgraph=True when compile_model=1; compile_model=0 uses eager mode.
Compilation happens on first use. One aggregate validity check remains per update.
Sampled diagnostics check actual RMS. Diagnostic writes use buffered files.

The mathematical fixed-norm policy is preserved, but replacing measured RMS with
target RMS and compiled reductions can change floating-point training trajectories.
Local 1001-step synthetic CPU/CUDA checks passed at RMS about 0.02; maximum CUDA
parameter difference was 5.29e-5 and RMS relative error 2.38e-7. A stress test at RMS
about 1 exceeded the 1e-4 parameter comparison threshold. This does not establish
full-run equivalence. Full-size cloud DDP and throughput remain unverified.

A 1.31M-parameter local laptop-GPU benchmark measured 83.2 to 36.5 ms per ordinary
step, with eager model forward/backward; this is not a cloud speedup prediction.

The five scripts match the source hashes in publication_manifest.json. GPU lane
allocation, global batch, TPP budgets, and schedules remain unchanged. Source-hash
checkpoint validation is still strict: old-source checkpoints cannot directly
resume with these optimized scripts. No running training processes were changed.
