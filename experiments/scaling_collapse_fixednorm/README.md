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

Local pre-publication verification checked actual meta-device model parameter counts at all full widths; source AST equality for model, initialization and fixed-RMS functions; all-step two-segment ELR values and separate gamma warmup policy; tied weights; CPU optimizer/projection execution on a tiny model; full-budget arithmetic and schedule endpoints. All five copied trainer files were byte-compared against the locally verified versions and syntax-compiled before publication. This is not full GPU/DDP/Inductor/data/checkpoint validation. No formal training is launched by this task.

## Depth variants (width768)

Two standalone entries preserve the same fan-in fixed-RMS model construction as yesterday. Only depth and the resulting parameter budget/ELR/schedule durations change; no 3/depth residual multiplier or depth-dependent initialization is added. Both default to 2 ranks/device batch64, global batch128. Use gradient accumulation with a smaller --device-batch-size if memory requires it.

| Script | Depth | Parameters | Updates | Tokens | Actual TPP | Peak ELR | Warmup / decay |
|---|---:|---:|---:|---:|---:|---:|---|
| train_fixednorm_w768_d6.py | 6 | 81112320 | 12377 | 1622278144 | 20.000391 | 0.07406130676167542 | 1238 / 11139 |
| train_fixednorm_w768_d24.py | 24 | 208546560 | 31822 | 4170973184 | 20.000201 | 0.046188656539899973 | 3182 / 28640 |

Both use peak ELR squared times updates=67.8888, the same width768/depth12 anchor. Gamma stays at absolute peak LR0.0018, warmup607/1560 updates respectively, then constant. All matrices including tied embedding are fixed to their own initial RMS; gamma is outside projection. Controlled matrix count changes with depth:37/145 respectively.

```bash
python train_fixednorm_w768_d6.py --describe-only
python train_fixednorm_w768_d24.py --describe-only
torchrun --standalone --nproc_per_node=2 train_fixednorm_w768_d6.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w768_d6
torchrun --standalone --nproc_per_node=2 train_fixednorm_w768_d24.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w768_d24
```

Verification covers all five entries; formal multi-GPU runs have not been launched.
