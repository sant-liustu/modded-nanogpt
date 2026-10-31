# Matrix SGD + gamma Adam: three-arm ELR comparison

New independent variant of ../sgd_wsd_elr_collapse. Original suite untouched.

| Launcher | Matrix target RMS | Matrix SGD LR |
|---|---|---|
| run_fixed.sh | r0 | base WSD LR |
| run_sine_p1.sh | r0*s(t) | base WSD LR*s(t) |
| run_sine_p2.sh | r0*s(t) | base WSD LR*s(t)^2 |

s(t)=1+0.5*sin(2*pi*t/301728); exactly one period, range0.5--1.5.
Every unique2D tensor is projected, tied embedding/head once. All49 RMSNorm
1D gamma tensors use Adam, including attention/MLP norms, Q/K norms and final
norm. Gamma is not projected and never receives sine LR compensation.

Gamma Adam: peak LR0.0003, betas(0.9,0.95), eps1e-8, weight_decay0,
foreach=False. Linear warmup over14791 updates, then CONSTANT0.0003 through
training end. Original source used AdamW with zero gamma decay, so the
mathematical update rule is Adam; this runner explicitly uses torch.optim.Adam.
Historical source: experiments/norm_control_schedule_collapse/
train_gpt2_gamma_cosine_wave_lrmatched_baselr_cosine_decay0_B0128_devB064.py,
embed LR0.0036, gamma LR0.5*embed LR, batch128, betas(0.9,0.95), gamma WD0.
Historical warmup was1000steps; this experiment retains the current WSD warmup
length14791. Default0.0003 is user-selected, rounded from historical0.0018*sqrt(4/128)
=0.0003182. It is not a validated optimal LR for batch4. No automatic batch-size scaling.

Matrix SGD remains momentum0/WD0, peak baseLR0.015, WSD warmup14791,
stable ends after215943 updates, then85785 linear decay updates. Width768,
depth12, heads6, context1024, batch4, seed0,301728 updates per arm:
1,235,877,888 tokens. Same initialization/data order for all three arms.
No clipping, compile or DDP; FP32 parameters and CUDA BF16 autocast.
This intervention changes BOTH gamma optimizer and tail LR policy relative to
the original SGD suite; differences cannot be attributed to optimizer alone.

## Launch

```bash
export DATA_DIR=/data/fineweb10B
export OUTPUT_ROOT=runs/sgd_wsd_elr_gamma_adam
bash run_fixed.sh
bash run_sine_p1.sh
bash run_sine_p2.sh
```

To explicitly compare another gamma LR, use a new output root and pass
`--gamma-lr 0.0009` to ALL three launchers. Keep the original baseline results.
Microbatch omitted means full batch4. Requires1,235,877,889 train tokens and
262,145 val tokens, prepared FineWeb uint16 format. No data cycling/downloads.

## Logs and recovery

steps.jsonl keeps every update: loss, overall gradient norm, base/matrix/gamma
LR, norm multipliers and scale-normalized matrix LR. Flush every100updates,
not subsampling. metrics.jsonl every500updates plus initialization, WSD
boundaries and final: val256contexts, batch32 (8forwards), parameter norms,
per-tensor RMS/norm and LR ratios. Gamma LR/norm ratios are descriptive only;
Adam update magnitude also depends on its saved moments. Per-tensor gradient and actual update monitoring is described below.

latest.pt every1000updates, boundaries/final/200000. Permanent stable_end.pt at
215943 BEFORE decay; permanent step200000.pt. Checkpoints include Adam step,
first/second moments, initial RMS targets, model/RNG/cursor and strict contract.
Resume by repeating identical command. No old pure-SGD checkpoint conversion:
use new output directories and train all three arms from initialization.
Direct mmap cursor resume; buffered logs roll back to the restored checkpoint.

## Verification

Run `python verify.py` locally (fixture sources reference Windows FineWeb paths).
Small width16/depth1/context8, full50304 vocabulary,24updates per arm: schedule
checks, exact CPU resumed weights/Adam moments/RNG/logs, permanent stable
checkpoint, stale/partial log recovery, changed-LR rejection, CUDA finite-loss
short runs. verification.json reports executed results. No full768 training or
collapse/convergence claim. Existing training processes are not touched.

## Per-tensor gradient and update monitors

Default every1000updates plus final. Set --gradient-norm-every and
--update-norm-every to1 for every update, or0 to disable each monitor.
Whole-model gradient_norm and loss remain every step.

- gradient_norm_history.jsonl: grad_l2 and grad_rms, after accumulation and
  before optimizer. Reuses L2 reductions already used for overall gradient norm.
- optimizer_update_norm_history.jsonl: raw_parameter_delta_norm (actual delta L2)
  and delta_rms. optimizer_before_projection is actual optimizer displacement;
  total_after_projection includes norm projection. Gamma's second measurement
  reuses the first because gamma is never projected.

RMS=L2/sqrt(numel), computed from the same L2 scalar, without another tensor
reduction. Only identity metadata (step/name/shape/numel, stage, optimizer/LR)
accompanies these two metrics. No maxima, radial/relative/finite or delta/LR
metrics. Existing overall nonfinite checks remain active.
Packed host transfers, buffered logs and checkpoint rollback remain. Update
snapshots cost about one extra FP32 model plus tensor scratch; cadence1 can
increase I/O/runtime. CPU on/off exactness and resume checks plus CUDA short
runs are covered in verify.py; full768 monitoring performance unmeasured.
