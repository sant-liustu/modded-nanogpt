# WSD SGD: three-arm ELR test

| Launcher | Matrix RMS | Matrix LR | Gamma LR |
|---|---|---|---|
| run_fixed.sh | initial RMS r0 | eta_WSD | eta_WSD |
| run_sine_p1.sh | r0 * s | eta_WSD * s | eta_WSD |
| run_sine_p2.sh | r0 * s | eta_WSD * s^2 | eta_WSD |

s(u)=1+0.5 sin(2 pi u), one complete period over training, u=tokens/total_tokens.
Starts and ends at1, reaches1.5 at25% and0.5 at75%. All unique trainable
2D tensors (including tied embedding/head once) are projected after every
SGD update. 1D gamma remains freely trained. Each tensor has its own initial
RMS target; this is not a constant global parameter norm.

For a scale-invariant objective L(cW)=L(W), its gradient scales as1/c.
The direction update then scales as eta/c^2. Thus p2 is the scale-invariant
SGD prediction; p1 is the empirical alternative. This entire GPT is NOT
scale invariant to independently rescaling all controlled tensors: projection
also changes its function. Failure to collapse does not disprove the conditional
eta/norm^2 prediction. Compare loss first on equal tokens/updates, not sum of
the different raw learning rates. RMS and Frobenius norm give the same scale
compensation within each fixed-shape tensor; they differ across tensor sizes.

## Configuration

Width768, depth12, heads6, context1024, batch4, seed0,301728 updates:
1,235,877,888 tokens PER ARM. Three complete runs, no shared trained prefix.
Same seed gives same starting weights; same mmap tape gives identical token
order. Base peak LR .01, pure SGD, momentum0, weight decay0, no clipping.
No compile or DDP. FP32 parameters, BF16 CUDA autocast; unchanged model.py.

WSD reproduces old w384 linear schedule proportions:4000/81600 warmup,
23200/81600 decay. At default budget:14791 warmup updates,85785 decay updates,
decay begins after215943 updates. The last update uses peakLR/decay_steps;
the conceptual schedule reaches zero after the training interval.
LR .01 is the peak, not the first warmup update LR.

Each update uses pre-update norm s(i/N), matrixLR=baseLR*s(i/N)^p, and projects
the result to s((i+1)/N). Gamma always receives baseLR. No tangent projection.

## Run

```bash
export DATA_DIR=/data/fineweb10B
export OUTPUT_ROOT=runs/sgd_wsd_elr
bash run_fixed.sh
bash run_sine_p1.sh
bash run_sine_p2.sh
```

Launch separately on available GPUs if desired. Each launcher has a separate
output directory. Supply `--width 384 --heads 6` etc. to change size, using a
different OUTPUT_ROOT. Omit microbatch for full batch4; use `--microbatch 1`
only if necessary. Do not change experiment flags when resuming.

Need at least1,235,877,889 training tokens and524,289 validation tokens.
No downloads or implicit data recycling. Uses the same prepared uint16 shard
format as the previous runs. Resume by repeating the original command.
Old8-1-1 checkpoints are incompatible with this new WSD experiment.

## Records and recovery

- steps.jsonl: EVERY optimizer update, pre-update train loss/gradient norm,
  base/matrix/gamma LR, pre/post norm multiplier, matrixLR/s and matrixLR/s^2,
  cumulative matrix LR, tokens and batch. Buffers flush every100updates and
  before checkpoint; this is not subsampling.
- metrics.jsonl: fixed512 validation contexts,32 per forward (16 forwards),
  global parameter norm, each tensor's measured Frobenius norm/RMS, LR/norm,
  LR/norm^2, LR/RMS and LR/RMS^2, target RMS and projection error.
  Every200updates; every20 within200updates of warmup end/decay start;
  also initialization, boundaries and final. Diagnostics use current weights
  and the next update LR (final uses last base LR), explicitly labeled.
- latest.pt: every1000updates, WSD boundaries, final and update200000.
  step200000.pt is permanent. Weights, initial norm targets, data offset,
  RNG, cumulative LR and strict experiment contract are restored.
- Recovery directly indexes mmap data; no token replay. JSONL recovery streams
  through log files to remove entries newer than the checkpoint and a partial
  final line. It does not load the entire log into memory.

Train/validation records are denser than checkpoints; interruption can discard
records after the last saved checkpoint. No formal training launched locally.

## Validation

`python verify.py` uses local FineWeb first tokens, width16/depth1/context8,
24updates per arm. It checks schedule matching, bounds, projection, exact CPU
continuous/resumed weights, RNG, data position and logs; stale and partial log
recovery; changed-LR rejection; and CUDA short runs if available. See
verification.json. The test fixture source paths are local Windows paths;
adjust these paths to run the verifier on another machine. Production launchers
use DATA_DIR and have no dependency on those test paths.

These checks verify implementation only, not ELR collapse or formal model
convergence. Existing training processes and original scripts are untouched.

Local validation on2026-10-28 passed for all three arms: exact CPU resume and
logs,24 CUDA updates each, maximum projection relative error1.84e-7. All three
Bash launchers also completed24 CPU updates with tiny model overrides. These
tests used the full50304 vocabulary and real FineWeb tokens. model.py is copied
verbatim from the existing runner (including its trailing blank line).
