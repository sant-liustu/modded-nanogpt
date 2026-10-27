# Measured diagnostic-overhead optimization

Changes: foreach gradient L2 norms plus aggregate L2; one CPU loss scalar for
both the pre-backward finite check and loss accounting; no repeated checkpoint
serialization at phase end or when reopening an already completed phase.
SGD updates, finite checks, schedule, evaluation frequency and latest-checkpoint
frequency remain unchanged. No live remote process was modified/restarted.

Local isolated benchmark, RTX 3070 Ti Laptop 8GB / PyTorch 2.5.1+cu124:
the full 12-layer width-768 six-head model, context 1024, global batch 4,
one full-batch forward/backward, BF16 autocast, real prepared FineWeb data.
Three alternating-order trials per variant, 20 timed updates per trial after
three warmups. CUDA synchronized at timing boundaries. Timings exclude val and
checkpoint writes. Medians:

| Variant | ms/update |
| --- | ---: |
| Original | 278.787 |
| Loss-scalar change only | 279.281 |
| Foreach norm only | 273.874 |
| Combined | 271.622 |

The combined local reduction was **2.57%** (7.16 ms/update). Loss-scalar-only
improvement is not established by these samples. Laptop timing variation is
visible; do not extrapolate a precise speedup or runtime saving to another GPU
or to larger scheduled batches. An earlier shorter, partly concurrent test
showed 3.6%; the isolated result above is the primary reported measurement.

Correctness: deterministic CPU original-vs-optimized updates produced bitwise
equal parameters and loss across three updates. Maximum relative difference
in the diagnostic norm was 3.76e-6; on the same full-model CUDA gradients in
the isolated run it was zero. Reduction order can change diagnostic rounding.
Injected NaN and Inf gradients and NaN loss are rejected before SGD changes
parameters. Full CPU/CUDA and interrupted-resume tests passed again after the
change. This does not establish a new learning result.

The optimization does not change checkpoint contract fields. Existing checkpoints
remain usable with the same explicit CLI settings; running Python processes
will not pick up this code until restarted/newly launched. No restart is required
merely because the optimized code has been published.

To measure on an otherwise idle server GPU, without accessing another process:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_update.py \
  --train '/data/fineweb_train_*.bin' --output performance.json \
  --steps 20 --rounds 3
```

Further potential work, not claimed or applied: separately time network-mounted
checkpoint writes before changing save cadence; profile forward/backward before
claiming hardware saturation. Compile/CUDA graphs and scientific batch changes
are outside this patch.
