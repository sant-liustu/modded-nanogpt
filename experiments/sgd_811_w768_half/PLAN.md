# SGD 8-1-1 factorization experiment

Approved scope: user's existing 12-layer, width-768, six-head GPT; FineWeb;
context 1024; all parameters use plain SGD with initial LR 0.01; initial
global batch 4; no momentum, decay, clipping, or weight norm projection.
model.py is a verbatim model-definition snapshot from the existing width-768
learnable-RMSNorm baseline, retaining Q/K norm, RoPE, ReLU squared, tied head.

Question: do fixed-batch/varying-LR and fixed-LR/varying-batch realizations
of the same ratio path produce approximately equal validation losses in T?
Agreement is empirical, not a parameter-trajectory identity.

Schedule: approximately 80% LR 0.01, 10% LR 0.0025, 10% LR 0.001,
with fractions defined in canonical fixed-batch updates (equivalently tokens).
No added warmup. Use fixed multipliers 1, 4, 10, never alternating values.
Round the middle/last phase to the nearest multiple of 4/10 reference updates;
assign the remainder to the prefix to preserve the exact token budget.
For each macro's integer g, the LR arm takes g
updates with batch 4, LR 0.01/g; batch arm takes one update with batch 4*g,
LR 0.01. Both consume exactly the same contiguous contexts per macro and
advance delta T=0.01. This deliberately changes the paper's middle LR level.

Train the shared 80% prefix once. Save model and data position, fork two tails.
Seed 0; BF16 autocast with FP32 parameters; fixed validation contexts.
Global batches use microbatch 1 with correctly weighted gradient accumulation.
Training budget is half of the user's width-768 linear experiment:
1,235,877,888 tokens / 301728 canonical batch-4 updates per trajectory.
The reference phase lengths are 241386, 30172, 30170 updates, respectively.
Smoke runs are explicitly separate. User requests scripts only, no formal local run.

New files only: model.py, train.py, verify.py, PLAN.md, README.md.
No changes to the existing dirty research repository; no new framework.
Metrics: validation CE on common T grid, training CE, T, consumed tokens,
LR, actual global batch, parameter and gradient norms. Compare within matched
schedule pairs. Do not interpret fit agreement as general optimizer theory.
Risks: discretization, insufficient pretraining, SGD instability, data repetition,
different seed noise, and low throughput on the local 8GB GPU. Verify data
size and prohibit silent cycling. Check integer schedule, replay, model identity,
accumulation, and checkpoint continuation before a full run.
