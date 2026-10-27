# Fixed-initial-RMS SGD 8-1-1 counterpart

Requested: a separate norm-control version of the existing non-control pair.
This package is isolated from the parent task's working files and Git state.

Question: does LR/batch factorization collapse persist when each 2D parameter
is projected to its own initialization RMS after every SGD optimizer step?
Projection changes the dynamics and its frequency differs between the arms;
collapse is an empirical question, not an invariant guaranteed by B/eta.

Policy: control all unique 2D trainable weights, including tied embedding/head
once. Do not control 1D RMSNorm gamma. No tangent-gradient modification, no WD,
no momentum, no clipping, no norm-dependent LR scaling. Start control from the
first update; use a newly trained norm-controlled prefix, not a plain-SGD prefix.
The two controlled tails share that controlled prefix. Initialization and data
seed match the plain baseline, permitting comparisons at matched data progress.

Model/data: unchanged model snapshot, width 768, depth 12, 6 heads, context 1024;
prepared GPT-2 FineWeb data, seed 0, BF16 autocast and FP32 weights.
Uniform initial LR .01, initial global batch 4; 1/4/10 integer schedule;
1,235,877,888 tokens per trajectory, canonical updates 301728. Full batches by
default; independent validation batch 32 over 512 contexts. Single GPU.

Metrics: existing loss/norm/progress JSONL plus maximum relative matrix RMS
error. Save original targets in checkpoints; reject plain-SGD checkpoint reuse.
Retain the 200000-update milestone. Data cursor resume continues to use mmap.

Implementation: copy the standalone runner and model, add a small projection
function and checkpoint target validation; separate launch paths and README.
No new framework. Review scope of controlled parameters, projection timing,
target persistence, and scientific interpretation of projection discretization.

Verification: small-model end-to-end CPU run, interruption/restore equality,
constant RMS after each update, tied-weight identity, gamma excluded, correct
batch/clock/token pairing. Small CUDA test only if practical; no formal run.
Passing code checks does not establish training stability or loss collapse.
