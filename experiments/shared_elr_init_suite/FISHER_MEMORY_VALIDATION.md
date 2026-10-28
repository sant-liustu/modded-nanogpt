# Fisher OOM fix and evidence, 2026-10-28

The cloud step-zero OOM exposed a full-length memory gap: previous validations
used short inputs and did not establish B64/T1024/two-GPU memory feasibility.
The old probe explicitly disabled autocast, computed unused real-label CE,
copied full logits by boolean indexing and retained full softmax probabilities.
The reported 76 GiB cannot be precisely decomposed without the cloud allocation
trace; the source-level redundant buffers and precision mismatch are verified.

Fix: CUDA BF16 autocast matches training. GPT exposes final hidden states without
head/real CE. Vocabulary projection, categorical sampling and pseudo-label CE
are bounded to token chunks (default 1024). Probability buffers are freed before
CE backward. Head chunks immediately accumulate direct head-weight and hidden
gradients; a single chain-rule backward through the trunk adds embedding/trunk
contributions, including shared embedding/head weights. Global N, 1024 sequences,
micro64, two ranks, two independent draws and SUM-before-square are preserved.

PASS verify_fisher_chunked.py on actual tied GPT (w128/d1/vocab32), masks and
partial chunks: compares full gradient vectors AND each tensor's Fisher trace
against a complete differentiable objective with identical labels. Max relative
trace differences: CPU FP32 5.83e-9, CUDA FP32 2.03e-9, CUDA BF16 2.32e-4.
Checks private gradients, RNG/modes and head-call token bounds.

PASS verify_fisher_distributed.py on two CPU/Gloo processes using the hidden/head
path: global scaling, seed separation, serial agreement and rank equality.
PASS verify_fisher_accumulation.py on generic fallback and independent draws.

PASS real FineWeb single-CUDA eager w128/d1/T8 four-update tests, token chunk4,
two draws, off/on equality, model-snapshot loading/RMS and exact resume:
- local_checks/fisher_train_20261028_204719/report.json
- local_checks/fisher_train_muon_20261028_204746/report.json

CUDA allocated-memory comparison, w128/d1/vocab100278/B2/T1024, token chunk1024,
one draw, eager and no optimizer state:

| implementation | peak allocated bytes |
|---|---:|
| old FP32 | 5,072,124,416 |
| old BF16-only | 5,105,135,616 |
| new chunked FP32 | 1,793,350,144 |
| new chunked BF16 | 1,613,369,344 |

Report: local_checks/fisher_memory_20261028_204553.json. Small-case peak fell
from about 4.72 GiB to 1.50 GiB (about 68%). This is a memory-path comparison,
not a speed benchmark, 80GB full-run guarantee or statistical accuracy study.
Full cloud w768/B64/T1024/NCCL/compiled-training memory remains unmeasured here.

Source now logs the head path, precision and chunk size BEFORE probing, and
stores them in successful Fisher records. Different precision/chunking changes
random-number consumption; do not require old/new sample values to match.
This source/config change retains strict checkpoint compatibility guards.
