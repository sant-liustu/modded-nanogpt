# Paper LLaMA 0.25B / FineWeb reproduction plan

Question: Does the paper blockwise Fisher ordering appear with its LLaMA architecture and ordinary AdamW baseline when OpenWebText is replaced by existing FineWeb data?

## Controls
- New isolated runner; preserve all existing AdamW/Muon runners.
- Standalone PyTorch implementation matching author LLaMA builder: hidden 1024, intermediate 4096, 8 layers, 16 heads, untied embedding/head, original initialization; no custom width scaling, QK norm, or ReLU-squared.
- Record Transformers version and resolved config; inherited HF defaults must be explicit. Public code may not pin the historical environment.
- Ordinary AdamW: beta=(0.9,0.95), eps=1e-8, weight decay=0.1 for every group including embedding/head/norm, global gradient clip=1.0. No norm projection, ELR scaling, embedding multiplier or blockwise LR.
- Uniform peak LR=8e-4; warmup=1000; cosine floor=4e-5; horizon=50000; stop at 20000 completed updates (9.8304B tokens), retaining the original cosine horizon. Match official zero-based scheduler convention and document update-count boundary.
- Global training batch=480, context=1024. Two ranks with local microbatch=8 and accumulation=30 (480 total); memory feasibility requires production GPU check.
- Existing FineWeb paths, tokenizer and vocabulary must be inspected before finalizing. Vocabulary changes count as an additional deviation; never silently truncate or remap token IDs.

## Measurement
- Reuse validated pseudo-label Fisher estimator: 1024 global sequences, local microbatch up to64, two independent label draws per point; all-reduce accumulated gradients before squaring. Chunk vocabulary head to bound memory.
- Measure at step0,100,1000, then every1000 and final. Retain original/current per-tensor RMS, raw trace/numel, per-draw statistics, SE, snapshots and final full resume checkpoint.
- Extend HF parameter mapping to Emb, Head, QK, VO, FFN, Norm. Compare paper five groups with input Emb only; show Head separately.
- Current chunk helper assumes custom GPT hidden API and width multiplier; adapt for HF hidden outputs with no output scaling. Verify chain-rule gradients for untied embedding/head against full-logits reference.
- Primary saved metric remains raw arithmetic Fisher. Paper appendix also mentions log(h); exact plotting aggregation is unavailable. Do not present arithmetic results as proof of matching unpublished plotting implementation. Consider per-coordinate log summaries as separately named supplemental measurements if needed.

## Implementation and checks after plan approval
- Add standalone paper LLaMA runner/launcher and pinned environment metadata; extend diagnostics only where necessary without changing old paths.
- Tiny CPU/CUDA model: full-vs-chunk gradient comparison, correct module coverage, AdamW update reference including clipping, measurement isolation, snapshot and exact resume.
- Small local end-to-end run only; no full local training and no cloud submission until launch configuration is reviewable.
- Review: actual tokenizer/vocab, untied initialization, complete parameter grouping, global batch arithmetic, scheduler, Fisher normalization and aggregation.

Expected: paper ordering may emerge by1000 steps. Counter-result: ordering absent despite aligned training; this would leave data sensitivity and unpublished measurement details unresolved rather than prove an implementation error.

Sources: https://arxiv.org/html/2502.19002v2#A1.SS1 ; https://github.com/wang-jinbo/BlockwiseLearningRate/blob/main/modelling_llama_new.py ; https://github.com/wang-jinbo/BlockwiseLearningRate/blob/main/train_llama_bw.py

User approved implementation, then requested a single self-contained PyTorch file with no Transformers dependency. Training samples random windows with replacement as in the official script; 10B processed tokens do not imply one unique-data pass. Test-only HF comparison may use an isolated dependency directory.

2026-10-31 user amendment: use the previous sequential shard reader, not random-window sampling. Training, validation and Fisher use isolated deterministic cursors; persist shard/position/epoch/rank/world/batch size for exact resume. Keep all model/optimizer/LR/batch/stop/Fisher-estimator settings fixed.

2026-10-31 gradient attribution: log each tensor pre-clip L2 norm, RMS and squared-norm fraction every update, after DDP/accumulation. Rank0 only, read-only reductions, no new abstraction. Save actual global clipping coefficient and flag with step. Validate known gradients, zero gradients, real short-run log coverage, and unchanged training state.
