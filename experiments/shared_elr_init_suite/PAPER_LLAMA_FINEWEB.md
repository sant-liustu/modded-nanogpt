# Standalone LLaMA 0.25B / FineWeb

Run `train_llama025b_fineweb.py` alone: it imports only PyTorch, NumPy and the standard library. No Transformers, local helper files, pretrained weights, norm control, ELR scaling or tied weights. Existing AdamW and Muon entries are unchanged.

```bash
torchrun --standalone --nproc_per_node=4 train_llama025b_fineweb.py \
  --input-bin 'data/fineweb10B/fineweb_train_*.bin' \
  --input-val-bin 'data/fineweb10B/fineweb_val_*.bin' \
  --output logs/llama025b_fineweb_cut20k
```

Output must be a new directory. Input is the existing nanoGPT v1/header+uint16 **GPT-2-tokenized** FineWeb format, vocabulary padded to 50304. No retokenization or silent vocabulary remapping. The local verified data lives at `F:/datasets/fineweb10B-gpt2/raw/`. **Sequential reading, as explicitly requested:** shards are sorted by filename. Each rank reads its disjoint slice of each consecutive global microbatch; all ranks advance together. Short shard tails are skipped, matching the previous loader. Wrap occurs only after all shards are exhausted. Training, validation and Fisher have separate cursors. This deliberately differs from the author's random-window sampler; model/optimizer/schedule settings remain unchanged.

Default settings:

| Item | Value |
| --- | --- |
| Model | 8 layers, width1024, intermediate4096, 16 heads (MHA) |
| Architecture | Pre-RMSNorm (eps1e-6), RoPE(theta10000), SwiGLU, no bias/dropout/QK norm |
| Initialization | All linear/embedding weights N(0,0.02^2), norm gain1, untied Emb/Head |
| Optimizer | Ordinary AdamW, beta(.9,.95), eps1e-8, WD.1 for ALL six groups |
| LR | Uniform peak8e-4; no embedding/head multiplier; global clip1.0 |
| Schedule | Official zero-based schedule: first update LR0, warmup1000, cosine horizon50000, floor4e-5 |
| Cutoff | 20000 **completed** updates; schedule is not compressed to20000 |
| Training | Global480 x context1024; 4 ranks x micro30 x accumulation4 |
| Precision | FP32 master params/optimizer; CUDA BF16 autocast; FP32 CE |
| Validation | Every1000 and cutoff, fixed20 local microbatches/rank, separate deterministic sampler |

Runtime defaults use eager SDPA, not torch.compile. CPU is available for tiny tests via `--device cpu --expected-world-size 1`. Small-test dimensions are CLI overrides; production defaults above reproduce the author's architecture/distributions, not the exact historical initialization RNG stream.

Fisher:

- Steps0,100,1000,2000,...,20000. Global1024 sequences; each rank256 sequences in4 microbatches of64. Two independent pseudo-label draws per point.
- Accumulate private gradients of summed pseudo-label CE divided by global valid-token count N; SUM across ranks BEFORE taking N*gradient^2. No optimizer update or gradient clipping inside measurement.
- Raw per-tensor trace/numel; group mean weighted by parameter count. Groups: **Emb (input only), Head, QK, VO, FFN, Norm**. Keep Head separate when comparing the paper's five bars.
- Vocabulary projection/softmax in token chunks1024; trunk backward once per microbatch. If probe memory is excessive on real hardware, lower `--fisher-micro-batch-size` (global1024 unchanged), or `--fisher-token-chunk-size` (1024 default).
- `fisher_history.jsonl` includes both draws, conditional Monte Carlo SE, initial/current per-tensor RMS, numel, shape, aliases, group and global input hash. SE covers label sampling, not input sampling.
- `fisher_models/step*.pt`: model-only snapshot at every successful probe. `fisher_initial_rms.json`: original initialization RMS.
- Raw arithmetic Fisher is deliberately retained. Paper appendix also mentions log(h); unpublished aggregation/measurement code prevents claiming exact plotting equivalence.

At step20000 a complete rank-specific resume checkpoint is saved under `checkpoints/step020000/`, including model, AdamW moments, all RNG states, sequential data cursor (shard/position/epoch/rank/world/microbatch size), original RMS and strict configuration/source/data metadata. Checkpoints from the old random-window script are intentionally incompatible. `complete.json` appears only after every rank has finished saving. Full resume checkpoints are saved every2000 completed updates by default (2000,4000,...,20000), plus the cutoff if it is not a multiple of2000. Model-only probe snapshots remain at their original frequency. Each full checkpoint has rank0.pt through rank3.pt plus complete.json; all ranks must be present for exact resume. Old two-rank checkpoints cannot be resumed directly as four-rank runs.

Resume on the SAME world size, source, input paths and settings, into a NEW output directory:

```bash
torchrun --standalone --nproc_per_node=4 train_llama025b_fineweb.py \
  --input-bin 'data/fineweb10B/fineweb_train_*.bin' \
  --input-val-bin 'data/fineweb10B/fineweb_val_*.bin' \
  --resume logs/llama025b_fineweb_cut20k/checkpoints/step020000 \
  --output logs/llama025b_fineweb_continue --stop-step 30000
```

Validation evidence (small shapes only):

- `verify_paper_llama.py`: same-weight strict state-dict load and forward/backward comparison against HF Transformers4.45.2 LLaMA; CPU FP32, CUDA FP32, CUDA BF16. Fisher versus full differentiable objective: max relative error ~1.22e-8 FP32 CUDA and5.17e-4 BF16 CUDA.
- Real FineWeb, width32/depth1/context8/globalbatch2, four CUDA updates: Fisher-on/off model/optimizer/data cursor/global RNG identical; intermediate checkpoint resumes exactly; clipping exercised; every probe snapshot and original/current RMS checked.
- `verify_paper_llama_distributed.py`: four actual CPU/Gloo processes, serial objective versus distributed Fisher, then actual standalone entry training+measurement+validation+checkpoint; rank weights equal; compare actual DDP accumulated AdamW updates with serial global-batch AdamW. Four-rank tests use the production batch arithmetic (4 ranks x30 x4=480), tiny model/context4, scaled checkpoint interval2, and restore from step2 to4. RNG/cursors match exactly; floating parameters/moments are compared within tight tolerances because fresh DDP buckets can change floating-point reduction order after resume.
- Test-only HF uses isolated tokenizers0.20.3/huggingface-hub0.26.5 to satisfy local Transformers4.45.2; those packages are **not runtime dependencies** of the delivered standalone script.
- No full 237M training, four-GPU NCCL, or full-length production memory test ran locally. These correctness checks do not establish cloud speed, peak memory, or reproduction of the paper's empirical ordering.

Sources:

- https://arxiv.org/html/2502.19002v2#A1.SS1
- https://github.com/wang-jinbo/BlockwiseLearningRate/blob/main/modelling_llama_new.py
- https://github.com/wang-jinbo/BlockwiseLearningRate/blob/main/train_llama_bw.py

Every optimizer update also writes one record to `tensor_grad_norm_history.jsonl`, after accumulation/DDP averaging and before global clipping. `parameters[name]` contains the pre-clip L2 `grad_norm`, `grad_rms=grad_norm/sqrt(numel)`, `squared_norm_fraction=grad_norm²/sum_tensor_norm²`, group, shape and numel. Fractions sum to1 for nonzero gradients. The record includes `step`, pre-clip `global_grad_norm`, `clip_threshold`, actual PyTorch `clip_coef=min(1,1/(global_grad_norm+1e-6))`, and `clipped` (coefficient below1). All tensors share this scaling; their post-clip norm is approximately their recorded norm times the coefficient. `train_history.jsonl` includes the same coefficient/flag for quick screening. Values are scalars, not saved gradient tensors. Logging is rank0-only and does not change gradients or optimizer updates. `verify_tensor_grad_norms.py` checks known/zero gradients, sum-of-squares reconstruction, and exact training-state agreement with per-tensor observation disabled (portable, without requiring a local Git commit).

Sequential-reader verification additionally checks exact next-token windows, disjoint rank slices, shard transition/tail dropping, full-corpus wrap, and cursor restoration.
