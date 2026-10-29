# Fixed-data w3072/d12, eight GPUs

Extends the fixed-data width ladder with width3072/depth12, head dimension128, tied vocabulary50304. Same global B128, context1024, seed0, sorted unshuffled training shards, 18858 updates (2471755776 consumed tokens), fixed initial matrix RMS, peak matrix ELR0.06, linear warmup1886/decay16972, gamma absolute LR0.0018 and original gamma warmup. This is fixed consumed tokens, not fixed TPP or FLOPs.

Eight DDP ranks, device batch16. Each rank replicates FP32 parameters/Adam state. Loss-only forwards use 1024-token vocabulary chunks with non-reentrant activation checkpointing of each projection/CE. The tied head, width scaling, ignore_index=-1, global valid-token mean and full-logits inference path are preserved. Checkpointing recomputes head work during backward to avoid retaining every chunk's vocabulary activations. Validation retains the inherited grad-enabled context; its chunked head also uses recomputation checkpoints.

Validation: 2097152 tokens = 2048 total sequences, every100 updates plus initial/final/checkpoint points, fixed validation prefix. Complete checkpoints every1000 updates and final; launcher requires a fresh output directory. Historical data and other model scripts are unchanged.

```bash
bash run_w3072_8gpu.sh --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w3072/checkpoints --log-root outputs/w3072/logs --run-name w3072_fixed_data
```

Use a dedicated eight-GPU allocation; do not overlap other launchers. For a bounded first check add `--save-at 2 --stop-after-save` and a fresh checkpoint directory; this covers initial validation, forward/backward, and checkpoint save, but not the step1000 diagnostic peak. Formal full-size CUDA/DDP/compiled memory fit remains unverified. CPU loss/gradient equivalence checks cover chunk tails, ignored labels, tied weights and width scaling.
