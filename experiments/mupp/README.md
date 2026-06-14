# muP Width Sweep Scripts

Run these scripts from the repository root. Each script is a full copy of
`train_gpt2.py` with the model width hard-coded in the file.

| script | n_embd | n_head | head_dim | width_multiplier |
| --- | ---: | ---: | ---: | ---: |
| `train_gpt2_mupp_w384.py` | 384 | 3 | 128 | 0.5 |
| `train_gpt2_mupp_w768.py` | 768 | 6 | 128 | 1.0 |
| `train_gpt2_mupp_w1152.py` | 1152 | 9 | 128 | 1.5 |
| `train_gpt2_mupp_w1536.py` | 1536 | 12 | 128 | 2.0 |
| `train_gpt2_mupp_w3072.py` | 3072 | 24 | 128 | 4.0 |

Shared defaults:

- `scale_base_model = 768`
- `scale_emb = 1.0`
- `save_every = 500`
- `tensor_norm_every = 1`
- `adamw_update_norm_every = 1`
- `activation_probe_every = 0`
- `spectral_norm_estimate_enabled = 0`

## LR Sweep: w384, w768, and w1152

These scripts sweep `embed_learning_rate`. The tied embedding/lm_head optimizer
uses this value directly. The transformer block optimizer uses
`0.5 * embed_learning_rate / width_multiplier`.

| script | n_embd | n_head | width_multiplier | embed/lm_head lr | hidden-block lr |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_gpt2_mupp_w384_lr0p0018.py` | 384 | 3 | 0.5 | 0.0018 | 0.0018 |
| `train_gpt2_mupp_w384_lr0p0027.py` | 384 | 3 | 0.5 | 0.0027 | 0.0027 |
| `train_gpt2_mupp_w384_lr0p0036.py` | 384 | 3 | 0.5 | 0.0036 | 0.0036 |
| `train_gpt2_mupp_w384_lr0p0054.py` | 384 | 3 | 0.5 | 0.0054 | 0.0054 |
| `train_gpt2_mupp_w384_lr0p0072.py` | 384 | 3 | 0.5 | 0.0072 | 0.0072 |
| `train_gpt2_mupp_w768_lr0p0018.py` | 768 | 6 | 1.0 | 0.0018 | 0.0009 |
| `train_gpt2_mupp_w768_lr0p0027.py` | 768 | 6 | 1.0 | 0.0027 | 0.00135 |
| `train_gpt2_mupp_w768_lr0p0036.py` | 768 | 6 | 1.0 | 0.0036 | 0.0018 |
| `train_gpt2_mupp_w768_lr0p0054.py` | 768 | 6 | 1.0 | 0.0054 | 0.0027 |
| `train_gpt2_mupp_w768_lr0p0072.py` | 768 | 6 | 1.0 | 0.0072 | 0.0036 |
| `train_gpt2_mupp_w1152_lr0p0018.py` | 1152 | 9 | 1.5 | 0.0018 | 0.0006 |
| `train_gpt2_mupp_w1152_lr0p0027.py` | 1152 | 9 | 1.5 | 0.0027 | 0.0009 |
| `train_gpt2_mupp_w1152_lr0p0036.py` | 1152 | 9 | 1.5 | 0.0036 | 0.0012 |
| `train_gpt2_mupp_w1152_lr0p0054.py` | 1152 | 9 | 1.5 | 0.0054 | 0.0018 |
| `train_gpt2_mupp_w1152_lr0p0072.py` | 1152 | 9 | 1.5 | 0.0072 | 0.0024 |

## Warmup-Cosine + Weight Decay Variants

For every non-cosine `train_gpt2_mupp_*.py` script, there is a copied
`*_cosine_wd0p1.py` variant.

These variants change only run configuration:

- LR schedule is linear warmup for `warmup_iters = 250`, then cosine decay to
  `0.1` times the peak LR over `cosine_decay_iters = 4850`.
- `weight_decay = 0.1`.
- The tied embedding/lm_head AdamW group uses `weight_decay=args.weight_decay`.
- The transformer block AdamW group also uses `weight_decay=args.weight_decay`.

Because `transformer.wte.weight` is tied to `lm_head.weight`, applying weight
decay to the `lm_head` optimizer group also applies it to the input embedding.

Additional `w1536` warmup-cosine + weight decay LR sweep scripts:

| script | n_embd | n_head | width_multiplier | embed/lm_head lr | hidden-block lr |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_gpt2_mupp_w1536_lr0p0018_cosine_wd0p1.py` | 1536 | 12 | 2.0 | 0.0018 | 0.00045 |
| `train_gpt2_mupp_w1536_lr0p0027_cosine_wd0p1.py` | 1536 | 12 | 2.0 | 0.0027 | 0.000675 |
| `train_gpt2_mupp_w1536_lr0p0036_cosine_wd0p1.py` | 1536 | 12 | 2.0 | 0.0036 | 0.0009 |
| `train_gpt2_mupp_w1536_lr0p0054_cosine_wd0p1.py` | 1536 | 12 | 2.0 | 0.0054 | 0.00135 |
| `train_gpt2_mupp_w1536_lr0p0072_cosine_wd0p1.py` | 1536 | 12 | 2.0 | 0.0072 | 0.0018 |

## Warmup-Cosine + Independent Weight Decay Variants

For every `*_cosine_wd0p1.py` script, there is a copied
`*_cosine_iwd0p1.py` variant.

These variants keep the same model width, head count, LR, warmup-cosine
schedule, monitoring, and base `weight_decay = 0.1`. The only optimizer change
is the transformer block AdamW group:

- tied embedding/lm_head: `weight_decay = args.weight_decay`
- transformer blocks: `weight_decay = args.weight_decay * width_multiplier`

This keeps the hidden-block product `eta * weight_decay` width-constant when
the hidden-block LR is scaled as `eta / width_multiplier`. Because the tied
embedding/lm_head LR is not divided by `width_multiplier`, its weight decay is
not multiplied by `width_multiplier`.

Hidden-block weight decay values with base `weight_decay = 0.1`:

| n_embd | width_multiplier | hidden-block weight_decay |
| ---: | ---: | ---: |
| 384 | 0.5 | 0.05 |
| 768 | 1.0 | 0.1 |
| 1152 | 1.5 | 0.15 |
| 1536 | 2.0 | 0.2 |
| 3072 | 4.0 | 0.4 |
