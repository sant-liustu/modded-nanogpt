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
