# muP Width Sweep Scripts

Run these scripts from the repository root. Each script is a full copy of
`train_gpt2.py` with the model width hard-coded in the file.

| script | n_embd | n_head | head_dim | width_multiplier |
| --- | ---: | ---: | ---: | ---: |
| `train_gpt2_mupp_w384.py` | 384 | 3 | 128 | 0.5 |
| `train_gpt2_mupp_w768.py` | 768 | 6 | 128 | 1.0 |
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
