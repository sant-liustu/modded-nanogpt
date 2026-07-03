# Deprecated MuonW lr=0.02 runs

The earlier MuonW B128 scripts and derived reference files used `muon_learning_rate = 0.02`.
That is not the original modded-nanogpt Muon recipe.

Use `muon_learning_rate = 0.00036`, i.e. `0.1 * embed_learning_rate` for `embed_learning_rate = 0.0036`.
The corrected scripts in this directory keep:

- `batch_size = 128`
- `device_batch_size = 64`
- `num_iterations = 20400`
- `warmup_iters = 1000`
- `warmdown_iters = 5800`
- `weight_decay = 0.0` or `0.1` according to the script name

Do not use the existing `reference_rmsnorm_gamma_muonw_*_block_plus_embedding_total_lr_over_norm.jsonl`
files as correct MuonW baseline references; they were derived from `muonlr0p02` runs.
