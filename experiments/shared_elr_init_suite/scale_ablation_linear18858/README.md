# AdamW fixed-norm scale ablations, w768/d12

Four entrypoints share train_adamw_matched.py. Keep all five Python files together.

| Entrypoint | Architecture change |
| --- | --- |
| train_hg_w768_d12_elr0p06_linear18858.py | independent Q/K/V input scales |
| train_hg_dp_w768_d12_elr0p06_linear18858.py | HG plus scale-only dual placement around block matrices; no added activation norms |
| train_sandwich_w768_d12_elr0p06_linear18858.py | RMSNorm on complete attention/MLP branch outputs, before residual addition; no HG/DP |
| train_sandwich_or_w768_d12_elr0p06_linear18858.py | sandwich; all gamma=beta*alpha/RMS(alpha) |

Common configuration: width768/depth12/6 heads, ReLU-squared MLP, original QK norm and tied embedding/head. AdamW betas0.9/0.95, WD0, fixed initial matrix RMS. GlobalB128, 2 GPUs x64, context1024, seed0. Matrix ELR0.06, linear warmup1886 then linear decay16972 to zero at update18858. Embedding/head ELR1x. Gamma and all added scales (including OR alpha/beta) use absolute LR0.0018 with warmup924 then constant, WD0, no norm projection. Matrix LR uses saved target RMS, matching historical fixed-training-budget baseline.

Two ablation branches: baseline->HG->HG+DP and baseline->sandwich->sandwich+OR. DP adds scale only; Q/K/V output scales are length128 shared among heads before existing QK norm/RoPE. Projection scales cover full feature dimensions. No SwiGLU conversion, no IWD, no extra logits normalization. OR alpha=ones/beta=1 initially. All common matrices initialize identically. HG/DP initially match baseline; sandwich changes initial function.

From a data-enabled working directory:

```bash
torchrun --standalone --nproc_per_node=2 /path/to/train_hg_w768_d12_elr0p06_linear18858.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin'
```

Replace the entrypoint for the other three. Use identical data shards/order across runs. Inspect settings without training:

```bash
python train_hg_w768_d12_elr0p06_linear18858.py --describe-only
```

Periodic validation defaults off, matching historical train-loss export. To compare NEW validation curves, consistently add --val-loss-every 100 (historical baseline has no validation curve). Old actual shard list was not recorded in the exported log, so exact historical data equivalence remains unconfirmed. For a same-trainer baseline use train_adamw_matched.py --scale-variant baseline with the same launch/data arguments.

Local validation: full18858 matrix/gamma schedules equal historical source; original loader/projection retained; formal parameter counts checked. Private synthetic single-CUDA eager width128/depth1/vocab256/B1/T16/32-update tests passed for baseline and four variants. New baseline matches historical trainer weights/optimizer states bitwise. Fixed RMS error <1e-7; finite gradients including scalar beta; sandwich+OR full-state resume bitwise equal. Formal two-rank/compiled training and long-run loss differences are unverified. No data, logs, checkpoints or synthetic tests are included in this publication.

## Added HG+DNP and HG+DNP+OR

New entries: train_hg_dnp_w768_d12_elr0p06_linear18858.py and train_hg_dnp_or_w768_d12_elr0p06_linear18858.py. They retain the exact same historical linear18858/ELR0.06/embedding1x/gamma0.0018 protocol.

HG+DNP has identical parameter names, shapes and initial values to HG+DP. The only model difference is enabling parameter-free RMS normalization immediately before six existing output-side scales: Q/K/V output (per-head128), attention output projection, MLP expansion, MLP output projection (last full feature dimension). Existing input scales are unchanged; original Q/K RMSNorm remains AFTER these output-scale sites, before RoPE, in both DP and DNP. This deliberate retained-QK setup keeps the DP/DNP intervention isolated; it is an adaptation to this QK-normalized model. Input-only scales at attention/MLP output matrices remain scale-only, exactly as in the existing DP control. No branch-level sandwich norms are added.

HG+DNP+OR uses the same placement but all gamma (including shared-head Q/K norm and final norm) become beta*alpha/RMS(alpha). Alpha=ones, beta=1 initially. Alpha and scalar beta each use the unchanged absolute gamma LR schedule; no matrix projection or WD on them. Formal parameter counts: HG+DNP123714816; HG+DNP+OR123714985. No IWD or SwiGLU modification.

Launch with torchrun --standalone --nproc_per_node=2 <new_entry.py>, using exactly the same --input-bin and --input-val-bin as the existing controls. Neither entry starts training until invoked.

Validation: preexisting five variants have bitwise-identical initial weights, logits and gradients after extension. New DNP matches DP parameter sites exactly, enabling normalization at exactly six output scales per block. Single-CUDA synthetic eager width128/depth1/B1/T16/32updates passed for both additions; fixed-RMS relative error <1e-7, finite gradients, preserved gamma LR and full-state OR resume. Formal two-rank/compiled training remains unverified.
