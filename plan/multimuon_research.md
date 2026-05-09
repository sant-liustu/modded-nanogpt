# MultiMuon Research Notes

## Repository state

- Current branch: `experiment/multimuon-v1`.
- The active optimizer implementation is in `train_gpt2.py`.
- The repository currently has a single-file `Muon` class, not a separate optimizer module.
- The training script uses AdamW for the tied embedding/lm head and Muon for `raw_model.transformer.h.parameters()`.
- The Muon path distributes orthogonalized updates by assigning parameters round-robin across DDP ranks, then all-reducing a flat bfloat16 update buffer.

## Existing Muon behavior

The current Muon update uses raw SGD-style momentum:

```text
buf_t = beta * buf_{t-1} + grad_t
g_for_ns = grad_t + beta * buf_t       when nesterov=True
g_for_ns = buf_t                       when nesterov=False
update = NewtonSchulz5(g_for_ns)
```

It does not use EMA normalization, `(1 - beta)` scaling, bias correction, second moments, adaptive denominator, or AdamW-style decoupled weight decay inside the Muon path.

## Target behavior

The change should introduce a fast and slow raw momentum pair while preserving the existing Muon orthogonalization and Nesterov shape:

```text
fast_t = beta_fast * fast_{t-1} + grad_t
slow_t = beta_slow,t * slow_{t-1} + grad_t
combined_t = (fast_t + alpha_t * slow_t) / (1 + alpha_t)
g_for_ns = grad_t + beta_fast * combined_t    when nesterov=True
g_for_ns = combined_t                         when nesterov=False
```

This keeps the original `slow_alpha=0` behavior equivalent to old Muon.

## Scheduled slow components

`alpha_t` uses linear warmup:

```text
alpha_t = slow_alpha * min(step / slow_alpha_warmup_steps, 1.0)
```

`beta_slow,t` uses half-life interpolation from the fast momentum to the final slow momentum:

```text
h(beta) = log(0.5) / log(beta) - 1
h_t = (1 - a_t) * h(beta_fast) + a_t * h(beta_slow_final)
beta_slow,t = 0.5 ** (1 / (h_t + 1))
```

where:

```text
a_t = min(step / slow_momentum_warmup_steps, 1.0)
```

Default experiment values:

```text
beta_fast = 0.95
beta_slow_final = 0.9999
slow_alpha = 1.6
slow_alpha_warmup_steps = args.num_iterations
slow_momentum_warmup_steps = args.num_iterations
```

## Non-goals

- Do not add AdEMAMix second-moment state.
- Do not add Adam-style adaptive denominator.
- Do not bias-correct the slow momentum.
- Do not add Muon weight decay in this change.
- Do not change Newton-Schulz coefficients, bfloat16 update buffer, distributed all-reduce, or the scheduler call order.
