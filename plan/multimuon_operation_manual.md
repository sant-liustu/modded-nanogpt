# MultiMuon Operation Manual

## Defaults

The planned default training configuration is:

```text
muon_momentum = 0.95
muon_slow_momentum = 0.9999
muon_slow_alpha = 1.6
slow_alpha_warmup_steps = num_iterations
slow_momentum_warmup_steps = num_iterations
```

This means slow momentum is introduced gradually over the whole benchmark horizon instead of being fully active at the beginning.

## How to audit the schedule

At step 1:

- `alpha_t` is approximately `slow_alpha / num_iterations`.
- `slow_momentum_t` is near `muon_momentum`.

At step `num_iterations`:

- `alpha_t == muon_slow_alpha`.
- `slow_momentum_t == muon_slow_momentum`.

With `muon_slow_alpha=0`, the combined momentum is exactly the fast momentum and the optimizer should match the original Muon Nesterov input.

## Suggested ablations

Use the default first:

```text
muon_slow_alpha = 1.6
muon_slow_momentum = 0.9999
```

If early training is unstable or update norms look too large, use a conservative alpha ablation:

```text
muon_slow_alpha = 0.4
```

If the slow buffer reacts too slowly for short training runs, test:

```text
muon_slow_momentum = 0.99
```

Keep both warmups aligned to `num_iterations` for benchmark comparability unless the experiment is explicitly about warmup length.

## Review checklist

- Confirm `MultiMuon` still only receives transformer block parameters.
- Confirm AdamW still handles `lm_head` parameters.
- Confirm scheduler order remains `opt.step()` then `sched.step()`.
- Confirm no Adam second moment or bias correction was added to the Muon path.
- Confirm no `(1 - beta)` scaling was added to fast or slow raw momentum.
