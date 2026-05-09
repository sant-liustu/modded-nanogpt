# MultiMuon Implementation Plan

## Scope

Implement MultiMuon inside `train_gpt2.py` with the smallest practical diff:

- Rename `Muon` to `MultiMuon`.
- Preserve the distributed update path and Newton-Schulz backend.
- Replace one raw momentum buffer with fast and slow raw momentum buffers.
- Add warmup schedules for slow momentum strength and coefficient.

## Code changes

1. Add `import math` near the existing standard-library imports.
2. Add small helper functions near the optimizer:
   - `_check_momentum_beta(name, beta)` validates `0 < beta < 1` for half-life scheduling.
   - `_linear_warmup(final_value, step, warmup_steps)` returns the scheduled alpha.
   - `_momentum_half_life(beta)` maps beta to half-life.
   - `_beta_from_momentum_half_life(half_life)` maps half-life back to beta.
   - `_scheduled_slow_momentum(momentum, slow_momentum, step, warmup_steps)` does half-life interpolation.
3. Rename `class Muon` to `class MultiMuon` and update the docstring to describe fast/slow raw momentum.
4. Extend `MultiMuon.__init__` defaults:
   - `slow_momentum=0.9999`
   - `slow_alpha=1.6`
   - `slow_alpha_warmup_steps=1`
   - `slow_momentum_warmup_steps=1`
5. Store new hyperparameters in optimizer defaults, not as global variables.
6. In `step()`:
   - Increment `group['step']` once per optimizer step.
   - Compute `alpha_t` and `slow_momentum_t` once per group.
   - Initialize `fast_momentum_buffer` and `slow_momentum_buffer`.
   - Update both buffers with raw momentum.
   - Form the normalized combined momentum.
   - Keep Nesterov as `grad + momentum * combined`.
7. Add `muon_momentum`, `muon_slow_momentum`, and `muon_slow_alpha` to `Hyperparameters`.
8. Instantiate `MultiMuon` with those fields and set both warmup lengths to `args.num_iterations`.

## Compatibility

- `slow_alpha=0` should reduce to the previous Muon momentum path.
- Existing checkpoints with `momentum_buffer` are not migrated by this minimal implementation; new runs create `fast_momentum_buffer` and `slow_momentum_buffer`.
- The AdamW optimizer for `lm_head` is unchanged.

## Validation

Run:

```bash
python -m py_compile train_gpt2.py
git diff --check
```

Optional GPU smoke test if data and hardware are available:

```bash
./run.sh
```
