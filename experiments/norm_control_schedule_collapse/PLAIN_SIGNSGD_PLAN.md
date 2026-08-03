# Plain SignSGD optimizer plan

## Definition

The optimizer implements only

```text
parameter <- parameter - learning_rate * sign(accumulated_gradient)
```

It is the memoryless plain SignSGD variant: momentum, Nesterov momentum, shape
scaling, and weight decay are all excluded. The optimizer acts after gradient
accumulation, DDP synchronization, and any AMP gradient unscaling performed by
the training loop.

## API and experiment boundary

- `PlainSignSGD(params, lr=...)` follows the `torch.optim.Optimizer` interface.
- Multiple parameter groups and PyTorch LR schedulers are supported.
- Sparse and complex gradients are rejected explicitly.
- No per-parameter state is allocated.
- The optimizer is isolated in `plain_signsgd.py`; it is not yet wired into the
  four norm-control training scripts.

## Verification

The smoke test checks the exact signed update (including a zero gradient),
different parameter-group learning rates, scheduler compatibility, absence of
optimizer state, rejection of non-plain variants, and CUDA bfloat16 behavior
when a GPU is available.
