"""AdEMAMix optimizer.

Adapted from Apple ml-ademamix:
https://github.com/apple/ml-ademamix/blob/main/pytorch/ademamix.py

MIT License
Copyright (c) 2024 Apple Inc.

This local copy keeps the algorithm and public arguments aligned with the
reference implementation while using a small amount of modern PyTorch style for
the in-place parameter update.
"""

import math

import torch
from torch.optim import Optimizer


def linear_warmup_scheduler(step, alpha_end, alpha_start=0.0, warmup=1):
    if warmup is None or step >= warmup:
        return alpha_end
    a = step / float(warmup)
    return (1.0 - a) * alpha_start + a * alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0.0, warmup=1):
    def half_life(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def inv_half_life(t):
        return math.pow(0.5, 1 / (t + 1))

    if warmup is None or step >= warmup:
        return beta_end
    a = step / float(warmup)
    return inv_half_life((1.0 - a) * half_life(beta_start) + a * half_life(beta_end))


class AdEMAMix(Optimizer):
    r"""Implements AdEMAMix.

    Args:
        params: iterable of parameters to optimize or parameter groups.
        lr: learning rate.
        betas: coefficients for fast EMA, second moment, and slow EMA.
        alpha: coefficient mixing the slow EMA into the update.
        beta3_warmup: steps used to warm beta3 in half-life space.
        alpha_warmup: steps used to linearly warm alpha.
        eps: denominator epsilon.
        weight_decay: decoupled weight decay, AdamW style.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        alpha=2.0,
        beta3_warmup=None,
        alpha_warmup=None,
        eps=1e-8,
        weight_decay=0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 3:
            raise ValueError(f"AdEMAMix requires three beta values, got {len(betas)}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            alpha_final = group["alpha"]
            beta3_warmup = group["beta3_warmup"]
            alpha_warmup = group["alpha_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_fast"] = (
                        torch.zeros_like(p, memory_format=torch.preserve_format)
                        if beta1 != 0.0
                        else None
                    )
                    state["exp_avg_slow"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_fast = state["exp_avg_fast"]
                exp_avg_slow = state["exp_avg_slow"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                if alpha_warmup is not None:
                    alpha = linear_warmup_scheduler(step, alpha_final, alpha_start=0.0, warmup=alpha_warmup)
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(step, beta3_final, beta_start=beta1, warmup=beta3_warmup)
                else:
                    beta3 = beta3_final

                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    fast = exp_avg_fast
                else:
                    fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1.0 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
                update = (fast.div(bias_correction1) + alpha * exp_avg_slow) / denom
                if weight_decay != 0.0:
                    update = update.add(p, alpha=weight_decay)
                p.add_(update, alpha=-lr)

        return loss
