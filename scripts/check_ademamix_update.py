from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from ademamix import AdEMAMix  # noqa: E402


def alpha_at(step: int, alpha_end: float, warmup: int | None) -> float:
    if warmup is None or step >= warmup:
        return alpha_end
    a = step / float(warmup)
    return a * alpha_end


def beta3_at(step: int, beta1: float, beta3_end: float, warmup: int | None) -> float:
    if warmup is None or step >= warmup:
        return beta3_end

    def half_life(beta: float, eps: float = 1e-8) -> float:
        return math.log(0.5) / math.log(beta + eps) - 1

    def inv_half_life(t: float) -> float:
        return math.pow(0.5, 1 / (t + 1))

    a = step / float(warmup)
    return inv_half_life((1.0 - a) * half_life(beta1) + a * half_life(beta3_end))


def manual_ademamix_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg_fast: torch.Tensor,
    exp_avg_slow: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    *,
    lr: float,
    betas: tuple[float, float, float],
    alpha: float,
    beta3_warmup: int | None,
    alpha_warmup: int | None,
    eps: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta1, beta2, beta3_final = betas
    beta3 = beta3_at(step, beta1, beta3_final, beta3_warmup)
    alpha_t = alpha_at(step, alpha, alpha_warmup)

    exp_avg_fast = exp_avg_fast * beta1 + grad * (1.0 - beta1)
    exp_avg_slow = exp_avg_slow * beta3 + grad * (1.0 - beta3)
    exp_avg_sq = exp_avg_sq * beta2 + grad.square() * (1.0 - beta2)

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + eps
    update = (exp_avg_fast / bias_correction1 + alpha_t * exp_avg_slow) / denom
    param = param - lr * (update + weight_decay * param)
    return param, exp_avg_fast, exp_avg_slow, exp_avg_sq


def main() -> None:
    torch.manual_seed(0)
    initial = torch.tensor([1.25, -0.5, 0.125], dtype=torch.float64)
    gradients = [
        torch.tensor([0.30, -0.20, 0.10], dtype=torch.float64),
        torch.tensor([-0.40, 0.05, 0.20], dtype=torch.float64),
        torch.tensor([0.10, 0.30, -0.25], dtype=torch.float64),
    ]
    kwargs = dict(
        lr=0.05,
        betas=(0.2, 0.7, 0.9),
        alpha=2.0,
        beta3_warmup=4,
        alpha_warmup=3,
        eps=1e-8,
        weight_decay=0.1,
    )

    param = torch.nn.Parameter(initial.clone())
    optimizer = AdEMAMix([param], **kwargs)
    manual = initial.clone()
    exp_avg_fast = torch.zeros_like(initial)
    exp_avg_slow = torch.zeros_like(initial)
    exp_avg_sq = torch.zeros_like(initial)

    for step, grad in enumerate(gradients, start=1):
        param.grad = grad.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        manual, exp_avg_fast, exp_avg_slow, exp_avg_sq = manual_ademamix_step(
            manual,
            grad,
            exp_avg_fast,
            exp_avg_slow,
            exp_avg_sq,
            step,
            **kwargs,
        )
        torch.testing.assert_close(param.detach(), manual, rtol=1e-12, atol=1e-12)

    state = optimizer.state[param]
    assert state["step"] == len(gradients)
    torch.testing.assert_close(state["exp_avg_fast"], exp_avg_fast, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state["exp_avg_slow"], exp_avg_slow, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state["exp_avg_sq"], exp_avg_sq, rtol=1e-12, atol=1e-12)
    print("AdEMAMix update check passed")


if __name__ == "__main__":
    main()
