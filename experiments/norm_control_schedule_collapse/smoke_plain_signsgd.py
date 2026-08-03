"""Numerical and API smoke tests for the local PlainSignSGD optimizer."""

from __future__ import annotations

import torch

from plain_signsgd import PlainSignSGD


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_exact_update_and_zero_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    parameter.grad = torch.tensor([4.0, -0.25, 0.0])
    optimizer = PlainSignSGD([parameter], lr=0.1)
    optimizer.step()
    assert_close(parameter, torch.tensor([0.9, -1.9, 3.0]))
    if optimizer.state:
        raise AssertionError(f"PlainSignSGD must be state-free, got {optimizer.state}")


def test_parameter_groups_and_scheduler() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([2.0])
    second.grad = torch.tensor([-3.0])
    optimizer = PlainSignSGD(
        [
            {"params": [first], "lr": 0.25},
            {"params": [second], "lr": 0.5},
        ],
        lr=1.0,
    )
    optimizer.step()
    assert_close(first, torch.tensor([0.75]))
    assert_close(second, torch.tensor([1.5]))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda _: 0.4, lambda _: 0.2])
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.1


def test_forbidden_variants_are_rejected() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    for forbidden_group in (
        {"params": [parameter], "momentum": 0.9},
        {"params": [parameter], "nesterov": True},
        {"params": [parameter], "weight_decay": 0.1},
        {"params": [parameter], "use_shape_scaling": True},
    ):
        try:
            PlainSignSGD([forbidden_group], lr=0.1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"forbidden SignSGD variant was accepted: {forbidden_group}")


def test_cuda_bfloat16_when_available() -> None:
    if not torch.cuda.is_available():
        return
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda", dtype=torch.bfloat16))
    parameter.grad = torch.tensor([2.0, -3.0], device="cuda", dtype=torch.bfloat16)
    optimizer = PlainSignSGD([parameter], lr=0.125)
    optimizer.step()
    assert_close(parameter.float().cpu(), torch.tensor([0.875, -0.875]))


def main() -> None:
    test_exact_update_and_zero_gradient()
    test_parameter_groups_and_scheduler()
    test_forbidden_variants_are_rejected()
    test_cuda_bfloat16_when_available()
    print("PlainSignSGD smoke tests passed")


if __name__ == "__main__":
    main()

