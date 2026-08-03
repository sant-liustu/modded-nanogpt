"""State-free plain SignSGD for controlled optimizer experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch
from torch import Tensor


class PlainSignSGD(torch.optim.Optimizer):
    """Apply the memoryless update ``parameter -= lr * sign(gradient)``.

    Gradients must already contain the desired accumulated and distributed
    gradient. In particular, DDP gradient synchronization and AMP unscaling
    must happen before :meth:`step` is called.

    This optimizer intentionally has no momentum, Nesterov, shape scaling, or
    weight decay. It stores no per-parameter optimizer state.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, defaults={"lr": lr})
        self._validate_plain_groups()

    def _validate_plain_groups(self) -> None:
        """Reject options that would change plain SignSGD semantics."""
        for group in self.param_groups:
            if float(group.get("momentum", 0.0)) != 0.0:
                raise ValueError("PlainSignSGD requires momentum=0")
            if bool(group.get("nesterov", False)):
                raise ValueError("PlainSignSGD requires nesterov=False")
            if float(group.get("weight_decay", 0.0)) != 0.0:
                raise ValueError("PlainSignSGD requires weight_decay=0")
            if bool(group.get("use_shape_scaling", False)):
                raise ValueError("PlainSignSGD does not support shape scaling")

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], Tensor]] = None,
    ) -> Optional[Tensor]:
        """Perform one plain SignSGD update and return an optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Revalidate because callers and schedulers may edit parameter groups.
        self._validate_plain_groups()
        for group in self.param_groups:
            lr = group["lr"]
            if lr < 0.0:
                raise ValueError(f"Invalid learning rate: {lr}")
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.layout != torch.strided:
                    raise RuntimeError("PlainSignSGD does not support sparse gradients")
                if gradient.is_complex():
                    raise RuntimeError("PlainSignSGD does not support complex gradients")
                parameter.add_(gradient.sign(), alpha=-lr)

        return loss

