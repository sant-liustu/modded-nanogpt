"""Two-process DDP ordering smoke test for the inline PlainSignSGD class."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


SCRIPT = Path(__file__).with_name(
    "train_gpt2_gamma_signsgd_delayed_constant_wsd_B0128_devB064.py"
)


def load_optimizer_class():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlainSignSGD"
    ]
    if len(classes) != 1:
        raise RuntimeError(f"expected one PlainSignSGD class in {SCRIPT}")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace["PlainSignSGD"]


def run_rank(rank: int, world_size: int, store_path: str) -> None:
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -1.0]]))
    model = DDP(model)

    # Local gradients are [1, 0] and [-3, 2]. DDP must average the raw
    # gradients to [-1, 1] before PlainSignSGD takes their signs.
    x = torch.tensor([[1.0, 0.0]]) if rank == 0 else torch.tensor([[-3.0, 2.0]])
    model(x).sum().backward()
    expected_synced_gradient = torch.tensor([[-1.0, 1.0]])
    torch.testing.assert_close(model.module.weight.grad, expected_synced_gradient)

    PlainSignSGD = load_optimizer_class()
    optimizer = PlainSignSGD(model.parameters(), lr=0.25)
    optimizer.step()
    expected_parameter = torch.tensor([[1.25, -1.25]])
    torch.testing.assert_close(model.module.weight, expected_parameter)
    if optimizer.state:
        raise AssertionError(f"PlainSignSGD must remain state-free: {optimizer.state}")

    gathered = [torch.empty_like(model.module.weight) for _ in range(world_size)]
    dist.all_gather(gathered, model.module.weight)
    for parameter in gathered:
        torch.testing.assert_close(parameter, expected_parameter)
    if rank == 0:
        print("PlainSignSGD two-rank DDP smoke test passed")
    dist.destroy_process_group()


def main() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory(prefix="signsgd_ddp_") as directory:
        store_path = str(Path(directory) / "store")
        torch.multiprocessing.spawn(
            run_rank,
            args=(world_size, store_path),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
