"""GPU smoke test for dynamic MuonH LR = target(MuonW LR/norm) * raw update."""

import ast
import gzip
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist


FULL_SCRIPT = Path(__file__).with_name(
    "train_small_batch_muonh_muonw_lr_over_update_match_rmsnorm_gamma_"
    "B0128_devB064_lr0p0036_wd0_wsd_seed00.py"
)
REFERENCE = Path(__file__).with_name(
    "reference_rmsnorm_gamma_qknorm_muonw_wd0_wsd_per_tensor_lr_over_norm.jsonl.gz"
)


def validate_reference():
    with gzip.open(REFERENCE, "rt", encoding="utf-8") as handle:
        first = json.loads(next(handle))
        last = first
        rows = 1
        for line in handle:
            last = json.loads(line)
            rows += 1
    assert rows == 20400
    assert first["update_step"] == 1
    assert last["update_step"] == 20400
    assert len(first["target_lr_over_tensor_norms"]) == 73
    print(f"MuonW gzip reference: PASS ({rows} steps, 73 tensors/step)")


def load_optimizers():
    tree = ast.parse(FULL_SCRIPT.read_text(encoding="utf-8"), filename=str(FULL_SCRIPT))
    wanted = {
        "zeropower_via_svd",
        "zeropower_via_newtonschulz5",
        "Adam",
        "AdamH",
        "MuonW",
    }
    body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    body.append(
        ast.parse(
            "zeropower_backends = dict(svd=zeropower_via_svd, "
            "newtonschulz5=zeropower_via_newtonschulz5)"
        ).body[0]
    )
    namespace = {"torch": torch, "dist": dist, "math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(FULL_SCRIPT), "exec"), namespace)
    return namespace["AdamH"], namespace["MuonW"]


def main():
    validate_reference()
    if not torch.cuda.is_available():
        print("SKIP: CUDA is required because the full MuonH optimizer stores updates on CUDA")
        return
    device = "cuda"
    AdamH, MuonH = load_optimizers()
    tied = torch.nn.Parameter(torch.randn(7, 3, device=device))
    first = torch.nn.Parameter(torch.randn(4, 3, device=device))
    second = torch.nn.Parameter(torch.randn(3, 5, device=device))
    targets = {id(tied): 2.50e-3, id(first): 1.30e-3, id(second): 3.10e-3}

    adam = AdamH([tied], lr=0.0, betas=(0.9, 0.95), weight_decay=0.0)
    muon = MuonH(
        [dict(params=[first, second], weight_decay=0.0)],
        lr=0.0,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        backend="svd",
        rank=0,
        world_size=1,
    )
    adam.param_groups[0]["target_lr_over_norm_by_parameter_id"] = {id(tied): targets[id(tied)]}
    muon.param_groups[0]["target_lr_over_norm_by_parameter_id"] = {
        id(first): targets[id(first)], id(second): targets[id(second)]
    }
    tied.grad = torch.randn_like(tied)
    first.grad = torch.randn_like(first)
    second.grad = torch.randn_like(second)
    adam.step()
    muon.step()

    records = {
        id(tied): adam.last_hyperball_records[id(tied)],
        id(first): muon.last_hyperball_records[id(first)],
        id(second): muon.last_hyperball_records[id(second)],
    }
    for parameter in (tied, first, second):
        record = records[id(parameter)]
        actual = record["lr"] / record["raw_update_fro_norm"]
        assert abs(actual - targets[id(parameter)]) < 1e-12, (actual, targets[id(parameter)])
        print(
            f"target={targets[id(parameter)]:.6g} raw_update={record['raw_update_fro_norm']:.6g} "
            f"dynamic_lr={record['lr']:.6g}"
        )
    print("MuonH dynamic LR/raw-update smoke: PASS")


if __name__ == "__main__":
    main()
