"""Tiny CUDA smoke test for the three global-mean-ELR controller policies.

This deliberately does not construct the full NanoGPT model or data loader.
It runs the same controller equations on 72 tiny block matrices plus one tied
embedding/head matrix, executes fused AdamW, and verifies the RMS projection.
"""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "global_mean_elr_reference.jsonl"
STEPS = (1, 2, 3)
POLICIES = ("embed2x", "embed1x", "embedbaselineelr")


def rms(parameter):
    return parameter.detach().float().square().mean().sqrt()


def load_targets():
    targets = {}
    with REFERENCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            step = int(record["update_step"])
            if step in STEPS:
                targets[step] = (
                    float(record["target_block_mean_elr"]),
                    float(record["target_embedding_elr"]),
                )
    if set(targets) != set(STEPS):
        raise RuntimeError(f"missing requested target steps: {sorted(set(STEPS) - set(targets))}")
    return targets


def run_policy(policy, targets, device):
    torch.manual_seed(1729)
    parameters = [torch.nn.Parameter(torch.randn(4, 4, device=device)) for _ in range(72)]
    embedding_index = 12
    embedding = torch.nn.Parameter(torch.randn(8, 4, device=device))
    parameters.insert(embedding_index, embedding)
    initial_rms = [rms(parameter).item() for parameter in parameters]
    optimizer = torch.optim.AdamW(
        [dict(params=[parameter], lr=1.0, weight_decay=0.0) for parameter in parameters],
        betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    max_block_mean_relative_error = 0.0
    max_embedding_relative_error = 0.0
    max_projection_relative_error = 0.0
    max_embedding_lr_ratio_error = 0.0

    for step in STEPS:
        target_block_mean_elr, target_embedding_elr = targets[step]
        rms_values = torch.stack([rms(parameter) for parameter in parameters])
        block_rms = torch.cat((rms_values[:embedding_index], rms_values[embedding_index + 1:]))
        global_block_lr = target_block_mean_elr / torch.mean(1.0 / block_rms).item()
        learning_rates = [global_block_lr] * len(parameters)
        if policy == "embed2x":
            learning_rates[embedding_index] = 2.0 * global_block_lr
        elif policy == "embed1x":
            learning_rates[embedding_index] = global_block_lr
        elif policy == "embedbaselineelr":
            learning_rates[embedding_index] = rms_values[embedding_index].item() * target_embedding_elr
        else:
            raise ValueError(policy)
        for group, learning_rate in zip(optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate

        actual_block_mean_elr = sum(
            learning_rates[index] / value.item()
            for index, value in enumerate(rms_values)
            if index != embedding_index
        ) / 72.0
        max_block_mean_relative_error = max(
            max_block_mean_relative_error,
            abs(actual_block_mean_elr - target_block_mean_elr) / target_block_mean_elr,
        )
        actual_embedding_elr = learning_rates[embedding_index] / rms_values[embedding_index].item()
        if policy == "embedbaselineelr":
            max_embedding_relative_error = max(
                max_embedding_relative_error,
                abs(actual_embedding_elr - target_embedding_elr) / target_embedding_elr,
            )
        else:
            expected_ratio = 2.0 if policy == "embed2x" else 1.0
            max_embedding_lr_ratio_error = max(
                max_embedding_lr_ratio_error,
                abs(learning_rates[embedding_index] / global_block_lr - expected_ratio),
            )

        loss = sum(parameter.square().mean() for parameter in parameters)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for parameter, target_rms in zip(parameters, initial_rms):
                parameter.mul_((target_rms / rms(parameter)).to(dtype=parameter.dtype))
        optimizer.zero_grad(set_to_none=True)
        max_projection_relative_error = max(
            max_projection_relative_error,
            max(abs(rms(parameter).item() - target_rms) / target_rms for parameter, target_rms in zip(parameters, initial_rms)),
        )

    return {
        "policy": policy,
        "max_block_mean_elr_relative_error": max_block_mean_relative_error,
        "max_embedding_elr_relative_error": max_embedding_relative_error,
        "max_embedding_lr_ratio_error": max_embedding_lr_ratio_error,
        "max_rms_projection_relative_error": max_projection_relative_error,
        "max_weight_decay": max(group["weight_decay"] for group in optimizer.param_groups),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fused-AdamW smoke test")
    targets = load_targets()
    results = [run_policy(policy, targets, torch.device("cuda")) for policy in POLICIES]
    print(json.dumps({"reference": str(REFERENCE), "steps": STEPS, "results": results}, indent=2))


if __name__ == "__main__":
    main()
