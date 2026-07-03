#!/usr/bin/env python3
"""Build MuonW lr/norm reference JSONL from a completed base run log."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RMSNORM_GAMMA_SUFFIXES = (
    ".q_norm.weight",
    ".k_norm.weight",
    ".attn_norm.weight",
    ".mlp_norm.weight",
)


def canonical_name(name: str) -> str:
    if name.startswith("_orig_mod."):
        return name[len("_orig_mod.") :]
    return name


def is_rmsnorm_gamma_name(name: str) -> bool:
    name = canonical_name(name)
    return name == "final_norm.weight" or name.endswith(RMSNORM_GAMMA_SUFFIXES)


def is_denominator_parameter(name: str) -> bool:
    name = canonical_name(name)
    return name == "transformer.wte.weight" or (
        name.startswith("transformer.h.") and not is_rmsnorm_gamma_name(name)
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc


def load_expected_denominator_count(run_dir: Path) -> int | None:
    metadata_path = run_dir / "tensor_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return sum(1 for record in metadata if is_denominator_parameter(record["name"]))


def load_denominator_norms(run_dir: Path) -> tuple[dict[int, float], dict[int, int]]:
    tensor_history = run_dir / "tensor_norm_history.jsonl"
    if not tensor_history.exists():
        raise FileNotFoundError(f"missing {tensor_history}")

    sum_squares_by_step: dict[int, float] = {}
    counts_by_step: dict[int, int] = {}
    for record in iter_jsonl(tensor_history):
        name = record.get("name")
        if not isinstance(name, str) or not is_denominator_parameter(name):
            continue
        step = int(record["step"])
        fro_norm = float(record["fro_norm"])
        sum_squares_by_step[step] = sum_squares_by_step.get(step, 0.0) + fro_norm * fro_norm
        counts_by_step[step] = counts_by_step.get(step, 0) + 1

    denominator_by_step = {
        step: math.sqrt(sum_square) for step, sum_square in sum_squares_by_step.items()
    }
    if not denominator_by_step:
        raise ValueError(f"found no denominator parameter norms in {tensor_history}")
    return denominator_by_step, counts_by_step


def load_muon_lr_by_update_step(run_dir: Path) -> dict[int, float]:
    update_history = run_dir / "muonw_update_norm_history.jsonl"
    if not update_history.exists():
        raise FileNotFoundError(f"missing {update_history}")

    lr_by_step: dict[int, float] = {}
    for record in iter_jsonl(update_history):
        if int(record.get("optimizer_index", -1)) != 1:
            continue
        update_step = int(record["step"])
        lr = float(record["lr"])
        old_lr = lr_by_step.get(update_step)
        if old_lr is not None and abs(old_lr - lr) > 1e-15:
            raise ValueError(
                f"MuonW lr is not unique at update_step={update_step}: {old_lr} vs {lr}"
            )
        lr_by_step[update_step] = lr

    if not lr_by_step:
        raise ValueError(f"found no MuonW optimizer_index=1 lr records in {update_history}")
    return lr_by_step


def build_reference(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir does not exist: {run_dir}")

    denominator_by_step, counts_by_step = load_denominator_norms(run_dir)
    lr_by_update_step = load_muon_lr_by_update_step(run_dir)
    expected_count = load_expected_denominator_count(run_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    missing_norm_steps: list[int] = []
    bad_count_steps: list[tuple[int, int]] = []
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for update_step in sorted(lr_by_update_step):
            pre_update_step = update_step - 1
            denominator = denominator_by_step.get(pre_update_step)
            if denominator is None:
                missing_norm_steps.append(pre_update_step)
                continue
            parameter_count = counts_by_step[pre_update_step]
            if expected_count is not None and parameter_count != expected_count:
                bad_count_steps.append((pre_update_step, parameter_count))
            reference_muon_lr = lr_by_update_step[update_step]
            row = {
                "denominator_parameter_count": parameter_count,
                "interpolation": "direct from per-step tensor_norm_history",
                "logged_step_interval": 1,
                "norm_scope": (
                    "transformer.h non-gamma weights + transformer.wte total Frobenius norm"
                ),
                "pre_update_step": pre_update_step,
                "reference_block_lr": reference_muon_lr,
                "reference_experiment": args.reference_experiment,
                "reference_lrnorm_denominator_total_fro_norm": denominator,
                "reference_muon_lr": reference_muon_lr,
                "reference_optimizer": "MuonW",
                "source_run_dir": str(run_dir),
                "target_lr_over_norm": reference_muon_lr / denominator,
                "update_step": update_step,
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            rows_written += 1

    if missing_norm_steps:
        preview = ", ".join(str(x) for x in missing_norm_steps[:10])
        raise ValueError(
            f"missing denominator norms for {len(missing_norm_steps)} pre-update steps: {preview}"
        )
    if bad_count_steps:
        preview = ", ".join(f"{step}:{count}" for step, count in bad_count_steps[:10])
        raise ValueError(
            f"denominator parameter count mismatch; expected {expected_count}, got {preview}"
        )
    if rows_written == 0:
        raise ValueError("wrote zero reference rows")

    print(f"wrote {rows_written} rows to {output_path}")
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed base run log directory.")
    parser.add_argument("--output", required=True, help="Output reference JSONL path.")
    parser.add_argument(
        "--reference-experiment",
        required=True,
        help="Reference experiment label to write into each JSONL row.",
    )
    return parser.parse_args()


def main() -> None:
    build_reference(parse_args())


if __name__ == "__main__":
    main()
