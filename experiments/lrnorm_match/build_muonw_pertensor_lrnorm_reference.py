#!/usr/bin/env python3
"""Build per-tensor MuonW lr/norm reference JSONL from a completed base run log."""

from __future__ import annotations

import argparse
import bisect
import gzip
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


def load_expected_names(run_dir: Path) -> list[str] | None:
    metadata_path = run_dir / "tensor_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return sorted(
        canonical_name(record["name"])
        for record in metadata
        if is_denominator_parameter(record["name"])
    )


def load_tensor_norms(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    tensor_history = run_dir / "tensor_norm_history.jsonl"
    if not tensor_history.exists():
        raise FileNotFoundError(f"missing {tensor_history}")

    by_name: dict[str, list[tuple[int, float]]] = {}
    for record in iter_jsonl(tensor_history):
        name = record.get("name")
        if not isinstance(name, str) or not is_denominator_parameter(name):
            continue
        by_name.setdefault(canonical_name(name), []).append(
            (int(record["step"]), float(record["fro_norm"]))
        )

    if not by_name:
        raise ValueError(f"found no denominator parameter norms in {tensor_history}")
    prepared = {}
    for name, series in by_name.items():
        series.sort()
        prepared[name] = (
            [step for step, _ in series],
            [value for _, value in series],
        )
    return prepared


def interpolate(series: tuple[list[int], list[float]], step: int) -> tuple[float, str]:
    steps, values = series
    pos = bisect.bisect_left(steps, step)
    if pos < len(steps) and steps[pos] == step:
        return values[pos], "direct"
    if pos == 0 or pos == len(steps):
        raise ValueError(f"cannot interpolate step={step}; available range {steps[0]}..{steps[-1]}")
    left_step, left_value = steps[pos - 1], values[pos - 1]
    right_step, right_value = steps[pos], values[pos]
    alpha = (step - left_step) / (right_step - left_step)
    return left_value + alpha * (right_value - left_value), "linear"


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


def load_reference_lr_by_update_step(path: Path) -> dict[int, float]:
    lr_by_step: dict[int, float] = {}
    for record in iter_jsonl(path):
        update_step = int(record["update_step"])
        lr_by_step[update_step] = float(record.get("reference_muon_lr", record["reference_block_lr"]))
    if not lr_by_step:
        raise ValueError(f"found no reference LR rows in {path}")
    return lr_by_step


def build_reference(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir does not exist: {run_dir}")

    tensor_norms = load_tensor_norms(run_dir)
    expected_names = load_expected_names(run_dir)
    if expected_names is not None and sorted(tensor_norms) != expected_names:
        missing = sorted(set(expected_names) - set(tensor_norms))
        extra = sorted(set(tensor_norms) - set(expected_names))
        raise ValueError(f"tensor norm names mismatch; missing={missing[:5]} extra={extra[:5]}")

    if args.step_reference_json:
        lr_by_update_step = load_reference_lr_by_update_step(Path(args.step_reference_json))
    else:
        lr_by_update_step = load_muon_lr_by_update_step(run_dir)
    embed_to_muon_lr_ratio = float(args.embed_lr) / float(args.muon_lr)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    interpolation_modes_seen: set[str] = set()
    output_open = gzip.open if output_path.suffix == ".gz" else output_path.open
    output_mode = "wt" if output_path.suffix == ".gz" else "w"
    with output_open(output_path, output_mode, encoding="utf-8", newline="\n") as f:
        for update_step in sorted(lr_by_update_step):
            pre_update_step = update_step - 1
            reference_muon_lr = lr_by_update_step[update_step]
            reference_embed_lr = embed_to_muon_lr_ratio * reference_muon_lr
            tensor_norm_by_name: dict[str, float] = {}
            target_lr_over_norm_by_name: dict[str, float] = {}
            total_sq = 0.0
            for name, series in tensor_norms.items():
                norm, interpolation_mode = interpolate(series, pre_update_step)
                interpolation_modes_seen.add(interpolation_mode)
                tensor_norm_by_name[name] = norm
                total_sq += norm * norm
                reference_lr = reference_embed_lr if name == "transformer.wte.weight" else reference_muon_lr
                target_lr_over_norm_by_name[name] = reference_lr / norm

            row = {
                "denominator_parameter_count": len(tensor_norm_by_name),
                "embed_to_muon_lr_ratio": embed_to_muon_lr_ratio,
                "interpolation": (
                    "direct from tensor_norm_history"
                    if interpolation_modes_seen == {"direct"}
                    else "linear from tensor_norm_history"
                ),
                "norm_scope": (
                    "per tensor Frobenius norms for transformer.h non-gamma weights and transformer.wte"
                ),
                "pre_update_step": pre_update_step,
                "reference_block_lr": reference_muon_lr,
                "reference_embed_lr": reference_embed_lr,
                "reference_experiment": args.reference_experiment,
                "reference_lrnorm_denominator_total_fro_norm": math.sqrt(total_sq),
                "reference_muon_lr": reference_muon_lr,
                "reference_optimizer": "MuonW",
                "reference_tensor_fro_norms": tensor_norm_by_name,
                "source_run_dir": args.source_run_dir_label or str(run_dir),
                "target_lr_over_tensor_norms": target_lr_over_norm_by_name,
                "update_step": update_step,
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            rows_written += 1

    if rows_written == 0:
        raise ValueError("wrote zero reference rows")
    print(f"wrote {rows_written} rows to {output_path}")
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed base run log directory.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output reference path (.jsonl or transparently compressed .jsonl.gz).",
    )
    parser.add_argument(
        "--reference-experiment",
        required=True,
        help="Reference experiment label to write into each JSONL row.",
    )
    parser.add_argument(
        "--source-run-dir-label",
        default=None,
        help="Optional stable provenance label instead of the local staging directory.",
    )
    parser.add_argument("--embed-lr", type=float, default=0.0036)
    parser.add_argument("--muon-lr", type=float, default=0.00036)
    parser.add_argument(
        "--step-reference-json",
        default=None,
        help="Optional totalnorm reference JSONL that supplies the complete update_step/reference LR grid.",
    )
    return parser.parse_args()


def main() -> None:
    build_reference(parse_args())


if __name__ == "__main__":
    main()
